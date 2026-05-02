#include "cuda_kernels.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// CUDA_CHECK wraps any CUDA runtime call and terminates the program with a descriptive
// error message if it fails. Without this, failed allocations and kernel launches return
// silently and corrupt the GPU state in ways that are difficult to diagnose later. The
// do-while(0) pattern allows the macro to be used safely after an if without braces.
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d -- %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// The cuBLAS library provides highly optimised matrix multiplication routines tuned for
// each GPU architecture. A single handle manages the internal state of the library.
// It is created on the first use and reused for every subsequent call, which avoids the
// overhead of repeated initialisation.
static cublasHandle_t cublas_handle = nullptr;

static void init_cublas() {
    if (!cublas_handle) {
        cublasCreate(&cublas_handle);
    }
}


// Matrix multiply: C = A * B
// A is (M x K) and B is (K x N), both stored row-major. C receives the result (M x N).
// cuBLAS assumes column-major layout, so calling it naively with row-major data would
// produce the transpose of the desired result. The standard fix is to request
// C^T = B^T * A^T instead, which cuBLAS interprets as column-major B times column-major A
// and writes the correct row-major C into the output buffer.
void cuda_matmul(const float* A, const float* B, float* C,
                 int M, int K, int N) {
    init_cublas();
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, B, N, A, K, &beta, C, N);
}

// C = A^T * B
// A has shape (K x M) stored row-major, so its transpose is (M x K). B is (K x N).
// The result C is (M x N). This form is used to compute weight gradients where the
// input is transposed: for example, dW = X^T * d_out in every linear layer.
void cuda_matmul_atb(const float* A, const float* B, float* C,
                     int M, int K, int N) {
    init_cublas();
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                N, M, K, &alpha, B, N, A, M, &beta, C, N);
}

// C = A * B^T
// A is (M x K) and B is (N x K). Transposing B gives (K x N), so the result is (M x N).
// This form is used when propagating gradients backward through a weight matrix:
// d_input = d_output * W^T.
void cuda_matmul_abt(const float* A, const float* B, float* C,
                     int M, int K, int N) {
    init_cublas();
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K, &alpha, B, K, A, K, &beta, C, N);
}


// Softmax kernel: one CUDA block processes one row of the (rows x cols) matrix.
// The block uses blockDim.x threads, which must be a power of two and at most 1024.
// Shared memory holds one float per thread for the parallel reduction steps.
//
// The algorithm runs in four phases:
//   1. Each thread scans its assigned columns and records the local maximum.
//   2. A parallel tree reduction across shared memory finds the global row maximum.
//      This subtraction prevents exp() from overflowing when logit values are large.
//   3. Each thread computes exp(x - max) for its columns and accumulates a partial sum.
//   4. A second tree reduction finds the total sum, then each thread normalises its values.
//
// Using shared memory for the reduction avoids global memory round trips and keeps the
// entire computation in registers and L1 cache.
__global__ void softmax_kernel(float* data, int rows, int cols) {
    extern __shared__ float smem[];  // one float per thread, allocated at launch
    int row = blockIdx.x;
    if (row >= rows) return;
    float* r = data + row * cols;  // pointer to the start of this row
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // Phase 1: each thread walks its assigned columns in strides of bdim and tracks
    // the largest value it encounters. Starting at -1e30 ensures any real value wins.
    float thread_max = -1e30f;
    for (int j = tid; j < cols; j += bdim)
        thread_max = fmaxf(thread_max, r[j]);
    smem[tid] = thread_max;
    __syncthreads();

    // Phase 2: binary tree reduction. At each step the active thread count halves.
    // After log2(bdim) steps, smem[0] holds the maximum across the entire row.
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float max_val = smem[0];
    __syncthreads();

    // Phase 3: subtract the row maximum before exponentiating to prevent overflow.
    // Each thread computes exp(x - max) in place and accumulates its partial sum.
    float thread_sum = 0.0f;
    for (int j = tid; j < cols; j += bdim) {
        r[j] = expf(r[j] - max_val);
        thread_sum += r[j];
    }
    smem[tid] = thread_sum;
    __syncthreads();

    // Phase 4a: tree reduction to find the total sum across the row.
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float sum = smem[0];
    __syncthreads();

    // Phase 4b: divide each element by the sum to produce a proper probability distribution.
    for (int j = tid; j < cols; j += bdim)
        r[j] /= sum;
}

void cuda_softmax(float* data, int rows, int cols) {
    // Find the smallest power of two that covers all columns. This ensures every column
    // is processed and that the binary reduction has a clean power-of-two thread count.
    int threads = 1;
    while (threads < cols && threads < 1024) threads <<= 1;
    size_t smem_bytes = threads * sizeof(float);
    softmax_kernel<<<rows, threads, smem_bytes>>>(data, rows, cols);
}


// Element-wise addition kernel. Each thread handles one element, identified by its
// global index computed from the block and thread indices.
__global__ void add_kernel(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

void cuda_add(const float* A, const float* B, float* C, int n) {
    add_kernel<<<(n + 255) / 256, 256>>>(A, B, C, n);
}

// Element-wise scalar multiply kernel. Each thread scales one element by s.
__global__ void scale_kernel(const float* A, float s, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] * s;
}

void cuda_scale(const float* A, float s, float* C, int n) {
    scale_kernel<<<(n + 255) / 256, 256>>>(A, s, C, n);
}


// GELU activation: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// The constant 0.7978845608 is sqrt(2/pi). This tanh approximation closely matches the
// exact GELU (which requires the Gaussian CDF) while being fast to compute in hardware.
// Unlike ReLU, GELU is smooth near zero and allows small negative values to pass through
// with reduced magnitude, which prevents dead neurons and improves gradient flow.
__device__ float gelu_device(float x) {
    const float sqrt2overpi = 0.7978845608f;
    return 0.5f * x * (1.0f + tanhf(sqrt2overpi * (x + 0.044715f * x * x * x)));
}

// Derivative of GELU with respect to x, needed for the backward pass.
// The derivative of tanh(u) with respect to u is sech^2(u) = 1 - tanh^2(u).
// The chain rule then gives: GELU'(x) = 0.5*(1 + tanh(u)) + 0.5*x*sech^2(u)*u'
// where u = sqrt(2/pi)*(x + 0.044715*x^3) and u' = sqrt(2/pi)*(1 + 3*0.044715*x^2).
__device__ float gelu_backward_device(float x) {
    const float sqrt2overpi = 0.7978845608f;
    float cube = x * x * x;
    float inner = sqrt2overpi * (x + 0.044715f * cube);
    float tanh_val = tanhf(inner);
    float sech2 = 1.0f - tanh_val * tanh_val;
    float inner_deriv = sqrt2overpi * (1.0f + 3.0f * 0.044715f * x * x);
    return 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * inner_deriv;
}

__global__ void gelu_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = gelu_device(in[i]);
}

void cuda_gelu(const float* in, float* out, int n) {
    gelu_kernel<<<(n + 255) / 256, 256>>>(in, out, n);
}

// Backward pass through GELU: out[i] = d_out[i] * GELU'(x[i]).
// d_out carries the gradient arriving from the layer above. x holds the pre-activation
// values saved during the forward pass.
__global__ void gelu_backward_kernel(const float* d_out, const float* x,
                                     float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = d_out[i] * gelu_backward_device(x[i]);
}

void cuda_gelu_backward(const float* d_out, const float* x, float* out, int n) {
    gelu_backward_kernel<<<(n + 255) / 256, 256>>>(d_out, x, out, n);
}


// Add-bias kernel. The matrix is laid out in row-major order as a flat array of
// rows * cols elements. The column index j = idx % cols selects the correct bias entry
// for each element, so every row receives the same bias regardless of its position.
__global__ void add_bias_kernel(const float* in, const float* bias,
                                float* out, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int j = idx % cols;  // column index selects the bias entry
        out[idx] = in[idx] + bias[j];
    }
}

void cuda_add_bias(const float* in, const float* bias, float* out,
                   int rows, int cols) {
    int n = rows * cols;
    add_bias_kernel<<<(n + 255) / 256, 256>>>(in, bias, out, rows, cols);
}


// Causal mask kernel. The attention score matrix is (T x T) stored row-major. Thread i
// owns element idx = row * T + col. If col > row, that element represents a future
// position and is set to -1e9. After softmax, exp(-1e9) is effectively zero, so the
// model assigns no probability to attending to tokens it has not yet seen.
__global__ void causal_mask_kernel(float* data, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * T;
    if (idx < total) {
        int i = idx / T, j = idx % T;  // row i, column j
        if (j > i) data[idx] = -1e9f;  // future position: zero it out after softmax
    }
}

void cuda_causal_mask(float* data, int T) {
    int n = T * T;
    causal_mask_kernel<<<(n + 255) / 256, 256>>>(data, T);
}


// Zero-fills n floats using cudaMemset. Used at the start of each micro-batch to clear
// gradient accumulators before the backward pass writes into them.
void cuda_zero(float* data, int n) {
    cudaMemset(data, 0, n * sizeof(float));
}

// Device-to-device copy. Both src and dst must be device pointers. This is the primitive
// used by gather_params_gpu and gather_grads_gpu to pack scattered weight matrices into
// a contiguous flat buffer without any PCIe transfer.
void cuda_copy(const float* src, float* dst, int n) {
    cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice);
}

// Allocates n floats of GPU memory and zero-initialises them. The zero-initialisation
// is important for the Adam moment vectors: m and v must both start at zero for the bias
// correction formula to produce correct results at step t = 1.
float* cuda_alloc(int n) {
    float* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(float)));
    return ptr;
}

// Frees a GPU allocation. The null guard allows callers to call cuda_free unconditionally
// without checking whether the pointer was ever allocated.
void cuda_free(float* ptr) {
    if (ptr) cudaFree(ptr);
}

// Host-to-device transfer. Copies n floats from the CPU array h_ptr to the GPU array
// d_ptr. Used when uploading gradient tensors computed on the CPU, such as d_logits.
void cuda_to_device(float* d_ptr, const float* h_ptr, int n) {
    CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, n * sizeof(float), cudaMemcpyHostToDevice));
}

// Device-to-host transfer. Copies n floats from the GPU array d_ptr to the CPU array
// h_ptr. This is a synchronous call that blocks until the GPU finishes writing. Used
// when the CPU needs to compute the gradient norm or the softmax probabilities.
void cuda_to_host(float* h_ptr, const float* d_ptr, int n) {
    CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
}


// Adam update kernel. Each thread handles one parameter independently, reading from and
// writing to global GPU memory. There are no data dependencies between threads, so no
// synchronisation is needed. The full 10.7 million parameter update completes in a single
// kernel dispatch.
//
// The update rule for parameter i at timestep t is:
//   m[i] = beta1 * m[i] + (1 - beta1) * grad[i]      exponential moving average of gradient
//   v[i] = beta2 * v[i] + (1 - beta2) * grad[i]^2    exponential moving average of squared gradient
//   params[i] -= lr_t * m[i] / (sqrt(v[i]) + eps)    bias-corrected weight update
//
// The bias-corrected learning rate lr_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t) is
// computed once on the CPU and passed as a scalar. This saves 10.7 million division
// operations from the kernel and moves them to a single CPU computation.
__global__ void adam_update_kernel(float* params, const float* grads,
                                   float* m, float* v,
                                   int n, float lr_t,
                                   float beta1, float beta2, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g  = grads[i];
    float mi = beta1 * m[i] + (1.0f - beta1) * g;         // update first moment
    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;     // update second moment
    m[i] = mi;
    v[i] = vi;
    params[i] -= lr_t * mi / (sqrtf(vi) + eps);           // apply bias-corrected update
}

void cuda_adam_update(float* params, const float* grads, float* m, float* v,
                      int n, float lr_t, float beta1, float beta2, float eps) {
    adam_update_kernel<<<(n + 255) / 256, 256>>>(
        params, grads, m, v, n, lr_t, beta1, beta2, eps);
}
