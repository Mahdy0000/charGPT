#pragma once

// cuda_kernels.cuh
//
// This header declares every CUDA wrapper function that is callable from ordinary C++
// translation units (.cpp files). The implementations live in cuda_kernels.cu, which
// is compiled by nvcc. Keeping declarations here and definitions in the .cu file lets
// the C++ compiler build matrix.hpp, model.hpp, optimizer.hpp, main.cpp, and chat.cpp
// without needing to understand CUDA syntax.
//
// All pointer arguments are device pointers unless stated otherwise. Callers must not
// pass host (CPU) pointers to these functions.


// Matrix multiply: C = A * B
// A has shape (M x K), B has shape (K x N), C has shape (M x N).
// Dispatches to cuBLAS cublasSgemm. Because cuBLAS uses column-major storage and our
// matrices are row-major, the call is expressed as C^T = B^T * A^T, which produces the
// correct row-major result without any transposition overhead.
void cuda_matmul(const float* A, const float* B, float* C,
                 int M, int K, int N);

// C = A^T * B
// A has shape (K x M) stored row-major, treated as its transpose (M x K). B has shape
// (K x N). C has shape (M x N). Used in backward passes to compute weight gradients of
// the form X^T * d_out.
void cuda_matmul_atb(const float* A, const float* B, float* C,
                     int M, int K, int N);

// C = A * B^T
// A has shape (M x K). B has shape (N x K) and is transposed to (K x N). C has shape
// (M x N). Used to multiply by weight matrix transposes such as d_scores * K and
// d_out * W^T in the attention backward pass.
void cuda_matmul_abt(const float* A, const float* B, float* C,
                     int M, int K, int N);

// Row-wise softmax applied in place across a (rows x cols) matrix.
// Each row is treated as one probability distribution. The kernel uses shared-memory
// parallel reduction to find the row maximum before subtracting it from the exponents,
// preventing overflow when logit values are large. The number of threads is chosen as
// the smallest power of two that covers the column count, up to 1024.
void cuda_softmax(float* data, int rows, int cols);

// Element-wise addition: C[i] = A[i] + B[i] for all i in [0, n).
// All three arrays must have at least n elements. Used for residual connections and
// gradient accumulation across micro-batches.
void cuda_add(const float* A, const float* B, float* C, int n);

// Element-wise scalar multiplication: C[i] = A[i] * s for all i in [0, n).
// Used to average accumulated gradients after gradient accumulation and to apply the
// gradient clipping scale factor.
void cuda_scale(const float* A, float s, float* C, int n);

// GELU activation applied element-wise: out[i] = gelu(in[i]).
// Uses the tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
// Applied to the hidden layer of the feed-forward network after the first linear projection.
void cuda_gelu(const float* in, float* out, int n);

// Backward pass through GELU: out[i] = d_out[i] * gelu'(x[i]).
// The derivative of the tanh approximation is computed analytically. x holds the
// pre-activation values saved during the forward pass. d_out is the gradient arriving
// from the next layer.
void cuda_gelu_backward(const float* d_out, const float* x, float* out, int n);

// Adds a bias vector of length cols to every row of a (rows x cols) matrix.
// out[t][j] = in[t][j] + bias[j] for every token t and dimension j. Applied after both
// linear projections in the feed-forward network.
void cuda_add_bias(const float* in, const float* bias, float* out,
                   int rows, int cols);

// Applies the causal mask to an attention score matrix of shape (T x T).
// Sets scores[i][j] = -1e9 for all j > i. After softmax, these positions receive
// essentially zero weight, ensuring that position i cannot attend to any future position.
void cuda_causal_mask(float* data, int T);

// Zeroes n floats starting at data using cudaMemset.
void cuda_zero(float* data, int n);

// Device-to-device memory copy: copies n floats from src to dst.
// Both pointers must be device pointers. This is the primitive used by gather_params_gpu
// and gather_grads_gpu to pack weight matrices into a flat contiguous buffer without any
// PCIe transfer.
void cuda_copy(const float* src, float* dst, int n);

// Adam optimizer update kernel. All six pointer arguments are device pointers.
// Each GPU thread handles one parameter independently, reading its gradient from d_grads,
// updating its first moment m and second moment v, and writing the new parameter value.
// The caller pre-computes the bias-corrected learning rate lr_t on the CPU as:
//   lr_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
// This is one multiply and two divisions rather than 10.7 million.
void cuda_adam_update(float* params, const float* grads, float* m, float* v,
                      int n, float lr_t, float beta1, float beta2, float eps);

// Allocates n floats on the GPU and zero-initialises them.
// Returns a device pointer. All Matrix objects and AdamOptimizer moment vectors call this
// function. The zero-initialisation is important for Adam: the moment vectors m and v must
// start at zero for the bias correction formula to be valid.
float* cuda_alloc(int n);

// Frees a GPU allocation previously returned by cuda_alloc.
// Guards against null pointers so callers do not need to check before calling.
void cuda_free(float* ptr);

// Copies n floats from the CPU host buffer h_ptr to the GPU device buffer d_ptr.
void cuda_to_device(float* d_ptr, const float* h_ptr, int n);

// Copies n floats from the GPU device buffer d_ptr to the CPU host buffer h_ptr.
// This is a synchronous transfer that blocks until the GPU has finished writing.
void cuda_to_host(float* h_ptr, const float* d_ptr, int n);
