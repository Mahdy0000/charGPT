#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <cassert>
#include "cuda_kernels.cuh"

// matrix.hpp
//
// Defines the Matrix struct, which is the fundamental data container for this project.
// Every weight tensor, activation, gradient, and intermediate result is a Matrix.
//
// The primary storage is d_data, a raw float pointer allocated in GPU device memory via
// cudaMalloc. The CPU is only involved when data must be read (to_cpu) or written
// (from_cpu) explicitly. All arithmetic operations call CUDA kernels directly on d_data,
// so a chain of matrix operations never leaves the GPU.
//
// Memory safety is handled through RAII: the destructor calls cuda_free, which means
// temporary matrices created during forward and backward passes are freed automatically
// when they go out of scope. Move semantics transfer ownership without copying memory.
// Copy semantics perform a device-to-device cudaMemcpy so that assignments produce
// independent GPU allocations rather than aliased pointers.

struct Matrix {
    int rows = 0, cols = 0;
    float* d_data = nullptr;  // primary storage: lives in GPU device memory

    // Default constructor produces an empty matrix with no GPU allocation.
    Matrix() {}

    // Allocating constructor. Calls cuda_alloc which allocates and zero-initialises
    // r * c floats on the GPU. Zero-initialisation is important for gradient accumulators
    // and bias vectors, which must start at zero.
    Matrix(int r, int c) : rows(r), cols(c) {
        d_data = cuda_alloc(r * c);
    }

    // Destructor. Frees the GPU allocation if one exists. The null check inside
    // cuda_free handles the case where d_data was already transferred away by a move.
    ~Matrix() { if (d_data) cuda_free(d_data); }

    // Move constructor. Transfers ownership of the GPU allocation from the source to
    // this matrix by copying the pointer and then nulling the source. The source's
    // destructor will then do nothing, preventing a double-free.
    Matrix(Matrix&& o) noexcept : rows(o.rows), cols(o.cols), d_data(o.d_data) {
        o.d_data = nullptr; o.rows = 0; o.cols = 0;
    }

    // Move assignment. Frees any existing allocation before taking ownership of the
    // source's memory. The self-assignment check guards against the degenerate case
    // a = std::move(a).
    Matrix& operator=(Matrix&& o) noexcept {
        if (this != &o) {
            if (d_data) cuda_free(d_data);
            rows = o.rows; cols = o.cols; d_data = o.d_data;
            o.d_data = nullptr; o.rows = 0; o.cols = 0;
        }
        return *this;
    }

    // Copy constructor. Allocates fresh GPU memory for this matrix and copies the
    // source data using a device-to-device cudaMemcpy. This produces two independent
    // GPU allocations that can be modified separately, which is required when saving
    // activations for the backward pass.
    Matrix(const Matrix& o) : rows(o.rows), cols(o.cols) {
        if (o.d_data) {
            d_data = cuda_alloc(rows * cols);
            cuda_copy(o.d_data, d_data, rows * cols);
        }
    }

    // Copy assignment. If the existing allocation is the same size, it is reused to
    // avoid a free and re-alloc. If the size differs, the old allocation is freed and
    // a new one is created.
    Matrix& operator=(const Matrix& o) {
        if (this != &o) {
            if (d_data && rows * cols != o.rows * o.cols) {
                cuda_free(d_data); d_data = nullptr;
            }
            rows = o.rows; cols = o.cols;
            if (o.d_data) {
                if (!d_data) d_data = cuda_alloc(rows * cols);
                cuda_copy(o.d_data, d_data, rows * cols);
            }
        }
        return *this;
    }

    // Explicit GPU memory release. Used when a large temporary matrix should be freed
    // before the end of its scope, such as a cached activation that is no longer needed.
    void free_gpu() {
        if (d_data) { cuda_free(d_data); d_data = nullptr; }
    }

    // Total number of elements. Equal to rows * cols.
    int size() const { return rows * cols; }

    // Uploads a CPU vector to the GPU allocation. The vector must have exactly rows * cols
    // elements. If d_data has not been allocated yet (rows and cols set but no GPU memory),
    // a new allocation is created automatically. Used after computing gradients on the CPU.
    void from_cpu(const std::vector<float>& v) {
        assert((int)v.size() == rows * cols);
        if (!d_data) d_data = cuda_alloc(rows * cols);
        cuda_to_device(d_data, v.data(), rows * cols);
    }

    // Downloads the GPU contents to a new CPU vector and returns it. This is a synchronous
    // transfer that stalls the CPU until the GPU finishes writing. Used when the CPU must
    // read activations or logits, such as for the softmax and loss computation.
    std::vector<float> to_cpu() const {
        std::vector<float> v(rows * cols);
        if (d_data) cuda_to_host(v.data(), d_data, rows * cols);
        return v;
    }

    // Initialises the matrix with values drawn from a normal distribution on the CPU,
    // then uploads the result to the GPU. Used during model construction to initialise
    // all weight matrices. The seed makes initialisation reproducible, and each weight
    // matrix receives a different seed to break symmetry between heads and layers.
    void randn(float mean, float std, unsigned int seed) {
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(mean, std);
        std::vector<float> v(rows * cols);
        for (auto& x : v) x = dist(gen);
        from_cpu(v);
    }

    // Sets all GPU elements to zero using cudaMemset. Called at the start of each
    // micro-batch to clear gradient tensors before the backward pass writes into them.
    void zero() {
        if (d_data) cuda_zero(d_data, rows * cols);
    }

    // Matrix multiply: returns a new matrix C = this * B.
    // this has shape (rows x cols) and B has shape (cols x B.cols). C has shape (rows x
    // B.cols). Dispatches to cuda_matmul which calls cuBLAS cublasSgemm. The result C
    // is allocated on the GPU and returned by value; the move constructor ensures no copy.
    Matrix matmul(const Matrix& B) const {
        Matrix C(rows, B.cols);
        cuda_matmul(d_data, B.d_data, C.d_data, rows, cols, B.cols);
        return C;
    }

    // Returns the transpose of this matrix as a new GPU allocation.
    // The transposition is performed on the CPU because this avoids writing a custom
    // CUDA transpose kernel. For moderate matrix sizes (up to 384 x 384) the transfer
    // overhead is acceptable. The resulting matrix has shape (cols x rows).
    Matrix T() const {
        auto v = to_cpu();
        std::vector<float> vt(rows * cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                vt[j * rows + i] = v[i * cols + j];  // row-major to column-major swap
        Matrix out(cols, rows);
        out.from_cpu(vt);
        return out;
    }

    // Row-wise softmax applied in place. Returns a new matrix with the same shape
    // where each row sums to exactly one. The CUDA kernel uses shared-memory parallel
    // reduction to find the row maximum before computing exp(), preventing overflow.
    Matrix softmax() const {
        Matrix out(*this);  // copies GPU memory via the copy constructor
        cuda_softmax(out.d_data, rows, cols);
        return out;
    }

    // Element-wise addition: returns a new matrix C where C[i] = this[i] + B[i].
    // Requires both matrices to have the same shape. Used for residual connections
    // where the sub-layer output is added back to the block's input.
    Matrix operator+(const Matrix& B) const {
        Matrix C(rows, cols);
        cuda_add(d_data, B.d_data, C.d_data, rows * cols);
        return C;
    }

    // In-place element-wise addition. Writes the result back into this matrix's GPU
    // allocation. Used when accumulating gradient contributions from multiple attention
    // heads into a single d_X matrix.
    Matrix& operator+=(const Matrix& B) {
        cuda_add(d_data, B.d_data, d_data, rows * cols);
        return *this;
    }

    // Scalar multiplication: returns a new matrix where every element is multiplied by s.
    // Used to scale the attention scores by 1/sqrt(d_k) and to apply the gradient
    // clipping factor to the flat gradient buffer.
    Matrix operator*(float s) const {
        Matrix C(rows, cols);
        cuda_scale(d_data, s, C.d_data, rows * cols);
        return C;
    }
};
