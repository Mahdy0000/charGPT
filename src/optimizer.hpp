#pragma once
#include <cmath>
#include "cuda_kernels.cuh"

// optimizer.hpp
//
// Implements the Adam (Adaptive Moment Estimation) optimizer with all state living
// entirely in GPU device memory.
//
// Standard Adam maintains two exponential moving averages for every parameter. The first
// moment m tracks the mean of recent gradients, providing momentum. The second moment v
// tracks the mean of recent squared gradients, providing an estimate of each parameter's
// gradient volatility. A parameter with consistently large gradients receives a smaller
// effective step than one with small gradients, which allows all parts of the model to
// learn at an appropriate rate simultaneously.
//
// The update rule for parameter i at timestep t:
//   m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
//   v[i] = beta2 * v[i] + (1 - beta2) * grad[i]^2
//   lr_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)   bias correction
//   params[i] -= lr_t * m[i] / (sqrt(v[i]) + eps)
//
// The bias correction term compensates for the fact that m and v are initialised to zero
// and therefore underestimate their true values in the first few steps. Without it, the
// first optimizer step would be far smaller than intended.
//
// In this implementation d_m and d_v are raw device pointers. The step function passes
// them directly to cuda_adam_update, which executes the complete update for all 10.7
// million parameters in a single GPU kernel dispatch. No data crosses the PCIe bus
// during the optimizer step.

struct AdamOptimizer {
    float lr;           // base learning rate, overwritten each step by the scheduler
    float beta1;        // decay rate for the gradient mean, typically 0.9
    float beta2;        // decay rate for the gradient variance, typically 0.999
    float eps;          // added to sqrt(v) to prevent division by zero
    int   t;            // timestep counter, incremented by step()
    int   param_count;  // total number of parameters, length of d_m and d_v

    float* d_m = nullptr;  // first moment vector, lives in GPU memory
    float* d_v = nullptr;  // second moment vector, lives in GPU memory

    // Default constructor. Produces an empty optimizer not yet connected to any model.
    AdamOptimizer()
        : lr(3e-4f), beta1(0.9f), beta2(0.999f), eps(1e-8f), t(0), param_count(0) {}

    // Allocating constructor. Reserves GPU memory for n first and second moment values and
    // zero-initialises both vectors. cuda_alloc calls cudaMemset internally, which is
    // required because the bias correction formula assumes m and v start at exactly zero.
    AdamOptimizer(int n, float lr = 3e-4f,
                  float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
        : lr(lr), beta1(beta1), beta2(beta2), eps(eps), t(0), param_count(n),
          d_m(cuda_alloc(n)),   // zero-initialised first moment
          d_v(cuda_alloc(n)) {} // zero-initialised second moment

    // Destructor frees the two GPU moment allocations when the optimizer goes out of scope.
    ~AdamOptimizer() {
        cuda_free(d_m);
        cuda_free(d_v);
    }

    // The optimizer owns two GPU allocations and cannot be copied without deep copying
    // device memory and resetting the timestep counter. Deletion is safer than an
    // incorrect shallow copy.
    AdamOptimizer(const AdamOptimizer&)            = delete;
    AdamOptimizer& operator=(const AdamOptimizer&) = delete;

    // Performs one Adam update step. d_params and d_grads are flat, contiguous GPU arrays
    // of length param_count. The caller is responsible for ensuring they contain the
    // concatenated parameters and averaged gradients from the current optimizer step.
    //
    // The bias-corrected learning rate lr_t is computed here on the CPU with two floating-
    // point operations. Doing this once in the calling thread is far cheaper than computing
    // it inside the GPU kernel for each of the 10.7 million parameters independently.
    // The resulting scalar is then passed to cuda_adam_update as a kernel argument.
    void step(float* d_params, const float* d_grads) {
        ++t;
        float lr_t = lr
            * std::sqrt(1.0f - std::pow(beta2, (float)t))
            / (1.0f - std::pow(beta1, (float)t));
        cuda_adam_update(d_params, d_grads, d_m, d_v,
                         param_count, lr_t, beta1, beta2, eps);
    }
};
