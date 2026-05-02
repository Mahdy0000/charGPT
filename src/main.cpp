// main.cpp
//
// Training entry point for the GPT character-level language model.
//
// The training loop uses gradient accumulation to simulate a large effective batch size
// without holding multiple sequences in VRAM simultaneously. Each optimizer step runs
// accum_steps micro-batches. After each micro-batch the gradients are summed on the GPU
// with cuda_add. After all micro-batches the accumulated buffer is divided by accum_steps
// with cuda_scale, clipped by global norm, and passed to the GPU-resident Adam optimizer.
// The parameter update is performed entirely on the GPU with no PCIe round trips.
//
// The learning rate follows a warmup-then-cosine-decay schedule. For the first
// warmup_steps optimizer steps the rate increases linearly from zero to max_lr. After
// that it follows a cosine curve from max_lr down to min_lr, providing large steps early
// in training when the weights are far from a good solution and smaller steps as they
// converge.

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>

#include "model.hpp"
#include "optimizer.hpp"

// Returns the learning rate for the given optimizer step.
// During warmup the rate grows linearly: lr = max_lr * (step + 1) / warmup_steps.
// After warmup a cosine decay schedule applies:
//   coeff = 0.5 * (1 + cos(pi * decay_ratio))
//   lr    = min_lr + coeff * (max_lr - min_lr)
// where decay_ratio runs from 0 at the end of warmup to 1 at max_steps.
float get_lr(int step, float max_lr, float min_lr, int warmup, int max_steps) {
    if (step < warmup) return max_lr * ((float)(step+1) / warmup);
    if (step >= max_steps) return min_lr;
    float r = (float)(step - warmup) / (float)(max_steps - warmup);
    return min_lr + 0.5f * (1.0f + std::cos(3.14159265f * r)) * (max_lr - min_lr);
}

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "  LLM in C++ -- GPU Training (CUDA)" << std::endl;
    std::cout << "============================================" << std::endl;

    // Load the raw text from disk. The entire file is read into a std::string at once.
    // For the Shakespeare dataset this is approximately 1.1 million characters.
    std::cout << "\n[1/5] Loading data..." << std::endl;
    std::ifstream file("data/input.txt");
    if (!file.is_open()) { std::cerr << "ERROR: data/input.txt not found" << std::endl; return 1; }
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    std::cout << "  " << text.size() << " characters loaded." << std::endl;

    // Build the character vocabulary from the training text and encode the entire text
    // as a sequence of integer token IDs. This encoded sequence is the training dataset.
    // For Shakespeare the vocabulary contains 65 characters.
    std::cout << "\n[2/5] Building char tokenizer..." << std::endl;
    CharTokenizer tokenizer;
    tokenizer.build(text);
    std::vector<int> data = tokenizer.encode(text);
    std::cout << "  Vocab: " << tokenizer.vocab_size << " | Tokens: " << data.size() << std::endl;

    // Model hyperparameters. These match the nanoGPT Shakespeare configuration.
    // d_ff = 4 * d_model is the standard feed-forward expansion ratio from the original
    // Transformer paper. block_size is the maximum context window in tokens.
    const int d_model    = 384;
    const int n_layers   = 6;
    const int n_heads    = 6;
    const int d_ff       = 1536;   // 4 * d_model
    const int block_size = 256;
    const int vocab_size = tokenizer.vocab_size;

    // Training hyperparameters.
    // accum_steps = 32 means each optimizer step sees 32 * 256 = 8,192 tokens, which
    // is the effective batch size. The higher max_lr relative to a single-sequence
    // baseline is appropriate because larger effective batches produce lower-variance
    // gradient estimates that can safely accept a larger step.
    const int   max_iters    = 5000;   // number of optimizer steps
    const int   accum_steps  = 32;     // micro-batches accumulated before each step
    const float max_lr       = 6e-4f;  // peak learning rate after warmup
    const float min_lr       = 1e-5f;  // floor reached at the end of cosine decay
    const int   warmup_steps = 200;    // steps over which lr grows from zero to max_lr
    const float grad_clip    = 1.0f;   // maximum global L2 norm of the gradient

    // Construct the model. All weight matrices are allocated on the GPU inside the GPT
    // constructor and initialised with small random values using Xavier-style scaling.
    std::cout << "\n[3/5] Creating model (GPU)..." << std::endl;
    GPT model(vocab_size, d_model, n_layers, n_heads, d_ff, block_size);
    int total_params = model.param_count();
    std::cout << "  " << n_layers << " layers, " << n_heads << " heads, d=" << d_model << std::endl;
    std::cout << "  Context: " << block_size << " tokens" << std::endl;
    std::cout << "  Parameters: " << total_params << " (~" << std::fixed
              << std::setprecision(1) << total_params / 1e6f << "M)" << std::endl;

    // Construct the optimizer. d_m and d_v are allocated on the GPU and zero-initialised.
    // No moment data crosses the PCIe bus at any point during training.
    std::cout << "\n[4/5] Adam optimizer (GPU)..." << std::endl;
    AdamOptimizer optimizer(total_params, max_lr);

    std::cout << "\n[5/5] Training (" << max_iters << " steps x "
              << accum_steps << " accum = "
              << max_iters * accum_steps << " micro-batches)..." << std::endl;
    std::cout << "--------------------------------------------" << std::endl;

    std::mt19937 rng(42);  // fixed seed for reproducible sequence sampling
    auto t0 = std::chrono::high_resolution_clock::now();
    float smooth_loss = 0;

    for (int iter = 0; iter < max_iters; ++iter) {
        auto it0 = std::chrono::high_resolution_clock::now();
        float lr = get_lr(iter, max_lr, min_lr, warmup_steps, max_iters);
        optimizer.lr = lr;  // the scheduler writes into the optimizer directly

        // Allocate a flat GPU buffer to accumulate gradients across micro-batches.
        // It starts zeroed. After each micro-batch, gather_grads_gpu collects all
        // gradients into a temporary flat buffer and cuda_add sums it into accum_grads.
        Matrix accum_grads(1, total_params);
        accum_grads.zero();
        float step_loss = 0;

        for (int micro = 0; micro < accum_steps; ++micro) {
            // Sample a random starting position in the encoded dataset. x is the input
            // sequence and y is the target: y[t] = x[t+1] for every position t. The
            // model is trained to predict the next token at every position simultaneously.
            int start = rng() % ((int)data.size() - block_size - 1);
            std::vector<int> x(data.begin()+start, data.begin()+start+block_size);
            std::vector<int> y(data.begin()+start+1, data.begin()+start+block_size+1);

            model.zero_grad();              // clear all gradient tensors before this pass
            Matrix logits = model.forward(x);  // forward pass: shape (block_size x vocab_size)

            // Cross-entropy loss computation on the CPU.
            // The logits are downloaded from the GPU and softmax is applied manually with
            // the max-subtraction trick to prevent overflow. The loss is the average
            // negative log probability of the correct token across all positions.
            auto logits_cpu = logits.to_cpu();
            std::vector<float> probs_cpu(block_size * vocab_size);
            float loss = 0;
            for (int t = 0; t < block_size; ++t) {
                float mx = -1e30f;
                for (int j = 0; j < vocab_size; ++j) mx = std::max(mx, logits_cpu[t*vocab_size+j]);
                float sum = 0;
                for (int j = 0; j < vocab_size; ++j) {
                    probs_cpu[t*vocab_size+j] = std::exp(logits_cpu[t*vocab_size+j] - mx);
                    sum += probs_cpu[t*vocab_size+j];
                }
                for (int j = 0; j < vocab_size; ++j) probs_cpu[t*vocab_size+j] /= sum;
                loss -= std::log(probs_cpu[t*vocab_size+y[t]] + 1e-9f);  // 1e-9 guards log(0)
            }
            loss /= block_size;   // average over the sequence length
            step_loss += loss;

            // Gradient of the loss with respect to the logits.
            // The combined derivative of softmax and cross-entropy is elegant: for every
            // token position t, d_logits[t][j] = probs[t][j] for all j, with the
            // correct token's position decreased by 1. Dividing by block_size normalises
            // consistently with the loss averaging above.
            std::vector<float> dlogits_cpu(block_size * vocab_size);
            for (int t = 0; t < block_size; ++t) {
                for (int j = 0; j < vocab_size; ++j)
                    dlogits_cpu[t*vocab_size+j] = probs_cpu[t*vocab_size+j] / block_size;
                dlogits_cpu[t*vocab_size+y[t]] -= 1.0f / block_size;  // subtract 1 at correct token
            }
            Matrix d_logits(block_size, vocab_size);
            d_logits.from_cpu(dlogits_cpu);  // upload gradient to GPU for backward pass

            model.backward(d_logits, x);  // backward pass: populates all gradient matrices

            // Gather this micro-batch's gradients into a temporary flat GPU buffer and
            // add them element-wise into accum_grads. Both operations are device-to-device;
            // no data leaves the GPU.
            Matrix micro_grads = model.gather_grads_gpu();
            cuda_add(accum_grads.d_data, micro_grads.d_data,
                     accum_grads.d_data, total_params);
        }

        // Average the accumulated gradients over the micro-batch count. This makes the
        // effective gradient estimate equivalent to a full batch of accum_steps sequences.
        step_loss /= accum_steps;
        cuda_scale(accum_grads.d_data, 1.0f / accum_steps,
                   accum_grads.d_data, total_params);

        // Clip the gradient: download once to compute the global L2 norm, then apply
        // the scale factor on the GPU if the norm exceeds grad_clip. One small PCIe
        // download per optimizer step is unavoidable for the norm computation.
        model.clip_grad_norm_gpu(accum_grads, grad_clip);

        // Gather all model parameters into a flat GPU buffer, run the Adam update on the
        // GPU, and write the updated values back to the individual weight matrices.
        // All three operations are device-to-device and require no CPU involvement.
        Matrix params_gpu = model.gather_params_gpu();
        optimizer.step(params_gpu.d_data, accum_grads.d_data);
        model.scatter_params_gpu(params_gpu);

        auto it1 = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(it1 - it0).count();

        // Exponential moving average of the loss with decay 0.95. This smooths out the
        // noise between individual optimizer steps and makes the trend easier to read.
        smooth_loss = (iter == 0) ? step_loss : 0.95f * smooth_loss + 0.05f * step_loss;

        if (iter % 50 == 0) {
            float elapsed = std::chrono::duration<float>(it1 - t0).count();
            float eta_s = (elapsed / (iter + 1)) * (max_iters - iter - 1);
            std::cout << "  step " << std::setw(5) << iter
                      << " | loss " << std::fixed << std::setprecision(4) << smooth_loss
                      << " | ppl " << std::setprecision(1) << std::exp(smooth_loss)
                      << " | lr " << std::scientific << std::setprecision(2) << lr
                      << " | " << std::fixed << std::setprecision(0) << ms << "ms"
                      << " | " << std::setprecision(1) << elapsed << "s"
                      << " | ETA " << std::setprecision(0) << eta_s / 60 << "m"
                      << std::endl;
        }

        // Print a generated sample every 500 steps to let the user observe qualitative
        // progress. Temperature 0.8 adds some randomness while keeping output coherent.
        if (iter > 0 && iter % 500 == 0) {
            std::cout << "\n  --- Sample at step " << iter << " ---" << std::endl;
            std::vector<int> seed = {tokenizer.stoi['\n']};
            auto gen = model.generate(seed, 300, 0.8f);
            std::cout << "  " << tokenizer.decode(gen).substr(0, 500) << std::endl;
            std::cout << "  -----------------------------------------\n" << std::endl;
        }
    }

    // Save the trained model weights and tokenizer vocabulary to a single binary file.
    // The chat program reads this file and reconstructs the model without access to the
    // training data.
    std::cout << "\nSaving model..." << std::endl;
    model.save("model.bin", tokenizer);

    std::cout << "\n============================================" << std::endl;
    std::cout << "  Training complete!" << std::endl;
    std::cout << "============================================\n" << std::endl;

    // Generate one final sample from a newline seed to show the trained model's output.
    std::vector<int> seed = {tokenizer.stoi['\n']};
    auto gen = model.generate(seed, 500, 0.8f);
    std::cout << tokenizer.decode(gen) << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "\nTotal: " << std::fixed << std::setprecision(1)
              << std::chrono::duration<float>(t1 - t0).count() / 60.0f << " min" << std::endl;
    std::cout << "Run: .\\build\\chat.exe model.bin" << std::endl;
    return 0;
}
