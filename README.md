# Building an LLM in C++
## *From First Principles to GPU-Accelerated Chatbot*

A complete implementation of a GPT-style Transformer trained entirely in C++ and CUDA, with no machine learning frameworks.

| Field | Detail |
|---|---|
| Author | Mahdy |
| Language | C++17 with CUDA |
| Target Hardware | NVIDIA RTX 4080 Laptop GPU, 12 GB VRAM |
| Model Scale | 10.7 million parameters (nanoGPT Shakespeare configuration) |
| Architecture | GPT-style Transformer: Pre-Norm, Multi-Head Attention, GELU, Adam |
| Tokenizer | Character-level, 65 tokens for the Shakespeare dataset |
| Training Speed | Approximately 9 seconds per optimizer step with 32 gradient accumulation micro-batches |


## What This Is

This project implements a Generative Pre-trained Transformer (GPT) language model from scratch. Every component is written by hand: the matrix operations, the attention mechanism, the backpropagation pass, the Adam optimizer, and the CUDA kernels that run all of it on the GPU. Nothing is delegated to PyTorch or TensorFlow.

The model is trained on the complete works of Shakespeare and learns to generate text in his style. After 5,000 optimizer steps the generated output contains coherent character names, dialogue formatting, and Elizabethan phrasing.

The companion guide `LLM_Complete_Guide_Sheet.pdf` explains every formula and every design decision in full, starting from what a neural network is and ending with a working chatbot.


## Architecture

The model follows the GPT-2 Pre-Norm design. The forward pass for a sequence of T tokens is:

```
x[t] = wte[ token_id[t] ] + wpe[ t ]          token embedding + position embedding

for each of 6 Transformer blocks:
    x = x + MultiHeadAttention( LayerNorm(x) ) first residual connection
    x = x + FeedForward( LayerNorm(x) )        second residual connection

logits = LayerNorm(x) * lm_head               shape: (T x 65)
```

Each attention head computes `softmax( Q * K^T / sqrt(d_k) ) * V` where Q, K, V are learned projections of the input and d_k is 64. Six heads run in parallel and their outputs are concatenated and projected back to d_model. A causal mask forces each position to attend only to itself and earlier positions.

The feed-forward network expands each token from 384 to 1,536 dimensions using GELU activation, then compresses back to 384. This is where the model stores factual associations learned during training.

| Hyperparameter | Value |
|---|---|
| Embedding dimension (d_model) | 384 |
| Transformer layers (n_layers) | 6 |
| Attention heads (n_heads) | 6 |
| Head dimension (d_k) | 64 |
| Feed-forward width (d_ff) | 1,536 |
| Context window (block_size) | 256 tokens |
| Vocabulary size | 65 characters |
| Total parameters | 10,786,560 |


## GPU Design

All weight matrices are stored as raw device pointers allocated with `cudaMalloc`. No data lives in CPU memory during training except when explicitly transferred. The training loop eliminates unnecessary PCIe transfers in three ways.

The Adam optimizer stores its moment vectors `m` and `v` in GPU memory. The `cuda_adam_update` kernel performs the complete update for all 10.7 million parameters in a single GPU dispatch. No parameter data crosses the PCIe bus during the optimizer step.

Gradient accumulation uses `cuda_add` to sum micro-batch gradients directly in GPU memory. After 32 micro-batches `cuda_scale` divides the accumulated buffer by 32. The optimizer then reads from that buffer without any host involvement.

Gradient clipping downloads the flat gradient buffer once to compute the global L2 norm, then uses `cuda_scale` to apply the clip ratio on the GPU. This is one small transfer per optimizer step compared to the 42 MB round trip that a CPU-side optimizer would require.


## File Responsibilities

| File | Responsibility |
|---|---|
| `src/matrix.hpp` | The `Matrix` struct. Primary storage is `d_data`, a GPU pointer. All arithmetic methods call CUDA kernels. RAII destructor prevents memory leaks. |
| `src/cuda_kernels.cuh` | C++ header declaring every CUDA wrapper function callable from `.cpp` files. |
| `src/cuda_kernels.cu` | CUDA implementation: `cuda_matmul` via cuBLAS, softmax with shared-memory reduction, causal mask, GELU, Adam update kernel, and memory management. |
| `src/tokenizer.hpp` | `CharTokenizer` with `build`, `encode`, `decode`, `save`, and `load` methods. Serialized into `model.bin` so the chat program needs no access to the training file. |
| `src/model.hpp` | `LayerNorm`, `AttentionHead`, `MultiHeadAttention`, `FeedForward`, `TransformerBlock`, and `GPT`. Includes GPU-native gather, scatter, and gradient clipping. |
| `src/optimizer.hpp` | `AdamOptimizer` with GPU-resident moment vectors and single-kernel dispatch per step. |
| `src/main.cpp` | Data loading, tokenizer construction, model and optimizer construction, gradient accumulation training loop, and final generation. |
| `src/chat.cpp` | Interactive terminal. Loads `model.bin`, accepts text prompts, calls `generate`, and decodes the output. |


## Prerequisites

**CUDA Toolkit 12.0 or later.** Download from `developer.nvidia.com/cuda-downloads`. After installation verify with:
```
nvcc --version
```

**Visual Studio 2022 Build Tools** with the MSVC C++ compiler. The included `build.bat` initializes the MSVC environment automatically.

**An NVIDIA GPU with SM 8.6 or similar.** The `build.bat` compiles for SM 8.6 (RTX 4080 Ampere). Change the `-arch=sm_86` flag to match your hardware if different.

**Shakespeare training data.** Download with:
```
mkdir data
curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```


## Building and Running

The project includes a `build.bat` that initializes the MSVC environment, adds CUDA to the PATH, and compiles both executables with full optimization:

```bat
.\build.bat
```

This produces `build\llm.exe` for training and `build\chat.exe` for inference.

**Train the model.** Training runs 5,000 optimizer steps, each averaging gradients over 32 random 256-token sequences. On an RTX 3050 Laptop GPU this takes approximately 10 to 15 hours.

```
.\build\llm.exe
```

The model saves to `model.bin` automatically when training completes.

**Chat with the trained model.** The chat program accepts an optional temperature argument. Higher temperature produces more varied output; lower temperature produces more repetitive but more grammatically stable output.

```
.\build\chat.exe model.bin
.\build\chat.exe model.bin -t 0.9 -l 400
```

Type `/quit` to exit, `/temp 0.7` to change temperature, `/len 300` to change generation length, and `/sample` to generate from a blank context.


## Expected Training Progress

| Step | Approximate Loss | Perplexity | What the Output Looks Like |
|---|---|---|---|
| 0 | 4.17 | 65 | Random characters with no pattern |
| 500 | 2.9 | 18 | Letter clusters beginning to form |
| 1,000 | 2.5 | 12 | Recognisable words appear |
| 2,500 | 2.1 | 8 | Shakespearean style becoming visible |
| 5,000 | 1.7 to 1.9 | 5 to 7 | Coherent phrases with character names |

The loss at step 2,000 in earlier single-batch training (loss 2.45) is equivalent to approximately step 100 in the gradient-accumulation version because each optimizer step now sees 32 times as many sequences.


## Common Issues

| Symptom | Fix |
|---|---|
| Loss becomes NaN immediately | Reduce `max_lr` from `6e-4` to `3e-4` in `main.cpp` and rebuild. |
| Loss stops decreasing after 500 steps | Verify that `model.zero_grad()` is called at the start of every micro-batch loop iteration, not outside it. |
| CUDA out of memory | Reduce `block_size` from 256 to 128, or reduce `accum_steps` from 32 to 16. |
| `nvcc` not found | Add the CUDA Toolkit `bin` directory to your PATH, or let `build.bat` do it by verifying the path on line 3 matches your installation. |
| Output is a single repeated character | The temperature is too low. Use `0.8` or higher. |
| Garbled characters in the Windows terminal | Run `chcp 65001` before launching the executable. |


## How It Compares to GPT-2

Both models use identical architectural decisions: Pre-Norm LayerNorm, multi-head causal self-attention with the same scaled dot-product formula, GELU activation, cross-entropy loss, Adam with beta1 of 0.9 and beta2 of 0.999, and gradient clipping by global L2 norm. The barrier between this implementation and GPT-2 is scale and data, not a different algorithm.

| Property | This Model | GPT-2 Small |
|---|---|---|
| Total parameters | 10.7 million | 117 million |
| Embedding dimension | 384 | 768 |
| Transformer layers | 6 | 12 |
| Context window | 256 tokens | 1,024 tokens |
| Tokenizer | Character-level, 65 tokens | BPE, 50,257 tokens |
| Training data | 1 MB Shakespeare | 40 GB WebText |
| Training hardware | RTX 4080 Laptop GPU | Multiple A100 GPUs |


## Further Reading

Vaswani et al., *Attention Is All You Need*, 2017. Every architecture decision in this project traces back to this paper.

Radford et al., *Language Models are Unsupervised Multitask Learners*, 2019. The GPT-2 paper describing the Pre-Norm variant.

Kingma and Ba, *Adam: A Method for Stochastic Optimization*, 2014. Derives the optimizer used here from first principles.

Karpathy, *nanoGPT*, GitHub. The reference implementation that informed the model configuration.


---
*Mahdy*
