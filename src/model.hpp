#pragma once
#include "matrix.hpp"
#include "tokenizer.hpp"
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <iostream>
#include <map>
#include <cstdint>
#include <random>

// model.hpp
//
// Defines every neural network layer and the full GPT model.
//
// The layers stack in this order: token and positional embeddings, six TransformerBlocks
// each containing LayerNorm + MultiHeadAttention + LayerNorm + FeedForward with residual
// connections, a final LayerNorm, and a linear language model head that projects from
// d_model to vocab_size. Every matrix multiply is dispatched to cuBLAS via the Matrix
// class; LayerNorm and the softmax backward pass run on the CPU because their sequential
// per-row logic is simpler to write correctly there and does not bottleneck throughput.
//
// The GPU-native gather/scatter family (gather_params_gpu, gather_grads_gpu,
// scatter_params_gpu) pack all weight matrices into a single flat device buffer using
// device-to-device cudaMemcpy, eliminating PCIe transfers for the optimizer step.

// Downloads one row from a GPU matrix into a CPU vector. Used by the generation loop
// to read the final row of the logit matrix (the prediction for the last token) without
// downloading the entire (T x vocab_size) tensor.
inline std::vector<float> get_row(const Matrix& m, int row) {
    std::vector<float> buf(m.cols);
    cuda_to_host(buf.data(), m.d_data + row * m.cols, m.cols);
    return buf;
}




// Layer Normalization. Applied before each sub-layer in a Pre-Norm transformer.
//
// For each token vector x of length d, the forward pass computes:
//   mean    = (1/d) * sum(x[j])
//   var     = (1/d) * sum((x[j] - mean)^2)
//   x_norm  = (x - mean) / sqrt(var + eps)
//   output  = gamma * x_norm + beta
//
// gamma is initialised to 1 and beta to 0, so the layer starts as identity.
// The model learns to adjust them if a subsequent layer needs a different scale or shift.
// eps = 1e-5 prevents division by zero when the variance of a token vector is zero.
//
// inv_std_cache stores 1/sqrt(var + eps) for every token position, saving recomputation
// during the backward pass. x_norm stores the normalised values before scaling.
struct LayerNorm {
    int d;
    Matrix gamma, beta;       // learned scale (shape 1 x d) and shift (shape 1 x d)
    Matrix d_gamma, d_beta;   // accumulated gradients for gamma and beta
    float eps;                // numerical stability constant, 1e-5

    Matrix x_input, x_norm;           // saved for backward pass
    std::vector<float> inv_std_cache; // 1/sqrt(var+eps) per token, saved for backward

    LayerNorm() : d(0), eps(1e-5f) {}
    LayerNorm(int d, float eps = 1e-5f) : d(d), eps(eps),
        gamma(1, d), beta(1, d), d_gamma(1, d), d_beta(1, d) {
        std::vector<float> ones(d, 1.0f), zeros(d, 0.0f);
        gamma.from_cpu(ones);   // gamma starts at 1: output equals x_norm initially
        beta.from_cpu(zeros);   // beta starts at 0: no shift initially
    }

    // Forward pass. Runs on the CPU because the sequential per-row reduction does not
    // parallelise across rows, and the row count T (up to 256) is too small to warrant
    // a custom CUDA kernel. The computation is downloaded once and the result uploaded once.
    Matrix forward(const Matrix& X) {
        x_input = X;
        int T = X.rows;
        auto x_cpu = X.to_cpu();
        auto g = gamma.to_cpu(), b = beta.to_cpu();
        std::vector<float> out_cpu(T * d), xn_cpu(T * d);
        inv_std_cache.resize(T);

        for (int i = 0; i < T; ++i) {
            float mean = 0;
            for (int j = 0; j < d; ++j) mean += x_cpu[i * d + j];
            mean /= d;
            float var = 0;
            for (int j = 0; j < d; ++j) {
                float diff = x_cpu[i * d + j] - mean;
                var += diff * diff;
            }
            var /= d;
            float inv_std = 1.0f / std::sqrt(var + eps);
            inv_std_cache[i] = inv_std;  // saved: backward pass divides by this
            for (int j = 0; j < d; ++j) {
                xn_cpu[i * d + j] = (x_cpu[i * d + j] - mean) * inv_std;
                out_cpu[i * d + j] = g[j] * xn_cpu[i * d + j] + b[j];
            }
        }
        x_norm = Matrix(T, d); x_norm.from_cpu(xn_cpu);
        Matrix out(T, d); out.from_cpu(out_cpu);
        return out;
    }

    // Backward pass through LayerNorm. Applies the chain rule through the normalisation,
    // scale, and shift operations. m1 and m2 are the two correction terms that arise from
    // differentiating the mean-subtraction and variance-normalisation steps:
    //   dx[i][j] = inv_std * (dy_g[j] - m1 - x_norm[i][j] * m2)
    // where dy_g = d_out * gamma element-wise, m1 = mean(dy_g), m2 = mean(dy_g * x_norm).
    Matrix backward(const Matrix& d_out) {
        int T = d_out.rows;
        auto dout_cpu = d_out.to_cpu();
        auto g = gamma.to_cpu();
        auto xn_cpu = x_norm.to_cpu();
        std::vector<float> dx_cpu(T * d, 0), dg(d, 0), db(d, 0);

        for (int i = 0; i < T; ++i) {
            float inv_std = inv_std_cache[i];
            for (int j = 0; j < d; ++j) {
                dg[j] += dout_cpu[i * d + j] * xn_cpu[i * d + j];  // gradient for gamma
                db[j] += dout_cpu[i * d + j];                        // gradient for beta
            }
            float m1 = 0, m2 = 0;
            for (int j = 0; j < d; ++j) {
                float dy_g = dout_cpu[i * d + j] * g[j];
                m1 += dy_g;
                m2 += dy_g * xn_cpu[i * d + j];
            }
            m1 /= d; m2 /= d;
            for (int j = 0; j < d; ++j)
                dx_cpu[i * d + j] = inv_std * (dout_cpu[i * d + j] * g[j] - m1 - xn_cpu[i * d + j] * m2);
        }
        d_gamma = Matrix(1, d); d_gamma.from_cpu(dg);
        d_beta = Matrix(1, d); d_beta.from_cpu(db);
        Matrix dx(T, d); dx.from_cpu(dx_cpu);
        return dx;
    }

    void zero_grad() { d_gamma.zero(); d_beta.zero(); }
};




// A single causal self-attention head. Computes:
//   Q = X * W_Q,  K = X * W_K,  V = X * W_V     linear projections to d_k dimensions
//   scores = Q * K^T / sqrt(d_k)                  scaled dot-product attention
//   scores[i][j] = -inf for j > i                 causal mask (future positions zeroed)
//   weights = softmax(scores)                      attention weights, sum to 1 per row
//   output = weights * V                           weighted sum of value vectors
//
// d_k = d_model / n_heads = 64. The 1/sqrt(d_k) scale prevents the dot products from
// growing so large that the softmax saturates and gradients vanish.
//
// All six cache matrices (X, Q, K, V, scores, weights) are retained after the forward
// pass so the backward pass can compute gradients without rerunning the forward.
struct AttentionHead {
    int d_model, d_k;
    Matrix W_Q, W_K, W_V;                                      // projection weights
    Matrix dW_Q, dW_K, dW_V;                                   // their gradients
    Matrix X_cache, Q_cache, K_cache, V_cache, scores_cache, weights_cache; // saved activations

    AttentionHead() : d_model(0), d_k(0) {}
    AttentionHead(int dm, int dk, unsigned int seed)
        : d_model(dm), d_k(dk),
          W_Q(dm, dk), W_K(dm, dk), W_V(dm, dk),
          dW_Q(dm, dk), dW_K(dm, dk), dW_V(dm, dk) {
        // Xavier initialisation: std = sqrt(2 / (fan_in + fan_out)). Keeps the variance
        // of Q*K^T dot products close to 1 at the start of training.
        float scale = std::sqrt(2.0f / (dm + dk));
        W_Q.randn(0, scale, seed); W_K.randn(0, scale, seed+1); W_V.randn(0, scale, seed+2);
    }

    Matrix forward(const Matrix& X) {
        X_cache = X;                            // save input for backward
        Q_cache = X.matmul(W_Q);               // (T x d_k)
        K_cache = X.matmul(W_K);               // (T x d_k)
        V_cache = X.matmul(W_V);               // (T x d_k)

        float scale = 1.0f / std::sqrt((float)d_k);  // 1/sqrt(64) = 0.125
        Matrix KT = K_cache.T();               // transpose K to (d_k x T)
        scores_cache = Q_cache.matmul(KT) * scale;   // (T x T) raw attention scores

        // Zero out future positions so position i cannot attend to positions j > i.
        cuda_causal_mask(scores_cache.d_data, scores_cache.rows);

        weights_cache = scores_cache.softmax();      // row-wise softmax: each row sums to 1
        return weights_cache.matmul(V_cache);         // (T x d_k) context vectors
    }

    Matrix backward(const Matrix& d_out) {
        int T = d_out.rows;
        float scale = 1.0f / std::sqrt((float)d_k);

        // Gradient through output = weights * V:
        //   d_V       = weights^T * d_out
        //   d_weights = d_out * V^T
        Matrix WT = weights_cache.T();
        Matrix d_V = WT.matmul(d_out);
        Matrix VT = V_cache.T();
        Matrix d_weights = d_out.matmul(VT);

        // Softmax backward: for each row, d_scores[i][j] = w[i][j]*(d_weights[i][j] - dot(d_weights[i], w[i])).
        // This is the derivative of softmax(s) * d_weights where dot is the inner product.
        auto dw_cpu = d_weights.to_cpu();
        auto w_cpu = weights_cache.to_cpu();
        std::vector<float> ds_cpu(T * T, 0);
        for (int i = 0; i < T; ++i) {
            float sum = 0;
            for (int k = 0; k < T; ++k) sum += dw_cpu[i*T+k] * w_cpu[i*T+k];
            for (int j = 0; j < T; ++j)
                ds_cpu[i*T+j] = w_cpu[i*T+j] * (dw_cpu[i*T+j] - sum);
        }
        // Re-apply the causal mask: future positions had zero weight, so their gradient
        // must also be zero to avoid injecting phantom signal into the backward pass.
        for (int i = 0; i < T; ++i)
            for (int j = i+1; j < T; ++j)
                ds_cpu[i*T+j] = 0;
        for (auto& v : ds_cpu) v *= scale;  // chain rule through the 1/sqrt(d_k) scale

        Matrix d_scores(T, T); d_scores.from_cpu(ds_cpu);

        // Gradients through Q = X*W_Q and K = X*W_K:
        Matrix d_Q = d_scores.matmul(K_cache);      // d_scores * K
        Matrix d_K = d_scores.T().matmul(Q_cache);  // d_scores^T * Q

        // Weight gradients: dW = X^T * d_projection
        Matrix XT = X_cache.T();
        dW_Q = XT.matmul(d_Q);
        dW_K = XT.matmul(d_K);
        dW_V = XT.matmul(d_V);

        // Input gradient: sum contributions from Q, K, and V paths.
        Matrix WQT = W_Q.T(), WKT = W_K.T(), WVT = W_V.T();
        Matrix d_X = d_Q.matmul(WQT) + d_K.matmul(WKT) + d_V.matmul(WVT);
        return d_X;
    }

    void zero_grad() { dW_Q.zero(); dW_K.zero(); dW_V.zero(); }
};




// Multi-Head Attention. Runs h independent AttentionHead instances in parallel, each
// with its own Q, K, V projections. Their outputs are concatenated along the feature
// axis to form a (T x d_model) matrix, then projected through W_O back to d_model.
// Running multiple heads lets the model attend to different aspects of the context
// simultaneously: one head might track syntactic structure while another tracks
// content words. d_k = d_model / h = 64 keeps the total computation constant.
struct MultiHeadAttention {
    int h, d_model, d_k;
    std::vector<AttentionHead> heads;   // h independent attention heads
    Matrix W_O, dW_O, concat_cache;    // output projection and its gradient

    MultiHeadAttention() : h(0), d_model(0), d_k(0) {}
    MultiHeadAttention(int nh, int dm, unsigned int seed)
        : h(nh), d_model(dm), d_k(dm/nh),
          W_O(dm, dm), dW_O(dm, dm) {
        for (int i = 0; i < h; ++i) heads.emplace_back(dm, d_k, seed + i*3);
        W_O.randn(0, 0.02f, seed + h*3);
    }

    Matrix forward(const Matrix& X) {
        int T = X.rows;
        // Run every head independently. Each produces a (T x d_k) output.
        std::vector<std::vector<float>> head_outs(h);
        for (int hi = 0; hi < h; ++hi) {
            Matrix ho = heads[hi].forward(X);
            head_outs[hi] = ho.to_cpu();
        }
        // Interleave head outputs into one (T x d_model) matrix on the CPU.
        // For token t, the slice [hi*d_k : (hi+1)*d_k] contains head hi's output.
        std::vector<float> cat(T * d_model);
        for (int hi = 0; hi < h; ++hi)
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < d_k; ++j)
                    cat[t * d_model + hi * d_k + j] = head_outs[hi][t * d_k + j];
        concat_cache = Matrix(T, d_model);
        concat_cache.from_cpu(cat);
        return concat_cache.matmul(W_O);  // project concatenation back to d_model
    }

    Matrix backward(const Matrix& d_out) {
        int T = d_out.rows;
        // Gradient through the output projection W_O.
        Matrix WOT = W_O.T();
        Matrix d_concat = d_out.matmul(WOT);  // gradient w.r.t. the concatenated heads
        Matrix CT = concat_cache.T();
        dW_O = CT.matmul(d_out);              // gradient w.r.t. W_O

        // Slice d_concat back into per-head gradients and run each head's backward.
        auto dc_cpu = d_concat.to_cpu();
        Matrix d_X(T, d_model);
        d_X.zero();
        for (int hi = 0; hi < h; ++hi) {
            std::vector<float> dh(T * d_k);
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < d_k; ++j)
                    dh[t * d_k + j] = dc_cpu[t * d_model + hi * d_k + j];
            Matrix d_head(T, d_k); d_head.from_cpu(dh);
            d_X += heads[hi].backward(d_head);  // accumulate input gradients from each head
        }
        return d_X;
    }

    void zero_grad() { dW_O.zero(); for (auto& h : heads) h.zero_grad(); }
};




// Position-wise Feed-Forward Network. Applied identically to every token position.
// The two-layer MLP expands the representation from d_model to d_ff (4x wider), applies
// GELU, then compresses back to d_model:
//   h_pre = X * W1 + b1       shape (T x d_ff), linear expansion
//   h     = GELU(h_pre)       element-wise non-linearity
//   out   = h * W2 + b2       shape (T x d_model), projection back
//
// The expansion stores learned associations between features. GELU is preferred over ReLU
// because it is smooth near zero and allows small negative values to pass through with
// reduced magnitude, which improves gradient flow and prevents dead neurons.
struct FeedForward {
    int d_model, d_ff;
    Matrix W1, W2, b1, b2;              // weights and biases for both linear layers
    Matrix dW1, dW2, db1, db2;          // their gradients
    Matrix X_cache, h_pre_cache, h_cache;  // pre-activation and post-GELU hidden state, saved for backward

    FeedForward() : d_model(0), d_ff(0) {}
    FeedForward(int dm, int dff, unsigned int seed)
        : d_model(dm), d_ff(dff),
          W1(dm, dff), W2(dff, dm), b1(1, dff), b2(1, dm),
          dW1(dm, dff), dW2(dff, dm), db1(1, dff), db2(1, dm) {
        W1.randn(0, 0.02f, seed); W2.randn(0, 0.02f, seed+1);
    }

    Matrix forward(const Matrix& X) {
        X_cache = X;                    // save input for dW1 computation in backward
        int T = X.rows;
        h_pre_cache = X.matmul(W1);    // first linear projection: (T x d_ff)
        cuda_add_bias(h_pre_cache.d_data, b1.d_data, h_pre_cache.d_data, T, d_ff);
        h_cache = Matrix(T, d_ff);
        cuda_gelu(h_pre_cache.d_data, h_cache.d_data, T * d_ff);  // GELU non-linearity
        Matrix out = h_cache.matmul(W2);    // second linear projection: (T x d_model)
        cuda_add_bias(out.d_data, b2.d_data, out.d_data, T, d_model);
        return out;
    }

    Matrix backward(const Matrix& d_out) {
        int T = d_out.rows;
        // Gradient through the second linear layer: dW2 = h^T * d_out.
        Matrix HT = h_cache.T();
        dW2 = HT.matmul(d_out);
        // db2 is the sum of d_out across all token positions (bias is shared across T).
        auto dout_cpu = d_out.to_cpu();
        std::vector<float> db2_cpu(d_model, 0);
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < d_model; ++j)
                db2_cpu[j] += dout_cpu[t * d_model + j];
        db2.from_cpu(db2_cpu);

        // Propagate gradient back through the second projection: d_h = d_out * W2^T.
        Matrix W2T = W2.T();
        Matrix d_h = d_out.matmul(W2T);

        // Chain rule through GELU: d_h_pre[i] = d_h[i] * GELU'(h_pre[i]).
        Matrix d_h_pre(T, d_ff);
        cuda_gelu_backward(d_h.d_data, h_pre_cache.d_data, d_h_pre.d_data, T * d_ff);

        // Gradient through the first linear layer.
        Matrix XT = X_cache.T();
        dW1 = XT.matmul(d_h_pre);  // dW1 = X^T * d_h_pre
        auto dhp_cpu = d_h_pre.to_cpu();
        std::vector<float> db1_cpu(d_ff, 0);
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < d_ff; ++j)
                db1_cpu[j] += dhp_cpu[t * d_ff + j];  // db1 summed across positions
        db1.from_cpu(db1_cpu);

        Matrix W1T = W1.T();
        Matrix d_X = d_h_pre.matmul(W1T);  // input gradient: d_h_pre * W1^T
        return d_X;
    }

    void zero_grad() { dW1.zero(); dW2.zero(); db1.zero(); db2.zero(); }
};




// One Transformer block implementing the Pre-Norm GPT-2 design.
// The Pre-Norm arrangement applies LayerNorm before each sub-layer rather than after.
// This stabilises gradients in deep networks: the residual stream x is always in its
// un-normalised form when added back, which preserves gradient magnitude across layers.
//
// Forward pass:
//   x1  = x  + Attention( LayerNorm1(x) )    first residual connection
//   out = x1 + FeedForward( LayerNorm2(x1) ) second residual connection
struct TransformerBlock {
    MultiHeadAttention attn;
    FeedForward ffn;
    LayerNorm ln1, ln2;  // one LayerNorm before attention, one before FFN

    TransformerBlock() {}
    TransformerBlock(int dm, int nh, int dff, unsigned int seed)
        : attn(nh, dm, seed), ffn(dm, dff, seed+100), ln1(dm), ln2(dm) {}

    Matrix forward(const Matrix& x) {
        Matrix ln1_out = ln1.forward(x);         // normalise before attention
        Matrix attn_out = attn.forward(ln1_out);
        Matrix x1 = x + attn_out;               // first residual: add raw x back
        Matrix ln2_out = ln2.forward(x1);        // normalise before FFN
        Matrix ffn_out = ffn.forward(ln2_out);
        return x1 + ffn_out;                     // second residual: add x1 back
    }

    // Backward mirrors the forward in reverse. Because both sub-layers use residual
    // connections, d_out flows directly to the earlier residual stream via addition,
    // and also through the sub-layer's backward pass. The two contributions are summed.
    Matrix backward(const Matrix& d_out) {
        Matrix d_ffn = ffn.backward(d_out);          // through FFN
        Matrix d_x1 = d_out + ln2.backward(d_ffn);   // residual: d_out + through ln2+FFN
        Matrix d_attn = attn.backward(d_x1);          // through attention
        Matrix d_x = d_x1 + ln1.backward(d_attn);    // residual: d_x1 + through ln1+attn
        return d_x;
    }

    void zero_grad() { attn.zero_grad(); ffn.zero_grad(); ln1.zero_grad(); ln2.zero_grad(); }
};




// The full GPT model. Owns the token embedding table, position embedding table, all
// six Transformer blocks, the final LayerNorm, and the language model head.
//
// wte (word token embeddings): shape (vocab_size x d_model). Row token_id gives the
//   384-dimensional embedding for that character.
// wpe (word position embeddings): shape (block_size x d_model). Row t gives the
//   position embedding for sequence position t. Added to wte before the first block.
// lm_head: shape (d_model x vocab_size). Projects the final hidden state to logits.
//   One logit per vocabulary token; the highest logit is the model's top prediction.
//
// The GPU-native gather/scatter pair (gather_params_gpu, scatter_params_gpu) concatenates
// all weight matrices into a single flat device buffer before the optimizer step and
// writes updated values back afterward. This avoids iterating over each matrix on the
// CPU and eliminates the associated PCIe overhead.
struct GPT {
    int vocab_size, d_model, n_layers, n_heads, d_ff, block_size;
    Matrix wte, wpe, dwte, dwpe;     // token and position embeddings plus their gradients
    std::vector<TransformerBlock> blocks;  // the n_layers Transformer blocks
    LayerNorm ln_f;                  // final LayerNorm before the language model head
    Matrix lm_head, dlm_head;        // output projection (d_model x vocab_size) and gradient
    Matrix x_embed_cache, final_hidden_cache;  // saved for the backward pass

    GPT() : vocab_size(0), d_model(0), n_layers(0), n_heads(0), d_ff(0), block_size(0) {}

    // Allocating constructor. All matrices are GPU-resident from the moment of construction.
    // Fixed seeds make the random initialisation reproducible: same seed always produces the
    // same starting weights, which is useful for debugging convergence issues.
    GPT(int V, int d, int N, int h, int dff, int T)
        : vocab_size(V), d_model(d), n_layers(N), n_heads(h), d_ff(dff), block_size(T),
          wte(V, d), wpe(T, d), dwte(V, d), dwpe(T, d),
          ln_f(d), lm_head(d, V), dlm_head(d, V) {
        wte.randn(0, 0.02f, 42);    // std 0.02 is the GPT-2 default for token embeddings
        wpe.randn(0, 0.01f, 43);    // slightly smaller for positional embeddings
        lm_head.randn(0, 0.02f, 44);
        for (int i = 0; i < N; ++i)
            blocks.emplace_back(d, h, dff, 1000 + i * 200);  // distinct seed per block
    }

    // Forward pass. Returns logits of shape (T x vocab_size).
    // Step 1: Look up token embeddings from wte and add position embeddings from wpe.
    //   x[t] = wte[ token_id[t] ] + wpe[ t ]
    // Step 2: Pass x through all six Transformer blocks sequentially.
    // Step 3: Apply the final LayerNorm and multiply by lm_head to get raw logits.
    // The embeddings are assembled on the CPU and uploaded once; all block operations
    // happen GPU-side with no further transfers.
    Matrix forward(const std::vector<int>& tokens) {
        int T = (int)tokens.size();
        // Assemble the embedding matrix from two lookup tables.
        auto wte_cpu = wte.to_cpu(), wpe_cpu = wpe.to_cpu();
        std::vector<float> emb(T * d_model);
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < d_model; ++j)
                emb[t * d_model + j] = wte_cpu[tokens[t] * d_model + j] + wpe_cpu[t * d_model + j];
        x_embed_cache = Matrix(T, d_model);
        x_embed_cache.from_cpu(emb);  // upload to GPU; all subsequent ops are GPU-side

        Matrix x = x_embed_cache;
        for (auto& block : blocks) x = block.forward(x);  // six sequential blocks
        final_hidden_cache = ln_f.forward(x);             // saved for dlm_head backward
        return final_hidden_cache.matmul(lm_head);         // (T x vocab_size) logits
    }

    // Backward pass. d_logits is the gradient of the loss w.r.t. the logit matrix.
    // The backward sequence is exactly the reverse of the forward:
    //   dlm_head  = final_hidden^T * d_logits
    //   d_hidden  = d_logits * lm_head^T
    //   d_x       = backward through final LayerNorm
    //   d_x       = backward through each block, highest index first
    // The resulting d_x is then split into gradients for wte and wpe.
    void backward(const Matrix& d_logits, const std::vector<int>& tokens) {
        int T = (int)tokens.size();
        Matrix FHT = final_hidden_cache.T();
        dlm_head = FHT.matmul(d_logits);    // weight gradient for the output projection
        Matrix LMT = lm_head.T();
        Matrix d_hidden = d_logits.matmul(LMT);   // propagate back through lm_head
        Matrix d_x = ln_f.backward(d_hidden);      // back through final LayerNorm
        for (int i = n_layers - 1; i >= 0; --i) d_x = blocks[i].backward(d_x);  // reverse order

        // Scatter d_x into the embedding gradients. Each token position contributes to
        // the row of dwte corresponding to its token ID and the row of dwpe at its position.
        auto dx_cpu = d_x.to_cpu();
        std::vector<float> dwte_cpu(vocab_size * d_model, 0);
        std::vector<float> dwpe_cpu(block_size * d_model, 0);
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < d_model; ++j) {
                dwte_cpu[tokens[t] * d_model + j] += dx_cpu[t * d_model + j];  // accumulate into token row
                dwpe_cpu[t * d_model + j] += dx_cpu[t * d_model + j];          // accumulate into position row
            }
        dwte.from_cpu(dwte_cpu);
        dwpe.from_cpu(dwpe_cpu);
    }

    // CPU-side gather: downloads all weights to a flat host vector. Kept for save().
    // The ordering of tensors must match scatter_params exactly so that serialisation
    // and deserialisation reconstruct the model identically.
    std::vector<float> gather_params() const {
        std::vector<float> p;
        auto push = [&](const Matrix& m) { auto v = m.to_cpu(); p.insert(p.end(), v.begin(), v.end()); };
        push(wte); push(wpe); push(lm_head);
        auto g = ln_f.gamma.to_cpu(), b = ln_f.beta.to_cpu();
        p.insert(p.end(), g.begin(), g.end());
        p.insert(p.end(), b.begin(), b.end());
        for (const auto& bl : blocks) {
            for (const auto& hd : bl.attn.heads) { push(hd.W_Q); push(hd.W_K); push(hd.W_V); }
            push(bl.attn.W_O);
            push(bl.ffn.W1); push(bl.ffn.W2); push(bl.ffn.b1); push(bl.ffn.b2);
            auto lg = bl.ln1.gamma.to_cpu(), lb = bl.ln1.beta.to_cpu();
            p.insert(p.end(), lg.begin(), lg.end()); p.insert(p.end(), lb.begin(), lb.end());
            auto l2g = bl.ln2.gamma.to_cpu(), l2b = bl.ln2.beta.to_cpu();
            p.insert(p.end(), l2g.begin(), l2g.end()); p.insert(p.end(), l2b.begin(), l2b.end());
        }
        return p;
    }

    // CPU-side gather for gradients. Same tensor ordering as gather_params.
    std::vector<float> gather_grads() const {
        std::vector<float> p;
        auto push = [&](const Matrix& m) { auto v = m.to_cpu(); p.insert(p.end(), v.begin(), v.end()); };
        push(dwte); push(dwpe); push(dlm_head);
        auto dg = ln_f.d_gamma.to_cpu(), db = ln_f.d_beta.to_cpu();
        p.insert(p.end(), dg.begin(), dg.end());
        p.insert(p.end(), db.begin(), db.end());
        for (const auto& bl : blocks) {
            for (const auto& hd : bl.attn.heads) { push(hd.dW_Q); push(hd.dW_K); push(hd.dW_V); }
            push(bl.attn.dW_O);
            push(bl.ffn.dW1); push(bl.ffn.dW2); push(bl.ffn.db1); push(bl.ffn.db2);
            auto lg = bl.ln1.d_gamma.to_cpu(), lb = bl.ln1.d_beta.to_cpu();
            p.insert(p.end(), lg.begin(), lg.end()); p.insert(p.end(), lb.begin(), lb.end());
            auto l2g = bl.ln2.d_gamma.to_cpu(), l2b = bl.ln2.d_beta.to_cpu();
            p.insert(p.end(), l2g.begin(), l2g.end()); p.insert(p.end(), l2b.begin(), l2b.end());
        }
        return p;
    }

    // Restores parameters from a flat host vector into the individual weight matrices.
    // Used by GPT::load() after reading the binary file.
    void scatter_params(const std::vector<float>& params) {
        int idx = 0;
        auto pull = [&](Matrix& m) {
            std::vector<float> v(m.size());
            for (int i = 0; i < m.size(); ++i) v[i] = params[idx++];
            m.from_cpu(v);
        };
        auto pull_vec = [&](Matrix& m) {
            std::vector<float> v(m.cols);
            for (int i = 0; i < m.cols; ++i) v[i] = params[idx++];
            m.from_cpu(v);
        };
        pull(wte); pull(wpe); pull(lm_head);
        pull_vec(ln_f.gamma); pull_vec(ln_f.beta);
        for (auto& bl : blocks) {
            for (auto& hd : bl.attn.heads) { pull(hd.W_Q); pull(hd.W_K); pull(hd.W_V); }
            pull(bl.attn.W_O);
            pull(bl.ffn.W1); pull(bl.ffn.W2); pull(bl.ffn.b1); pull(bl.ffn.b2);
            pull_vec(bl.ln1.gamma); pull_vec(bl.ln1.beta);
            pull_vec(bl.ln2.gamma); pull_vec(bl.ln2.beta);
        }
    }

    
    int param_count_fast() const {
        int n = wte.size() + wpe.size() + lm_head.size()
              + ln_f.gamma.cols + ln_f.beta.cols;
        for (const auto& bl : blocks) {
            for (const auto& hd : bl.attn.heads)
                n += hd.W_Q.size() + hd.W_K.size() + hd.W_V.size();
            n += bl.attn.W_O.size();
            n += bl.ffn.W1.size() + bl.ffn.W2.size()
               + bl.ffn.b1.cols  + bl.ffn.b2.cols;
            n += bl.ln1.gamma.cols + bl.ln1.beta.cols
               + bl.ln2.gamma.cols + bl.ln2.beta.cols;
        }
        return n;
    }
    int param_count() const { return param_count_fast(); }

    // GPU-native parameter gather. Copies all weight matrices into a single contiguous
    // device buffer using device-to-device cudaMemcpy. No data crosses the PCIe bus.
    // The flat buffer is passed directly to the Adam kernel, which reads and writes it
    // without any CPU involvement. The tensor ordering matches gather_grads_gpu and
    // scatter_params_gpu so the optimizer updates the correct parameter for each gradient.
    Matrix gather_params_gpu() const {
        Matrix flat(1, param_count_fast());
        int off = 0;
        auto push = [&](const Matrix& m, int sz) {
            cuda_copy(m.d_data, flat.d_data + off, sz); off += sz;
        };
        push(wte, wte.size()); push(wpe, wpe.size()); push(lm_head, lm_head.size());
        push(ln_f.gamma, ln_f.gamma.cols); push(ln_f.beta, ln_f.beta.cols);
        for (const auto& bl : blocks) {
            for (const auto& hd : bl.attn.heads) {
                push(hd.W_Q, hd.W_Q.size());
                push(hd.W_K, hd.W_K.size());
                push(hd.W_V, hd.W_V.size());
            }
            push(bl.attn.W_O, bl.attn.W_O.size());
            push(bl.ffn.W1,  bl.ffn.W1.size());
            push(bl.ffn.W2,  bl.ffn.W2.size());
            push(bl.ffn.b1,  bl.ffn.b1.cols);
            push(bl.ffn.b2,  bl.ffn.b2.cols);
            push(bl.ln1.gamma, bl.ln1.gamma.cols);
            push(bl.ln1.beta,  bl.ln1.beta.cols);
            push(bl.ln2.gamma, bl.ln2.gamma.cols);
            push(bl.ln2.beta,  bl.ln2.beta.cols);
        }
        return flat;
    }

    // GPU-native gradient gather. Same layout as gather_params_gpu. Called once per
    // micro-batch to pack the per-layer gradient matrices into a flat buffer before
    // adding them into the accumulation buffer with cuda_add.
    Matrix gather_grads_gpu() const {
        Matrix flat(1, param_count_fast());
        int off = 0;
        auto push = [&](const Matrix& m, int sz) {
            cuda_copy(m.d_data, flat.d_data + off, sz); off += sz;
        };
        push(dwte, dwte.size()); push(dwpe, dwpe.size()); push(dlm_head, dlm_head.size());
        push(ln_f.d_gamma, ln_f.d_gamma.cols); push(ln_f.d_beta, ln_f.d_beta.cols);
        for (const auto& bl : blocks) {
            for (const auto& hd : bl.attn.heads) {
                push(hd.dW_Q, hd.dW_Q.size());
                push(hd.dW_K, hd.dW_K.size());
                push(hd.dW_V, hd.dW_V.size());
            }
            push(bl.attn.dW_O, bl.attn.dW_O.size());
            push(bl.ffn.dW1,  bl.ffn.dW1.size());
            push(bl.ffn.dW2,  bl.ffn.dW2.size());
            push(bl.ffn.db1,  bl.ffn.db1.cols);
            push(bl.ffn.db2,  bl.ffn.db2.cols);
            push(bl.ln1.d_gamma, bl.ln1.d_gamma.cols);
            push(bl.ln1.d_beta,  bl.ln1.d_beta.cols);
            push(bl.ln2.d_gamma, bl.ln2.d_gamma.cols);
            push(bl.ln2.d_beta,  bl.ln2.d_beta.cols);
        }
        return flat;
    }

    // GPU-native parameter scatter. Copies sections of the flat buffer back to the
    // individual weight matrices after the optimizer has updated them. Called once per
    // optimizer step. All copies are device-to-device.
    void scatter_params_gpu(const Matrix& flat) {
        int off = 0;
        auto pull = [&](Matrix& m, int sz) {
            cuda_copy(flat.d_data + off, m.d_data, sz); off += sz;
        };
        pull(wte, wte.size()); pull(wpe, wpe.size()); pull(lm_head, lm_head.size());
        pull(ln_f.gamma, ln_f.gamma.cols); pull(ln_f.beta, ln_f.beta.cols);
        for (auto& bl : blocks) {
            for (auto& hd : bl.attn.heads) {
                pull(hd.W_Q, hd.W_Q.size());
                pull(hd.W_K, hd.W_K.size());
                pull(hd.W_V, hd.W_V.size());
            }
            pull(bl.attn.W_O, bl.attn.W_O.size());
            pull(bl.ffn.W1,  bl.ffn.W1.size());
            pull(bl.ffn.W2,  bl.ffn.W2.size());
            pull(bl.ffn.b1,  bl.ffn.b1.cols);
            pull(bl.ffn.b2,  bl.ffn.b2.cols);
            pull(bl.ln1.gamma, bl.ln1.gamma.cols);
            pull(bl.ln1.beta,  bl.ln1.beta.cols);
            pull(bl.ln2.gamma, bl.ln2.gamma.cols);
            pull(bl.ln2.beta,  bl.ln2.beta.cols);
        }
    }

    // Gradient clipping by global L2 norm. Downloads the flat gradient buffer once to
    // compute the norm on the CPU, then applies the scale factor on the GPU if needed.
    // This is the one unavoidable PCIe transfer per optimizer step: the norm is a scalar
    // reduction that requires all values to be visible simultaneously, which the GPU can
    // do with a reduction kernel but would require additional engineering to avoid the
    // download. The transfer is small (42 MB) relative to the time saved by GPU training.
    void clip_grad_norm_gpu(Matrix& grads_flat, float max_norm) {
        auto g = grads_flat.to_cpu();    // download once to compute the global norm
        float norm = 0;
        for (float v : g) norm += v * v;
        norm = std::sqrt(norm);
        if (norm > max_norm)
            cuda_scale(grads_flat.d_data, max_norm / norm,
                       grads_flat.d_data, grads_flat.size());  // clip on GPU, no re-upload
    }

    // Zeros every gradient tensor in the model. Called at the start of each micro-batch
    // so that the backward pass accumulates into a clean slate rather than summing onto
    // leftover values from the previous micro-batch.
    void zero_grad() {
        dwte.zero(); dwpe.zero(); dlm_head.zero();
        ln_f.zero_grad();
        for (auto& b : blocks) b.zero_grad();
    }

    // CPU-side gradient clipping. Kept as a fallback; replaced by clip_grad_norm_gpu
    // in the GPU training loop.
    void clip_grad_norm(std::vector<float>& grads, float max_norm) {
        float norm = 0;
        for (float g : grads) norm += g * g;
        norm = std::sqrt(norm);
        if (norm > max_norm) {
            float s = max_norm / norm;
            for (float& g : grads) g *= s;
        }
    }

    // Autoregressive generation. Appends max_new tokens to the provided context one at a
    // time. At each step the context is cropped to block_size to stay within the model's
    // maximum sequence length, the forward pass is run, and the logit vector for the last
    // position is extracted. Dividing by temperature sharpens or flattens the distribution
    // before sampling. The sampled token is appended and the loop continues.
    std::vector<int> generate(std::vector<int> ctx, int max_new, float temperature = 0.8f) {
        std::mt19937 rng(std::random_device{}());  // seeded from hardware entropy each call
        for (int i = 0; i < max_new; ++i) {
            // Crop to the last block_size tokens if the context has grown too long.
            std::vector<int> crop(ctx.end() - std::min((int)ctx.size(), block_size), ctx.end());
            Matrix logits = forward(crop);
            int T = (int)crop.size();
            auto last = get_row(logits, T - 1);  // logits for the final position only
            for (auto& v : last) v /= temperature;  // divide before softmax to control sharpness
            float mx = *std::max_element(last.begin(), last.end());
            float sum = 0;
            for (auto& v : last) { v = std::exp(v - mx); sum += v; }  // numerically stable softmax
            for (auto& v : last) v /= sum;
            std::discrete_distribution<int> dist(last.begin(), last.end());
            ctx.push_back(dist(rng));  // sample next token and append to context
        }
        return ctx;
    }

    // Serialises the model to a binary file. The format is:
    //   [4 bytes]  magic number 0x4C4C4D43 ('LLMC') for file validation
    //   [24 bytes] six int32 hyperparameters: vocab_size, d_model, n_layers, n_heads, d_ff, block_size
    //   [variable] tokenizer vocabulary (see CharTokenizer::save)
    //   [4 bytes]  parameter count as int32
    //   [sz*4 bytes] all parameters as float32 in gather_params order
    void save(const std::string& path, const CharTokenizer& tokenizer) const {
        std::ofstream out(path, std::ios::binary);
        if (!out.is_open()) { std::cerr << "ERROR: Cannot write " << path << std::endl; return; }
        uint32_t magic = 0x4C4C4D43;  // 'LLMC': identifies this as a valid model file
        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        int32_t hp[6] = { vocab_size, d_model, n_layers, n_heads, d_ff, block_size };
        out.write(reinterpret_cast<const char*>(hp), sizeof(hp));
        tokenizer.save(out);  // embed the vocabulary so chat.exe needs no training data
        auto params = gather_params();
        int32_t sz = (int32_t)params.size();
        out.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
        out.write(reinterpret_cast<const char*>(params.data()), sz * sizeof(float));
        out.close();
        std::cout << "  Saved to " << path << " (" << sz*4/1024/1024 << " MB)" << std::endl;
    }

    // Loads a model from a binary file written by save(). Reconstructs the tokenizer and
    // all weight matrices. Returns a default-constructed (empty) GPT on any error.
    static GPT load(const std::string& path, CharTokenizer& tokenizer) {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) { std::cerr << "ERROR: Cannot open " << path << std::endl; return GPT(); }
        uint32_t magic; in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != 0x4C4C4D43) { std::cerr << "ERROR: Bad magic" << std::endl; return GPT(); }
        int32_t hp[6]; in.read(reinterpret_cast<char*>(hp), sizeof(hp));
        tokenizer.load(in);  // restore vocabulary before constructing the model
        GPT model(hp[0], hp[1], hp[2], hp[3], hp[4], hp[5]);
        int32_t sz; in.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        std::vector<float> params(sz);
        in.read(reinterpret_cast<char*>(params.data()), sz * sizeof(float));
        model.scatter_params(params);  // distribute loaded weights back to each matrix
        in.close();
        std::cout << "  Loaded " << path << " (" << hp[0] << " vocab, " << hp[2] << "L, d=" << hp[1] << ")" << std::endl;
        return model;
    }
};
