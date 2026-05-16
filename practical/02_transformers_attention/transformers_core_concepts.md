# 02 — Transformers & Attention Mechanisms

> The core architecture behind every modern LLM. You MUST know this deeply.

---

## 1. Why Transformers? (The Problem They Solved)

Before Transformers (2017), NLP used **RNNs and LSTMs**:
- Processed tokens **sequentially** (one by one)
- Struggled with **long-range dependencies** (forgot early context)
- Could not be **parallelized** during training (slow)

**Transformers** solved all three:
- Process all tokens **in parallel**
- Use **attention** to relate any token to any other token directly
- Scale efficiently to billions of parameters

> "Attention Is All You Need" — Vaswani et al., 2017 (Google Brain)

---

## 2. High-Level Transformer Architecture

```
Input Tokens
     ↓
Token Embeddings + Positional Encoding
     ↓
┌─────────────────────────┐
│   Transformer Block     │  ×N layers
│  ┌───────────────────┐  │
│  │  Multi-Head       │  │
│  │  Self-Attention   │  │
│  └────────┬──────────┘  │
│           │ + Residual  │
│      Layer Norm         │
│  ┌───────────────────┐  │
│  │  Feed-Forward     │  │
│  │  Network (FFN)    │  │
│  └────────┬──────────┘  │
│           │ + Residual  │
│      Layer Norm         │
└─────────────────────────┘
     ↓
Output (logits over vocabulary)
```

---

## 3. Attention Mechanism — The Heart of Transformers

### What is Attention?
A mechanism that lets each token **look at all other tokens** and decide how much to "attend" to each one.

Think of it as: *"When processing the word 'bank', should I pay more attention to 'river' or 'money' in this sentence?"*

### Scaled Dot-Product Attention

Every token produces three vectors:
- **Q (Query)**: What am I looking for?
- **K (Key)**: What do I contain?
- **V (Value)**: What do I actually give?

```
Attention(Q, K, V) = softmax( QK^T / √d_k ) * V
```

Step by step:
1. Compute similarity scores: `QK^T` (dot product of query with all keys)
2. Scale: divide by `√d_k` to prevent very large values
3. Softmax: convert scores to probabilities (sum to 1)
4. Weighted sum: multiply probabilities by V to get output

### Why Scale by √d_k?
For large `d_k`, dot products grow large → softmax becomes very "peaky" (near 0 or 1) → vanishing gradients. Scaling prevents this.

---

## 4. Multi-Head Attention

Instead of one attention, run **h attention heads in parallel**, each with different learned Q, K, V projections.

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O

head_i = Attention(Q*W_Q_i, K*W_K_i, V*W_V_i)
```

Why multiple heads?
- Each head can attend to **different aspects** of the input
- Head 1 might learn syntactic relations, Head 2 semantic relations
- Richer, more expressive representations

**Typical values:**
| Model | d_model | Heads | d_k |
|-------|---------|-------|-----|
| GPT-2 small | 768 | 12 | 64 |
| BERT base | 768 | 12 | 64 |
| GPT-3 | 12288 | 96 | 128 |

---

## 5. Types of Attention

### Self-Attention
Q, K, V all come from the **same sequence**. Each token attends to all other tokens in the same sequence.
→ Used in encoder (BERT) and decoder (GPT)

### Cross-Attention
Q comes from one sequence, K and V from **another sequence**.
→ Used in encoder-decoder models (T5) between encoder output and decoder

### Causal (Masked) Self-Attention
Each token can only attend to **previous tokens** (not future ones).
→ Used in GPT-style decoders (autoregressive generation)
Implemented by masking future positions to -∞ before softmax.

```
Token 1: can attend to [Token 1]
Token 2: can attend to [Token 1, Token 2]
Token 3: can attend to [Token 1, Token 2, Token 3]
```

---

## 6. Positional Encoding

Transformers have no inherent sense of order (unlike RNNs). Positional encoding adds position information.

### Sinusoidal Positional Encoding (Original)
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
Added directly to token embeddings.

### Learned Positional Embeddings
Learn a separate embedding for each position (used in BERT, GPT-2).

### RoPE (Rotary Position Embedding)
- Used in modern LLMs (LLaMA, GPT-NeoX, Mistral)
- Encodes relative position directly in attention computation
- Better at generalizing to longer sequences

### ALiBi (Attention with Linear Biases)
- Used in MPT, BLOOM
- Adds a linear bias to attention scores based on distance
- Extrapolates better to longer sequences

---

## 7. Feed-Forward Network (FFN)

After attention, each position passes through a **2-layer MLP** independently:

```
FFN(x) = GELU(x * W_1 + b_1) * W_2 + b_2
```

- Typically 4× wider than d_model
- Example: d_model=768 → FFN hidden dim = 3072
- Acts as "memory" — stores factual knowledge about the world

---

## 8. Residual Connections & Layer Norm

### Residual Connection (Skip Connection)
```
output = LayerNorm(x + sublayer(x))
```
- Adds the input directly to the output of each sub-layer
- Prevents vanishing gradients in deep networks
- Allows gradients to flow directly to early layers

### Pre-Norm vs Post-Norm
| Type | Formula | Used In |
|------|---------|---------|
| Post-Norm | LayerNorm(x + sublayer(x)) | Original Transformer, BERT |
| Pre-Norm | x + sublayer(LayerNorm(x)) | GPT-3, LLaMA (more stable) |

---

## 9. Encoder vs Decoder vs Encoder-Decoder

| Architecture | Attention Type | Examples | Best For |
|-------------|----------------|---------|---------|
| **Encoder-only** | Bidirectional self-attention | BERT, RoBERTa | Classification, NER, embeddings |
| **Decoder-only** | Causal (masked) self-attention | GPT-2, GPT-3, LLaMA | Text generation, completion |
| **Encoder-Decoder** | Bidirectional encoder + causal decoder + cross-attention | T5, BART | Translation, summarization, QA |

---

## 10. KV Cache

During inference (text generation), the model generates one token at a time.
Re-computing K and V for all previous tokens every step is wasteful.

**KV Cache**: Store the K and V matrices from past tokens. Only compute for the new token.

```
Without KV cache: O(n²) computation per step
With KV cache: O(n) computation per step
```

Critical for efficient inference. Used in all production LLM serving systems.

---

## 11. Context Window & Attention Complexity

**Context window**: Maximum number of tokens the model can process at once.

Attention has **O(n²) memory and compute** with sequence length n.
- This is why long contexts are expensive.
- Active research area: Flash Attention, Sparse Attention, Linear Attention.

### Flash Attention
- Reorders attention computation to minimize memory I/O
- Same mathematical result, much faster and memory-efficient
- Flash Attention 2 / 3 are standard in modern training

---

## 12. Transformer Scaling

Key insight from "Scaling Laws" (Kaplan et al., 2020):
- Model performance improves predictably with:
  - More **parameters**
  - More **data**
  - More **compute**

Larger models → better performance on all tasks (emergent abilities).

---

## 13. Interview Questions — Transformers

**Q: Explain the attention mechanism.**
> Each token creates Query, Key, and Value vectors. Attention scores are computed as scaled dot products of Query with all Keys, then softmaxed to get weights. These weights are applied to Values to produce the output — allowing each token to gather information from relevant other tokens.

**Q: Why is Multi-Head Attention better than single-head?**
> Multiple heads allow the model to attend to different aspects simultaneously — one head might capture syntactic patterns while another captures semantic relationships. The concatenated outputs give richer representations.

**Q: What is causal masking and why is it used?**
> In autoregressive generation (GPT), the model predicts the next token based only on past tokens. Causal masking prevents tokens from attending to future positions during training, ensuring the model learns to generate left-to-right.

**Q: What is the difference between BERT and GPT architecturally?**
> BERT uses bidirectional self-attention (encoder-only) — every token attends to all other tokens. GPT uses causal (left-to-right) attention (decoder-only) — tokens only attend to previous tokens. BERT is better for understanding tasks; GPT is better for generation.

**Q: Why do Transformers need positional encoding?**
> Self-attention is permutation-invariant — it doesn't care about token order. Positional encoding injects position information so the model knows the sequence order.

**Q: What is the KV cache and why does it matter?**
> During autoregressive generation, Key and Value matrices from previous tokens don't change. Caching them avoids recomputation and reduces inference time from O(n²) to O(n) per generation step.

**Q: What is Flash Attention?**
> A hardware-aware attention algorithm that rewrites the attention computation to minimize GPU memory reads/writes (I/O bound operations), achieving the same mathematical result 2-4× faster with significantly less memory.

---

## Quick Reference Cheat Sheet

```
Core Formula:    Attention(Q,K,V) = softmax(QK^T / √d_k) * V
Heads:           Multiple parallel attention heads (different projections)
Self-Attention:  Q,K,V from same sequence
Causal Mask:     Prevents attending to future tokens (GPT)
Cross-Attention: Q from decoder, K,V from encoder (T5)
Residuals:       x + sublayer(x) — prevents vanishing gradients
FFN:             2-layer MLP after attention, 4x wider
Positional Enc:  Sinusoidal, Learned, RoPE, ALiBi
KV Cache:        Cache K,V for efficient inference
```
