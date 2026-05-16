# Modern Transformer Improvements — What Changed After 2017

> The original "Attention is All You Need" paper was brilliant but imperfect.
> This file covers every major improvement that made modern LLMs work at scale.
> Every improvement here is asked in production interviews.

---

## 1. FLASH ATTENTION — The Engineering Miracle

### The Problem (Memory Bandwidth Bottleneck):
Standard attention materializes a full n×n attention matrix in GPU HBM
(High Bandwidth Memory — the slow GPU DRAM).

For n=2048: 2048×2048 = 4M floats = 16MB per head per layer
For GPT-3: 96 layers × 96 heads × 16MB = **147GB of read/write PER FORWARD PASS**

GPU compute is fast. GPU memory I/O is slow. Standard attention is I/O bound.

### The Solution (Tiling + Kernel Fusion):
FlashAttention never materializes the full n×n matrix.

```
Standard Attention:
  Step 1: Load Q, K from HBM → compute S = QKᵀ → write S to HBM
  Step 2: Load S from HBM → compute P = softmax(S) → write P to HBM
  Step 3: Load P, V from HBM → compute O = PV → write O to HBM
  Total HBM reads/writes: HUGE

Flash Attention:
  Split Q, K, V into tiles that fit in SRAM (fast on-chip memory)
  Compute attention entirely in SRAM, tile by tile
  Write final O to HBM once
  Total HBM reads/writes: TINY
```

### Why it's mathematically correct:
Online softmax algorithm: you can compute softmax incrementally as you see
more values, keeping only a running max and sum. No need to see all values first.
This enables the tiling approach.

### Real numbers:
- FlashAttention 2: **2-4× faster** than standard attention
- **5-20× less memory** for attention
- Enables sequences that would otherwise run out of memory
- FlashAttention 3 (2024): optimized for H100 GPUs, another 2× speedup

### Impact:
Without FlashAttention, GPT-4's 128K context window would require
terabytes of GPU memory. FlashAttention makes long context practical.

---

## 2. ROPE (ROTARY POSITION EMBEDDING) — The Position Revolution

### Why sinusoidal/learned embeddings had problems:
1. Fixed maximum length — can't generalize beyond training length
2. Absolute positions — model doesn't naturally learn relative positions
3. Learned embeddings: 4096 positions = 4096 × d_model extra parameters

### RoPE Insight:
Instead of adding position to the token embedding BEFORE attention,
encode position INSIDE the attention mechanism by rotating Q and K.

```
Standard: query_final = query_embed + pos_embed
RoPE:     query_final = Rotate(query_embed, position_angle)
```

The rotation angle is proportional to position:
```
θ_i = position × 10000^(-2i/d)
```

### The magic property:
When you compute Q·K (dot product in attention):
```
Q_pos_m · K_pos_n = f(content, m-n)
```
The dot product depends only on the RELATIVE position (m-n), not absolute positions.
This is exactly what the model needs — "how far apart are these two tokens?"

### Why LLaMA/Mistral/GPT-NeoX use RoPE:
1. No extra parameters for positions
2. Naturally encodes relative position
3. Can be extended to longer contexts after training (context interpolation)
4. Generalizes better to unseen lengths

### Context Length Extension with RoPE:
LLaMA 2 was trained on 4096 tokens but researchers extended it to 32K+.
How? **NTK-Aware Scaling** — adjust the base value (10000) to spread rotations:
```
new_base = old_base × (new_length/train_length)^(d/(d-2))
LLaMA 3.1: trained on 8192, extended to 128K with this technique
```

---

## 3. GROUPED QUERY ATTENTION (GQA) — Memory Efficiency at Scale

### The KV Cache Problem at Scale:
During inference, you cache K and V for all previous tokens (KV cache).

For LLaMA 2 70B serving 100 users with 4K context:
```
KV cache per user = 80 layers × 2 × seq_len × num_heads × d_head × 2 bytes
                  = 80 × 2 × 4096 × 64 × 128 × 2 = 10.7 GB
100 users = 1.07 TERABYTES of KV cache
```
Impossible. So either you serve fewer users or reduce the KV cache.

### Three Attention Variants:

**Multi-Head Attention (MHA) — Original:**
```
Q: h heads  K: h heads  V: h heads
h Q heads, h K heads, h V heads — full KV cache
```

**Multi-Query Attention (MQA) — Falcon, PaLM:**
```
Q: h heads  K: 1 head  V: 1 head
h Q heads share 1 K and 1 V — h× smaller KV cache
Risk: quality drops on complex tasks
```

**Grouped Query Attention (GQA) — LLaMA 2/3, Mistral:**
```
Q: h heads  K: g heads  V: g heads  (where g << h, e.g. h=32, g=8)
Groups of 4 Q heads share 1 K,V pair — 4× smaller KV cache
Sweet spot: minimal quality drop, big memory savings
```

### Real production impact:
LLaMA 3 70B: 64 Q heads, 8 KV heads (8× compression)
- Without GQA: 70B model needs ~80GB for KV cache at batch=32, seq=4K
- With GQA: ~10GB for KV cache — fits alongside the model weights

---

## 4. SWIGLU — Better Than RELU, GELU

### Evolution of Activations in Transformers:

**ReLU (original):** `max(0, x)`
- Problem: dying neurons (x < 0 → gradient = 0 → neuron never updates)

**GELU (BERT, GPT-2):** `x × Φ(x)` (Φ = cumulative normal distribution)
- Smooth approximation of ReLU
- Better gradient flow
- Still a single-path computation

**SwiGLU (LLaMA, PaLM, 2022):**
```
SwiGLU(x, W, V) = SiLU(xW) ⊙ (xV)
SiLU(x) = x × sigmoid(x)
```
Two independent linear projections, element-wise multiplied (GLU = Gated Linear Unit).

### Why SwiGLU is better:
1. Gating mechanism: one path controls information flow of another
2. Empirically 5-10% better performance on downstream tasks
3. PaLM paper (Google, 2022) showed this conclusively
4. Now used in every major LLM: LLaMA, Mistral, Gemma, PaLM

**Side effect:** SwiGLU requires 3 weight matrices (W₁, W₂, W₃) vs 2 for standard FFN.
To keep parameter count equal, FFN hidden dim is 8/3 × d_model instead of 4×.
LLaMA 3 8B: d_model=4096, FFN hidden=14336 (≈ 3.5 × d_model, close to 8/3≈2.67 × 4096=10922 but adjusted for efficiency)

---

## 5. RMSNORM — Faster Normalization

### Layer Norm formula (original):
```
LayerNorm(x) = (x - μ) / √(σ² + ε) × γ + β
```
Requires computing: mean (μ), variance (σ²), then normalize, scale, shift.

### RMSNorm (T5, LLaMA):
```
RMSNorm(x) = x / √(mean(x²) + ε) × γ
```
Only requires: root mean square (RMS), then scale.
No mean subtraction. No bias term (β removed).

### Why it works:
The mean subtraction (re-centering) in LayerNorm was shown to be less important
than the scaling. RMSNorm keeps the scaling part and drops re-centering.

**Result:** ~15% faster normalization. Small improvement × 32 layers × every forward pass = meaningful speedup at scale.

---

## 6. ALIBI — ATTENTION WITH LINEAR BIASES

### Used by: MPT, BLOOM

### Alternative to positional encoding entirely.
Instead of adding/rotating position into embeddings, add a position-based
bias directly to attention scores:

```
Attention score = QKᵀ/√d_k + m × [-0, -1, -2, -3, ...]
```
Where `m` is a head-specific slope, and the bias is proportional to distance.

**Key property:** Tokens far apart get penalized more (more negative bias).
Model naturally prefers local attention but CAN attend globally when needed.

**Advantage:** Excellent length extrapolation — train on 2K tokens, works on 100K.
**Disadvantage:** Not as strong as RoPE on absolute position tasks.

---

## 7. MIXTURE OF EXPERTS (MoE) — IN DEPTH

### The Scaling Dilemma:
```
Standard scaling: 2× params → 2× quality → 2× compute cost
MoE scaling:      8× params → ~2× quality → ~1.2× compute cost
```

### Architecture:
Replace EVERY FFN layer with multiple expert FFNs + a router:

```
STANDARD LAYER:
  Input x → Attention → FFN → Output

MoE LAYER:
  Input x → Attention → Router → Expert 1 (if selected)
                               → Expert 2 (if selected)
                               → Expert 3 (if selected)
                               ...
                               → Expert N (if selected)
             → weighted sum of selected expert outputs → Output
```

### The Router:
```
router_logits = x × W_router    (W_router: d_model × num_experts)
gates = softmax(router_logits)
top2 = top-2 experts by gate score
output = gate1 × Expert1(x) + gate2 × Expert2(x)
```

### Load Balancing Loss:
Without this, all tokens go to experts 1 and 2 forever.
Others never train. Classic "rich get richer" collapse.

Solution: add auxiliary loss:
```
L_aux = α × Σ (fraction_tokens_to_expert_i × mean_router_prob_for_expert_i)
```
Encourages uniform distribution of tokens across experts.

### Expert Specialization (What Actually Happens):
Researchers analyzed Mixtral's experts and found:
- Expert routing correlates with **syntax** (verbs go to certain experts)
- Some experts specialize in **programming languages**
- Some experts handle **specific languages** (French, German, etc.)
- Specialization emerges from training, not from design

### Models Using MoE:
| Model | Experts | Active | Total Params | Active Params |
|-------|---------|--------|-------------|---------------|
| Mixtral 8×7B | 8 | 2 | 46.7B | 12.9B |
| Mixtral 8×22B | 8 | 2 | 141B | 39B |
| DeepSeek-V2 | 160 | 6 | 236B | 21B |
| GPT-4 (rumored) | ? | ? | ~1.8T | ~110B |

---

## 8. SPECULATIVE DECODING — 2-4× Faster Inference

### The Problem:
LLM generation is sequential — each token requires a full forward pass.
A 70B model takes ~50ms per token. 200 tokens = 10 seconds. Too slow.

### The Insight:
Most tokens in a sequence are "easy" — common words, predictable continuations.
A tiny model can predict these correctly most of the time.

### How It Works:
```
Small draft model (1B): predicts [" the", " cat", " sat", " on", " the"]
Large verify model (70B): one forward pass verifies all 5 tokens in parallel
Result:
  - " the" → accepted (prob ratio check passes)
  - " cat" → accepted
  - " sat" → rejected (large model disagrees)
  → keep " the", " cat", resample from large model at " sat"
```

One large model forward pass verified 2 tokens instead of generating 1.
If draft accuracy is 80%, you process ~5 tokens per large model forward pass.
**2-4× throughput improvement with identical output quality.**

### Why identical quality?
The acceptance criterion maintains the exact same output distribution as
the large model alone (rejection sampling proof). Mathematically guaranteed.

### Used in production by:
- Google (Gemini serving)
- Meta (LLaMA serving)
- DeepMind (Gemma serving)

---

## 9. CONTEXT LENGTH SCALING — HOW WE GOT TO 1M TOKENS

### The Memory Wall:
Standard attention: O(n²) memory. 1M tokens = 10¹² entries in attention matrix.
Physically impossible.

### Solutions Used:

**1. FlashAttention** — Doesn't materialize n×n matrix. Needed foundation.

**2. Sliding Window Attention (Mistral):**
Each token attends to nearest 4096 tokens only.
Through multiple layers, information propagates across full sequence.

**3. Sparse Attention (Longformer, BigBird):**
Mix of local attention + global attention tokens (CLS-like tokens that attend everywhere).

**4. Linear Attention:**
Reformulate softmax(QKᵀ)V to Q(Kᵀ softmax(V)) using kernel trick.
O(n×d²) instead of O(n²×d). Approximate but scalable.

**5. Ring Attention:**
Distribute sequence across multiple GPUs (sequence parallelism).
Each GPU handles a chunk of the sequence, passes KV to next GPU in a ring.
Allows ~1M+ context by distributing across 128 GPUs.

**6. RoPE Scaling:**
NTK-aware scaling allows length generalization beyond training.

### Current Context Champions:
| Model | Context | Method |
|-------|---------|--------|
| Gemini 1.5 Pro | 1M tokens | Multi-head attention + efficiency |
| Claude 3 | 200K tokens | Sparse attention techniques |
| GPT-4 Turbo | 128K tokens | FlashAttention + efficiency |
| LLaMA 3.1 | 128K tokens | RoPE scaling + FlashAttention |

---

## 10. THE KV CACHE — Production Critical

### What it is:
Every time a transformer processes a token, it computes K and V for that token.
In autoregressive generation (token by token), these don't change for past tokens.

KV cache = save K and V for all past tokens. Reuse them. Only compute new token's K,V.

### Size calculation:
```
KV cache size = 2 (K+V) × num_layers × num_kv_heads × d_head × seq_len × bytes_per_element

LLaMA 3 8B at 4096 context, BF16:
= 2 × 32 × 8 × 128 × 4096 × 2 bytes
= 536 MB per sequence

At batch_size=32:
= 17 GB of KV cache (+ 16GB for model weights = 33GB total)
```

### KV Cache Optimizations:

**PagedAttention (vLLM):**
OS-style memory paging. Allocate KV cache in non-contiguous pages.
No fragmentation. 2-4× more sequences served simultaneously.

**Quantized KV Cache:**
Store KV in INT8 instead of BF16. 2× memory reduction.
Minimal quality loss for most tasks.

**KV Cache Eviction:**
For very long contexts, selectively evict old/unimportant tokens from KV cache.
StreamingLLM: always keep first 4 tokens (attention sinks) + recent window.

---

## INTERVIEW BLAST — Modern Improvements

**"What is FlashAttention and why does every production system use it?"**
> "FlashAttention rewrites attention to stay in GPU SRAM instead of reading/writing
> to HBM. Standard attention does O(n²) HBM memory I/O. FlashAttention tiles the
> computation into blocks that fit in SRAM, reducing HBM I/O to O(n). Same mathematical
> result, 2-4× faster, 5-20× less memory. Without it, 128K context would be impossible."

**"What is RoPE and how does it enable long context?"**
> "RoPE encodes position by rotating Q and K vectors. The key property: Q·K depends
> on relative position, not absolute. This means the model naturally learns positional
> relationships. For long context, we adjust the rotation base frequency
> (NTK-aware scaling), which lets a model trained on 8K tokens generalize to 128K
> without retraining — just fine-tune briefly at the target length."

**"What is speculative decoding?"**
> "A small draft model generates K candidate tokens. The large model verifies all K
> in a single forward pass using an acceptance criterion that maintains the exact output
> distribution. Tokens that pass are kept; the first rejection causes a resample from
> the large model. Result: 2-4× throughput with mathematically identical output quality."
