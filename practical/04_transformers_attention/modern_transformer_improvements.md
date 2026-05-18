# Modern Transformer Improvements — What Changed After 2017

> The original "Attention is All You Need" paper was brilliant but imperfect.
> This file covers every major improvement that made modern LLMs work at scale.
> Every improvement here is asked in production interviews.

---

## 1. FLASH ATTENTION — The Engineering Miracle

**What it is:** A hardware-aware reimplementation of the standard attention calculation that achieves mathematically identical results but reads and writes GPU memory far less often — making it dramatically faster and cheaper.

**Analogy:** Imagine doing math homework. Standard attention is like reading each number from a textbook, writing it on a whiteboard, erasing it, reading the next number, writing it, etc. FlashAttention is like keeping all the numbers in your head while you calculate, then only writing down the final answer. Same answer, vastly less whiteboard use.

### The Problem (Memory Bandwidth Bottleneck):

**What it is:** The GPU is not compute-limited (it's fast enough to do the math), but memory-bandwidth-limited — it spends most of its time reading/writing data, not computing.

Standard attention materializes a full n×n attention matrix in GPU HBM
(High Bandwidth Memory — the slow GPU DRAM).

For n=2048: 2048×2048 = 4M floats = 16MB per head per layer
For GPT-3: 96 layers × 96 heads × 16MB = **147GB of read/write PER FORWARD PASS**

GPU compute is fast. GPU memory I/O is slow. Standard attention is I/O bound.
Think of it like a powerful calculator that takes 10 seconds to read each number from paper — the bottleneck isn't the math, it's the paper reading.

### The Solution (Tiling + Kernel Fusion):

**What it is:** Breaking the n×n matrix into small blocks that fit in fast on-chip memory (SRAM), computing attention entirely there, and only writing the result once.

FlashAttention never materializes the full n×n matrix.

```
Standard Attention (SLOW):
  Step 1: Load Q, K from HBM → compute S = QKᵀ → write S to HBM
          ↑ S is the full n×n matrix — for n=2048 this is ~16MB written to slow DRAM
  Step 2: Load S from HBM → compute P = softmax(S) → write P to HBM
          ↑ Read 16MB, process, write 16MB back — two slow round-trips
  Step 3: Load P, V from HBM → compute O = PV → write O to HBM
          ↑ One more read of the 16MB attention matrix
  Total HBM reads/writes: O(n²) — proportional to the size of the attention matrix

Flash Attention (FAST):
  Split Q, K, V into tiles that fit in SRAM (fast on-chip memory, ~20MB total)
  For each tile of Q, iterate over tiles of K and V:
    Compute partial attention within SRAM — stays in fast memory the whole time
    Accumulate running result using online softmax trick
  Write final O to HBM once — only the final answer leaves fast memory
  Total HBM reads/writes: O(n) — proportional to just the sequence length, not n²
```

### Why it's mathematically correct:

**What it is:** The mathematical trick that makes tiling possible — computing softmax without seeing all the values at once.

Online softmax algorithm: you can compute softmax incrementally as you see
more values, keeping only a running max and sum. No need to see all values first.

```
Normal softmax of [3, 1, 4, 1, 5]:
  Step 1: see all values, find max=5
  Step 2: compute e^(x-5) for each, then divide by sum
  → Requires seeing ALL values before computing ANY

Online softmax (FlashAttention's trick):
  Step 1: see 3 → running_max=3, running_sum=e^0=1
  Step 2: see 1 → 1 < 3, update: running_sum += e^(1-3) = e^(-2)
  Step 3: see 4 → 4 > 3, rescale old sum: running_max=4, running_sum = old_sum*e^(3-4) + e^0
  ...
  Final: identical result to normal softmax, but computed incrementally
  → Can process one tile at a time without materializing the full matrix
```

This enables the tiling approach — process a block of Q against a block of K/V, then next block, etc.

### Real numbers:
- FlashAttention 2: **2-4× faster** than standard attention
- **5-20× less memory** for attention (because the n×n matrix is never stored)
- Enables sequences that would otherwise run out of memory
- FlashAttention 3 (2024): optimized for H100 GPUs, another 2× speedup

### Impact:
Without FlashAttention, GPT-4's 128K context window would require
terabytes of GPU memory just for attention matrices. FlashAttention makes long context practical.

**WHY this matters for you:** Every production LLM training and inference system uses FlashAttention. You'll be expected to know why it exists and what problem it solves.

**Interview answer:** "FlashAttention rewrites attention to stay in GPU SRAM instead of reading/writing to HBM. Standard attention does O(n²) HBM memory I/O. FlashAttention tiles the computation into blocks that fit in SRAM, reducing HBM I/O to O(n). Same mathematical result, 2-4× faster, 5-20× less memory. Without it, 128K context would be impossible."

---

## 2. ROPE (ROTARY POSITION EMBEDDING) — The Position Revolution

**What it is:** A way to encode where each token sits in the sequence, but instead of adding a fixed vector to each token, it rotates the Query and Key vectors — making attention scores naturally depend on relative distance between tokens.

**Analogy:** Imagine two compass needles. One points North (position 5) and another points Northeast (position 7). When you compute their dot product, you get a value that depends on the angle between them — which is exactly their relative position (2 steps apart). RoPE applies this idea to Q and K vectors.

### Why sinusoidal/learned embeddings had problems:
1. Fixed maximum length — can't generalize beyond training length
2. Absolute positions — model doesn't naturally learn relative positions
3. Learned embeddings: 4096 positions = 4096 × d_model extra parameters to train

### RoPE Insight:

**What it is:** The key conceptual shift — instead of modifying token embeddings before attention, modify Q and K inside attention using rotation, so the dot product inherently encodes relative position.

Instead of adding position to the token embedding BEFORE attention,
encode position INSIDE the attention mechanism by rotating Q and K.

```
Standard: query_final = query_embed + pos_embed
          ← position is mixed into the content, hard to separate
RoPE:     query_final = Rotate(query_embed, position_angle)
          ← position is encoded as a rotation, which separates cleanly
```

The rotation angle is proportional to position:
```
θ_i = position × 10000^(-2i/d)
← dimension i of the vector is rotated by this angle
← different dimensions rotate at different speeds (frequencies)
← low i → fast rotation, high i → slow rotation (like clock hands)
```

### The magic property:

**What it is:** The mathematical guarantee that makes RoPE superior — the dot product of two rotated vectors naturally measures their relative position, not absolute.

When you compute Q·K (dot product in attention):
```
Q_pos_m · K_pos_n = f(content, m-n)
← f depends on the content of Q and K (as usual)
← but also only on (m-n): the RELATIVE position, not absolute positions!
← the model automatically knows "these two tokens are 5 positions apart"
```
The dot product depends only on the RELATIVE position (m-n), not absolute positions.
This is exactly what the model needs — "how far apart are these two tokens?"

### Why LLaMA/Mistral/GPT-NeoX use RoPE:
1. **No extra parameters** for positions (unlike learned embeddings)
2. **Naturally encodes relative position** — Q·K gives relative distance for free
3. **Can be extended to longer contexts** after training (context interpolation)
4. **Generalizes better** to unseen lengths than fixed learned embeddings

### Context Length Extension with RoPE:

**What it is:** A technique to use a model trained on short sequences for much longer sequences, by adjusting the rotation frequencies.

LLaMA 2 was trained on 4096 tokens but researchers extended it to 32K+.
How? **NTK-Aware Scaling** — adjust the base value (10000) to spread rotations:

```
new_base = old_base × (new_length/train_length)^(d/(d-2))
← this formula "stretches" the rotation frequencies to cover more positions
← like adjusting a radio to receive a wider frequency range

LLaMA 3.1: trained on 8192, extended to 128K with this technique
← 16× context length extension without full retraining!
← only requires a brief fine-tuning run at the target context length
```

**WHY this works:** The rotation frequencies were calibrated for a certain length. Scaling the base frequency stretches them to cover more positions, like zooming out on a map — you see the same territory, just at a coarser scale.

**Interview answer:** "RoPE encodes position by rotating Q and K vectors. The key property: Q·K depends on relative position, not absolute. This means the model naturally learns positional relationships. For long context, we adjust the rotation base frequency (NTK-aware scaling), which lets a model trained on 8K tokens generalize to 128K without retraining — just fine-tune briefly at the target length."

---

## 3. GROUPED QUERY ATTENTION (GQA) — Memory Efficiency at Scale

**What it is:** A middle-ground design between standard multi-head attention (MHA) and multi-query attention (MQA), where groups of Query heads share a single Key-Value pair — drastically reducing the memory needed to store the KV cache during inference.

**Analogy:** In a library, if every reader (Query head) needed their own private copy of every book (K and V), you'd need 64 copies of every book for 64 readers. GQA says: group readers into sets of 8, and each group of 8 shares 1 copy. You still serve all readers, but with 8× fewer books.

### The KV Cache Problem at Scale:

**What it is:** The memory problem that motivated GQA — during inference, the KV cache per user grows so large it becomes the limiting factor in how many users you can serve.

During inference, you cache K and V for all previous tokens (KV cache).

For LLaMA 2 70B serving 100 users with 4K context:
```
KV cache per user = 2 (K+V) × num_layers × num_kv_heads × d_head × seq_len × bytes_per_element
                  = 2 × 80 × 64 × 128 × 4096 × 2 bytes
                  = 10.7 GB per user
← 10.7 GB just for context cache for ONE user, ONE conversation
100 users = 1.07 TERABYTES of KV cache
← completely impossible — a single A100 has 80GB total
```
Impossible. So either you serve fewer users or reduce the KV cache.

### Three Attention Variants:

**Multi-Head Attention (MHA) — Original:**
```
Q: h heads  K: h heads  V: h heads
← every head has its OWN K and V head
← full KV cache: h K-heads + h V-heads stored per token
```

**Multi-Query Attention (MQA) — Falcon, PaLM:**
```
Q: h heads  K: 1 head  V: 1 head
← ALL query heads share the SAME single K and V head
← h× smaller KV cache (e.g. 64× reduction for 64 heads)
Risk: quality drops on complex tasks (one K/V pair loses information)
```

**Grouped Query Attention (GQA) — LLaMA 2/3, Mistral:**
```
Q: h heads  K: g heads  V: g heads  (where g << h, e.g. h=32, g=8)
← Query heads organized into groups; each group of (h/g) Q-heads shares 1 K,V pair
← e.g. h=32 Q-heads, g=8 K/V-heads → 4 Q-heads share each K/V pair
← 4× smaller KV cache vs MHA
Sweet spot: minimal quality drop, big memory savings
```

### Real production impact:
LLaMA 3 70B: 64 Q heads, 8 KV heads (8× compression)
```
Without GQA: 70B model KV cache at batch=32, seq=4K ≈ 80GB
             ← fills an entire A100 just with cache!
With GQA:   ≈ 10GB for KV cache
             ← fits alongside the 140GB model weights on 2 A100s
← This difference is what makes the model deployable vs undeplosyable
```

**WHY this matters for production:** Being able to serve 8× more users with the same hardware directly translates to 8× lower serving cost per user. For a production LLM serving millions of queries, this is the difference between profitable and unprofitable.

**Interview answer:** "GQA shares KV heads across multiple query heads. LLaMA 3 70B has 64 query heads but only 8 KV heads — 8× smaller KV cache. At 128K context with batch size 32, this is the difference between fitting in memory and not."

---

## 4. SWIGLU — Better Than RELU, GELU

**What it is:** A gated activation function used in the FFN layer that outperforms both ReLU and GELU by adding a learned "gate" that decides how much of each neuron's computation to let through.

**Analogy:** Think of a river flowing through valves. ReLU is a valve that's fully open for positive values, fully shut for negative ones. GELU is a valve with a smooth open/close transition. SwiGLU is TWO rivers — one carries information, the other controls how open the valve is. The valve (gate) is itself learned, making it much more expressive.

### Evolution of Activations in Transformers:

**ReLU (original):** `max(0, x)`
- Turns all negative values to zero
- Problem: dying neurons (x < 0 → gradient = 0 → neuron never updates again)
- Once a neuron goes permanently negative, it stops contributing entirely

**GELU (BERT, GPT-2):** `x × Φ(x)` (Φ = cumulative normal distribution)
- Smooth approximation of ReLU — slight negative values pass through a little
- Better gradient flow — no permanently dead neurons
- Still a single-path computation (one linear layer, one activation)

**SwiGLU (LLaMA, PaLM, 2022):**
```
SwiGLU(x, W, V) = SiLU(xW) ⊙ (xV)
SiLU(x) = x × sigmoid(x)   ← "Swish" activation, smooth and unbounded
⊙ = element-wise multiplication
```
Two independent linear projections, element-wise multiplied (GLU = Gated Linear Unit).

```python
# Standard FFN (before SwiGLU):
x = input_token          # shape: (batch, seq, d_model)
h = linear_1(x)          # expand: d_model → 4*d_model
h = GELU(h)              # single activation on expanded representation
out = linear_2(h)        # contract: 4*d_model → d_model
# One path: expand → activate → contract

# SwiGLU (modern LLMs — LLaMA, PaLM, Gemma):
x = input_token          # shape: (batch, seq, d_model)

gate = linear_gate(x)    # shape: (batch, seq, ffn_dim) — GATE: controls information flow
# This projection learns "which neurons should be active for this input?"

content = linear_content(x)  # shape: (batch, seq, ffn_dim) — CONTENT: the actual values
# This projection learns "what information do I want to pass through?"

# SiLU = x * sigmoid(x): smooth, slightly negative values allowed, unbounded positive
# sigmoid squashes to (0,1), multiplied by x gives a continuous gating signal
gate_activated = SiLU(gate)  # values: near-0 for very negative, near-x for very positive

# Element-wise multiply: gate decides HOW MUCH of each content neuron flows through
output_gated = gate_activated * content  # shape: (batch, seq, ffn_dim)
# Neuron i: gate_i is high → content_i passes through
# Neuron i: gate_i is low  → content_i is suppressed
# The model learns WHICH neurons matter for each type of input

output = linear_proj(output_gated)  # contract back: ffn_dim → d_model
# W₃: final projection to combine gated outputs back to model dimension
```

### Why SwiGLU is better:
1. **Gating mechanism**: one path controls information flow of another path
2. **Empirically 5-10% better** performance on downstream tasks
3. **PaLM paper (Google, 2022)** showed this conclusively across many tasks
4. **Now used in every major LLM**: LLaMA, Mistral, Gemma, PaLM

**Side effect:** SwiGLU requires 3 weight matrices (W₁, W₂, W₃) vs 2 for standard FFN.
To keep parameter count equal, FFN hidden dim is 8/3 × d_model instead of 4×.
LLaMA 3 8B: d_model=4096, FFN hidden=14336 (≈ 3.5 × d_model, slightly wider than 8/3 for hardware efficiency)

**WHY SwiGLU beats GELU?** The gating mechanism lets the model "switch off" irrelevant computations per-input. It's like having a dimmer switch (continuous learned gate) rather than a single on/off switch (ReLU). More information routes dynamically per context.

---

## 5. RMSNORM — Faster Normalization

**What it is:** A simplified version of Layer Normalization that drops the mean-subtraction step, keeping only the root-mean-square scaling — which turns out to be what actually matters for training stability.

**Analogy:** Layer Normalization is like standardizing test scores by first subtracting the class average (re-centering), then dividing by the standard deviation (rescaling). RMSNorm skips the subtraction and just does the rescaling. It turns out the centering step didn't matter much — the rescaling alone provides stability.

### Layer Norm formula (original):
```
LayerNorm(x) = (x - μ) / √(σ² + ε) × γ + β
← Step 1: subtract mean μ (re-centering — removes mean bias)
← Step 2: divide by std dev (rescaling — brings to unit variance)
← Step 3: apply learned scale γ and shift β
Requires computing: mean, variance, then normalize, scale, shift. = 4 operations
```

### RMSNorm (T5, LLaMA):
```
RMSNorm(x) = x / √(mean(x²) + ε) × γ
← Skip mean subtraction entirely
← Only compute root mean square (RMS), then scale
← Apply learned scale γ (no β shift term)
Requires computing: just RMS, then scale. = 2 operations
```

```python
# Layer Norm (original):
mu = x.mean(dim=-1, keepdim=True)          # compute mean: 1 pass through x
var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)  # compute variance: 2nd pass
x_norm = (x - mu) / (var + eps).sqrt()     # normalize: subtract mean, divide by std
output = gamma * x_norm + beta             # scale and shift

# RMSNorm (LLaMA, T5):
rms = (x ** 2).mean(dim=-1, keepdim=True).sqrt()  # compute RMS: 1 pass through x
# No mean subtraction — skip the re-centering step entirely
x_norm = x / (rms + eps)                  # scale by RMS only
output = gamma * x_norm                   # apply learned scale (no beta shift)
# Slightly simpler, slightly faster, equivalent stability in practice
```

### Why it works:

**What it is:** The empirical finding that mean-subtraction doesn't contribute much to normalization's stability benefit.

The mean subtraction (re-centering) in LayerNorm was shown to be less important
than the scaling. RMSNorm keeps the scaling part and drops re-centering.

**Result:** ~15% faster normalization per layer. Small improvement, but at scale:
- 32 layers × 2 norms per layer = 64 norm operations per forward pass for LLaMA 3 8B
- 15% faster each = meaningful total speedup across millions of forward passes
- No measurable quality loss in practice

**WHY remove β as well?** The beta (shift) term compensates for the mean that was subtracted. If you're not subtracting the mean, there's nothing to compensate for — beta becomes redundant and can be removed.

---

## 6. ALIBI — ATTENTION WITH LINEAR BIASES

**What it is:** A positional encoding approach that doesn't modify token embeddings at all — instead it directly penalizes attention scores based on how far apart two tokens are. Farther apart = lower attention score.

**Analogy:** Imagine you're at a party and trying to hear someone speak. If they're standing next to you (1 step away), you hear them clearly. If they're across the room (100 steps away), it's harder. ALiBi adds this "distance penalty" directly to attention — nearby tokens get full attention, far tokens get penalized.

### Used by: MPT, BLOOM

### Alternative to positional encoding entirely.
Instead of adding/rotating position into embeddings, add a position-based
bias directly to attention scores:

```
Attention score (i attending to j) = QKᵀ/√d_k + m × (j - i)
← base score: standard QK dot product
← bias: m × distance (negative when j < i, i.e., j is in the past)
← m: head-specific slope (different heads have different sensitivity to distance)
```

```python
# Example with 4 tokens:
# For head with slope m = -0.5 (moderately distance-sensitive)

# Standard QK scores (before bias):
raw_scores = [[4.2, 3.1, 2.8, 1.9],   # token 0 attending to tokens 0,1,2,3
              [3.5, 4.8, 3.2, 2.1],   # token 1 attending to tokens 0,1,2,3
              ...

# ALiBi distance bias (m = -0.5, causal — only past tokens):
# entry [i, j] = m * (j - i) for j <= i, else 0 (can't attend future)
alibi_bias = [[0,   0,   0,   0],   # token 0: no past to penalize
              [-0.5, 0,   0,   0],   # token 1: token 0 is 1 step back → -0.5
              [-1.0, -0.5, 0,  0],   # token 2: token 0 is 2 steps → -1.0; token 1 → -0.5
              [-1.5, -1.0, -0.5, 0]] # token 3: 3 steps → -1.5; 2 steps → -1.0; etc.

final_scores = raw_scores + alibi_bias  # add bias BEFORE softmax
# ← far-away tokens get increasingly negative scores
# ← after softmax, far tokens get exponentially lower attention weights
```

**Key property:** Tokens far apart get penalized more (more negative bias).
Model naturally prefers local attention but CAN attend globally when needed.

**Advantage:** Excellent length extrapolation — train on 2K tokens, works on 100K.
No special scaling tricks needed — the bias formula extends naturally to any distance.

**Disadvantage:** Not as strong as RoPE on absolute position tasks.
When the model needs to know "is this token early or late in the sequence?"
the answer is harder to extract from a relative bias.

**WHY ALiBi works at long context?** The bias is linear in distance, so it extends naturally to any distance without needing special scaling tricks. RoPE needs explicit "NTK-aware scaling" to extend; ALiBi extends automatically because the formula has no upper limit.

---

## 7. MIXTURE OF EXPERTS (MoE) — IN DEPTH

**What it is:** An architecture where the standard feed-forward network (FFN) in each Transformer layer is replaced by multiple "expert" FFNs, with a learned router deciding which 1-2 experts handle each token. Most parameters are inactive for any given token — meaning you get the quality of a large model at the cost of a smaller one.

**Analogy:** A hospital with 8 specialist departments (cardiologists, neurologists, orthopedics, etc.). When a patient arrives, a triage router sends them to the 2 most relevant departments. The hospital has expertise of 8 specialists but each patient only uses 2. You get broad expertise without everyone consulting every specialist for every visit.

### The Scaling Dilemma:
```
Standard scaling: 2× params → ~2× quality → 2× compute cost per token
MoE scaling:      8× params → ~2× quality → ~1.2× compute cost per token
← MoE: 8× more capacity, but only uses 2 experts → compute barely increases
← This "sparse activation" is the key insight
```

### Architecture:

**What it is:** Exactly what happens inside a single MoE layer — the router, the experts, and how they combine.

Replace EVERY FFN layer with multiple expert FFNs + a router:

```
STANDARD LAYER:
  Input x → Attention → FFN → Output
  ← ALL of the FFN parameters participate for every token

MoE LAYER:
  Input x → Attention → Router → Expert 1 (if selected for this token)
                               → Expert 2 (if selected for this token)
                               → Expert 3 (NOT selected — does nothing)
                               ...
                               → Expert N (NOT selected — does nothing)
             → weighted sum of selected expert outputs → Output
  ← Most experts are SILENT for any given token — sparse activation
```

### The Router:

**What it is:** A small linear layer that converts a token's representation into a probability distribution over experts, then selects the top-K.

```
router_logits = x × W_router    (W_router: d_model × num_experts)
← W_router is a small matrix that projects to num_experts scores
← one score per expert: "how relevant is this expert for this token?"
gates = softmax(router_logits)
← convert scores to probabilities: which experts does this token need?
top2 = top-2 experts by gate score
← select the 2 highest-probability experts (k=2 in most models)
output = gate1 × Expert1(x) + gate2 × Expert2(x)
← weighted combination: high-gate expert contributes more to output
← gates act as learned mixing weights
```

```python
class MoELayer(nn.Module):
    def __init__(self, d_model, num_experts, top_k, ffn_dim):
        self.router = nn.Linear(d_model, num_experts)  # tiny: d_model → num_experts
        # Each expert is a full FFN — same as standard FFN but there are many of them
        self.experts = nn.ModuleList([FFN(d_model, ffn_dim) for _ in range(num_experts)])
        self.top_k = top_k  # typically 2 — use 2 experts per token

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        router_logits = self.router(x)  # shape: (batch, seq_len, num_experts)
        # For each token, compute a score for each expert

        gates = F.softmax(router_logits, dim=-1)  # shape: (batch, seq_len, num_experts)
        # Convert scores to probabilities — which experts are best for this token?

        top_k_gates, top_k_indices = gates.topk(self.top_k, dim=-1)
        # top_k_gates: values (e.g., [0.7, 0.3]) — mixing weights for selected experts
        # top_k_indices: which experts (e.g., [3, 7]) — expert IDs to use

        # Renormalize selected gates to sum to 1
        top_k_gates = top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)
        # So 0.7 and 0.3 → they already sum to 1, but floating point issues can arise

        # Run only the selected experts (all others are silent — no compute)
        output = torch.zeros_like(x)  # initialize output tensor
        for k in range(self.top_k):
            expert_idx = top_k_indices[..., k]  # which expert for this k-th slot
            gate_weight = top_k_gates[..., k:k+1]  # mixing weight for this k-th expert
            expert_output = self.experts[expert_idx](x)  # run this expert
            output += gate_weight * expert_output   # weighted contribution
        # Output: weighted combination of top-2 expert outputs
        return output
```

### Load Balancing Loss:

**What it is:** An auxiliary training loss that forces tokens to be distributed evenly across experts — without it, all tokens collapse onto 1-2 experts and the rest never train.

Without this, all tokens go to experts 1 and 2 forever.
Others never train. Classic "rich get richer" collapse.
This is called "expert collapse" — a critical failure mode for MoE.

Solution: add auxiliary loss:
```
L_aux = α × Σ (fraction_tokens_to_expert_i × mean_router_prob_for_expert_i)
← fraction_tokens: how many tokens are routed to expert i
← mean_router_prob: average router probability for expert i
← their product is high when routing is concentrated → penalize concentration
← α is a small weight (e.g. 0.01) — don't want to override the main task loss
```
Encourages uniform distribution of tokens across experts.

### Expert Specialization (What Actually Happens):

**What it is:** An emergent behavior — during training, experts spontaneously develop specializations even though no one designed them to.

Researchers analyzed Mixtral's experts and found:
- Expert routing correlates with **syntax** (verbs go to certain experts)
- Some experts specialize in **programming languages** (code tokens route together)
- Some experts handle **specific natural languages** (French, German, etc. have their experts)
- Specialization emerges from training, not from design

This mirrors how different human specialists develop expertise naturally from experience.

### Models Using MoE:
| Model | Experts | Active | Total Params | Active Params |
|-------|---------|--------|-------------|---------------|
| Mixtral 8×7B | 8 | 2 | 46.7B | 12.9B |
| Mixtral 8×22B | 8 | 2 | 141B | 39B |
| DeepSeek-V2 | 160 | 6 | 236B | 21B |
| GPT-4 (rumored) | ? | ? | ~1.8T | ~110B |

**WHY MoE makes economic sense:** Inference cost scales with active parameters, not total. A 47B MoE model costs the same to run as a 13B dense model, but has 47B of "capacity" (learned knowledge). You pay for 13B, get quality approaching 47B.

---

## 8. SPECULATIVE DECODING — 2-4× Faster Inference

**What it is:** A technique that uses a small, fast "draft" model to propose multiple tokens, then uses the large model to verify all of them in a single forward pass — dramatically reducing the number of expensive large-model calls needed.

**Analogy:** In a law firm, a junior lawyer drafts the contract (fast, cheap), and the senior lawyer reviews and approves/edits it in one pass (accurate, expensive). Without this, the senior lawyer would write every word themselves — much slower. Speculative decoding is that junior-senior workflow applied to LLM generation.

### The Problem:
LLM generation is sequential — each token requires a full forward pass.
A 70B model takes ~50ms per token. 200 tokens = 10 seconds. Too slow.

### The Insight:
Most tokens in a sequence are "easy" — common words, predictable continuations.
A tiny model can predict these correctly most of the time.

### How It Works:
```
Small draft model (1B): predicts [" the", " cat", " sat", " on", " the"]
← Fast: 1B model generates 5 candidate tokens in ~5ms total

Large verify model (70B): one forward pass verifies all 5 tokens in parallel
← The 70B model sees all 5 proposed tokens simultaneously
← It computes its own distribution for each position
← For each position, it checks: "does my distribution agree with the draft?"
Result:
  - " the" → accepted (70B model agrees this would be its choice)
  - " cat" → accepted (agrees)
  - " sat" → rejected (large model disagrees — it would say "slept" here)
  → keep " the", " cat", resample from large model at " sat" position
```

One large model forward pass verified 2 tokens instead of generating 1.
If draft accuracy is 80%, you process ~5 tokens per large model forward pass.
**2-4× throughput improvement with identical output quality.**

### Why identical quality?

**What it is:** The mathematical guarantee that speculative decoding produces the exact same distribution as the large model alone.

The acceptance criterion maintains the exact same output distribution as
the large model alone (rejection sampling proof). Mathematically guaranteed.

```python
# Acceptance criterion (rejection sampling):
for each position i:
    p_small = small_model.probability(token_i)  # draft model's probability for this token
    p_large = large_model.probability(token_i)  # large model's probability for this token

    # Accept if small model was at least as confident as large model
    acceptance_prob = min(1, p_large / p_small)
    # p_large > p_small: small model was LESS confident → might accept anyway
    # p_large < p_small: small model was MORE confident → penalize overconfidence

    if random() < acceptance_prob:
        accept token_i  # keep the draft's token
    else:
        # Resample from corrected distribution
        token_i = sample from (p_large - p_small) / normalizer
        # This corrects for the bias introduced by the draft
        break  # stop here, generate rest from large model
# Mathematical guarantee: expected output distribution = large model alone
```

### Used in production by:
- Google (Gemini serving)
- Meta (LLaMA serving)
- DeepMind (Gemma serving)

**WHY speculative decoding matters:** In production, generation latency is the main user-experience factor. A 2-4× speedup without quality loss is enormous. Users go from waiting 10 seconds to waiting 3 seconds — completely different experience.

**Interview answer:** "A small draft model generates K candidate tokens. The large model verifies all K in a single forward pass using an acceptance criterion that maintains the exact output distribution. Tokens that pass are kept; the first rejection causes a resample from the large model. Result: 2-4× throughput with mathematically identical output quality."

---

## 9. CONTEXT LENGTH SCALING — HOW WE GOT TO 1M TOKENS

**What it is:** The engineering techniques that made it possible to extend transformer context windows from 512 tokens (2018) to 1 million tokens (2024).

**Analogy:** Originally, Transformers could read a page at a time. Each technique described here is like adding more reading capacity — first a chapter, then a book, then a library. Each technique removes a different bottleneck.

### The Memory Wall:
Standard attention: O(n²) memory. 1M tokens = 10¹² entries in attention matrix.
At 4 bytes each: 4 terabytes just for one attention layer. Physically impossible.

### Solutions Used:

**1. FlashAttention** — Doesn't materialize n×n matrix. Needed foundation.
← Reduces memory from O(n²) to O(n) for the attention computation itself

**2. Sliding Window Attention (Mistral):**

**What it is:** Each token only attends to its nearby neighbors within a fixed window, rather than all previous tokens. Information propagates globally through multiple layers.

Each token attends to nearest 4096 tokens only.
Through multiple layers, information propagates across full sequence.
```
Layer 1: token 1000 sees tokens 996-1000 (window=4)
Layer 2: token 1000's representation already contains info from 992-1000 (through layer 1 outputs)
Layer 3: token 1000 effectively reaches back to 988-1000
Layer N: information from far back is available through the "telephone chain" of layers
← But this is a GOOD telephone chain — no information loss per hop
```

**3. Sparse Attention (Longformer, BigBird):**

**What it is:** Mix of local (window) attention for most tokens plus a few special "global" tokens that can see everything — dramatically reducing compute while maintaining long-range capability.

Mix of local attention + global attention tokens (CLS-like tokens that attend everywhere).
```
Most tokens: attend to neighbors only (local, cheap)
Special global tokens: attend to all tokens (expensive but few of them)
Result: O(n × window + n × n_global) instead of O(n²)
```

**4. Linear Attention:**
Reformulate softmax(QKᵀ)V to Q(Kᵀ softmax(V)) using kernel trick.
O(n×d²) instead of O(n²×d). Approximate but scalable.
Trade-off: slightly lower quality but handles any length.

**5. Ring Attention:**

**What it is:** Distributing the sequence across multiple GPUs, where each GPU processes its chunk and passes KV information around a ring — eliminating the single-machine memory constraint.

Distribute sequence across multiple GPUs (sequence parallelism).
Each GPU handles a chunk of the sequence, passes KV to next GPU in a ring.
Allows ~1M+ context by distributing across 128 GPUs.
```
GPU 1: processes tokens 1-10K, passes KV to GPU 2
GPU 2: receives KV from GPU 1, processes tokens 10K-20K, passes to GPU 3
...
← Each GPU only stores O(n/128) tokens in memory
← 128 GPUs combined handle 128× more tokens than any single GPU
```

**6. RoPE Scaling:**
NTK-aware scaling allows length generalization beyond training.
← Covered in section 2 above

### Current Context Champions:
| Model | Context | Method |
|-------|---------|--------|
| Gemini 1.5 Pro | 1M tokens | Multi-head attention + efficiency |
| Claude 3 | 200K tokens | Sparse attention techniques |
| GPT-4 Turbo | 128K tokens | FlashAttention + efficiency |
| LLaMA 3.1 | 128K tokens | RoPE scaling + FlashAttention |

---

## 10. THE KV CACHE — Production Critical

**What it is:** The memory buffer that stores the Key and Value matrices from all previously processed tokens, enabling autoregressive generation to skip recomputing them. Understanding KV cache is essential for any LLM production engineer.

### What it is:
Every time a transformer processes a token, it computes K and V for that token.
In autoregressive generation (token by token), these don't change for past tokens.

KV cache = save K and V for all past tokens. Reuse them. Only compute new token's K,V.

### Size calculation:

**What it is:** How to calculate exactly how much memory the KV cache uses — important for capacity planning in production.

```
KV cache size = 2 (K+V) × num_layers × num_kv_heads × d_head × seq_len × bytes_per_element

LLaMA 3 8B at 4096 context, BF16:
= 2 × 32 × 8 × 128 × 4096 × 2 bytes
= 2 × 32 × 8 × 128 × 4096 × 2
= 536 MB per sequence

← 2: both K and V matrices
← 32: number of transformer layers
← 8: number of KV heads (GQA compresses 32 query heads to 8 KV heads)
← 128: d_head (dimension per head = 4096/32 = 128)
← 4096: sequence length (tokens in context)
← 2: bytes per BF16 value

At batch_size=32 (serving 32 users simultaneously):
= 536MB × 32 = 17 GB of KV cache
+ 16GB for BF16 model weights = 33GB total
→ fits on a single A100 40GB GPU with 7GB to spare
```

### KV Cache Optimizations:

**PagedAttention (vLLM):**

**What it is:** Applying virtual memory techniques from operating systems to KV cache management — allocating memory in pages rather than contiguous blocks.

OS-style memory paging. Allocate KV cache in non-contiguous pages.
No fragmentation. 2-4× more sequences served simultaneously.
```
Without PagedAttention:
  KV cache for each sequence is one contiguous block
  Problem: internal fragmentation — allocating 4096 tokens but only using 100
  → 97.6% waste until the sequence grows

With PagedAttention:
  KV cache allocated in small "pages" (e.g., 16 tokens each)
  Multiple sequences share the same physical memory pool
  Pages are allocated and freed as sequences grow and finish
  → Near-zero fragmentation → 2-4× more sequences in same GPU memory
```

**Quantized KV Cache:**
Store KV in INT8 instead of BF16. 2× memory reduction.
Minimal quality loss for most tasks.
```
BF16 KV: 2 bytes per value → 536MB for LLaMA 3 8B at 4096 context
INT8 KV: 1 byte per value  → 268MB — half the memory
INT4 KV: 0.5 bytes        → 134MB — quarter memory (more quality loss)
← Choose based on quality/memory trade-off for your use case
```

**KV Cache Eviction:**

**What it is:** Selectively removing less important tokens from the KV cache when it gets full, enabling effectively-infinite context without infinite memory.

For very long contexts, selectively evict old/unimportant tokens from KV cache.
StreamingLLM: always keep first 4 tokens (attention sinks) + recent window.
```
Keep: [Token 1, Token 2, Token 3, Token 4]  ← "attention sinks" — always high attention
      + [last 1024 tokens]                   ← recent context window
Evict: [tokens 5 through n-1025]             ← far past, lower average attention
Result: constant memory regardless of sequence length — infinite streaming possible
← Quality trade-off: can't access specific facts from evicted past context
```

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

---

## Multi-head Latent Attention (MLA) — DeepSeek-V2's KV Cache Trick

**What it is:** An extreme KV cache compression technique where instead of caching the full K and V matrices, you cache a small "latent" compressed version and decompress it on the fly — achieving 93% KV cache memory reduction.

Standard GQA still caches K and V for every layer. At 128K context, this is huge.

MLA compresses K and V into a LOW-RANK LATENT VECTOR before caching:

```python
# Standard attention caching:
# Cache K: shape (seq_len, n_heads, d_k) = e.g. (128K, 64, 128) = 1 GB per layer
# Cache V: shape (seq_len, n_heads, d_v) = same
# Total per layer: 2 GB

# MLA (Multi-head Latent Attention):
# Instead of caching K and V separately,
# learn a compressed joint representation C:
# C has shape (seq_len, d_c) where d_c << n_heads × d_k
# d_c = 512 vs n_heads × d_k = 64 × 128 = 8192
# Compression ratio: 8192 / 512 = 16× smaller!

# At inference time:
C = x @ W_compress    # compress to latent: shape (seq_len, d_c=512)
← This is what we CACHE — 512 dims instead of 8192 dims per token

# When we need K and V for attention:
K = C @ W_K_up        # decompress back up: (seq_len, n_heads × d_k)
V = C @ W_V_up        # decompress back up: (seq_len, n_heads × d_v)
← These are computed on the fly from the cached C, not stored separately
```

Memory saving: DeepSeek-V2 reduces KV cache by 93.3% vs MHA
- MHA cache per token: 2 × n_layers × n_heads × d_k bytes = 2 × 60 × 128 × 128 = 1.97 MB
- MLA cache per token: 2 × n_layers × d_c bytes = 2 × 60 × 512 = 60 KB
- That's a 33× reduction!

This allows DeepSeek-V2 to serve at much lower cost despite being a 236B MoE model.

**Interview answer:** "MLA is DeepSeek-V2's technique to compress KV cache into low-rank latent vectors, reducing KV cache memory by 93% versus standard MHA while maintaining model quality."

---

## Attention Sink — Why First Tokens Always Get High Attention

**What it is:** An observed phenomenon where the first token(s) in any sequence accumulate disproportionately high attention scores — even when they're irrelevant to the current token — because softmax forces all attention weights to sum to 1.

**Analogy:** Imagine 10 people at a party and you MUST distribute 100% of your attention across them. If 9 people are boring, you can't just give 0% to everyone — softmax forces you to give attention to SOMEONE. So you "dump" your unwanted attention on the nearest person (the BOS token at position 1). They become the "attention sink."

Observation: In every LLM, the first token (BOS or first word) accumulates disproportionately high attention scores, even when irrelevant to the current token.

Why: Softmax must sum to 1. When no token is truly relevant, the model "dumps" attention on a safe token (BOS) rather than spreading evenly — BOS acts as a "sink" for unwanted attention.

Problem for streaming: If you use a sliding window and remove the first token, attention patterns break → model output degrades catastrophically.
```
Normal sliding window (BROKEN):
  Keep last W tokens: [tok_n-W, ..., tok_n]
  Problem: BOS (tok_0) is gone! Attention patterns that relied on "dumping" to BOS
           now have nowhere safe to go → model behavior degrades suddenly

StreamingLLM fix:
  Always keep: [BOS, tok_1, tok_2, tok_3]  ← first 4 "sink" tokens (always preserved)
               + [tok_n-W, ..., tok_n]       ← recent window
  Evict: everything in between
  Result: attention sinks remain available → model behavior stays consistent
          This enables infinite-length streaming inference without recomputation
```

Solution — StreamingLLM (MIT, 2023):
- Keep the first 4 "sink" tokens in the KV cache always
- Use sliding window for the rest
- Enables infinite-length streaming inference without recomputation

**WHY 4 tokens specifically?** Experiments showed the first 4 tokens collectively absorb the "sink" attention. Keeping only 1 sometimes misses it if position 1 doesn't perfectly absorb all sinks. 4 is a safe margin.

**Interview answer:** "Attention sink — the first tokens accumulate disproportionate attention. Remove them and model behavior degrades. StreamingLLM fixes this by always keeping sink tokens in the KV cache."
