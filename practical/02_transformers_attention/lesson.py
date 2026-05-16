# ==============================================================
# SESSION 2: Transformers & Attention — Production Level
# ==============================================================
# The #1 most asked topic in GenAI interviews.
# We build EVERYTHING from scratch using only numpy.
# Run: python lesson.py
# ==============================================================

import numpy as np

# --------------------------------------------------------------
# CONCEPT 1: Why Attention? The Problem with RNNs
# --------------------------------------------------------------
# THEORY:
#   Before 2017, NLP used RNNs (Recurrent Neural Networks).
#   RNNs process tokens ONE BY ONE — sequentially.
#
#   Problems with RNNs:
#   1. SLOW: can't parallelize (token N needs token N-1 to finish)
#   2. FORGETS: "The cat that sat on the mat was..." — by "was",
#      RNN has almost forgotten "cat" (vanishing gradient)
#   3. FIXED bottleneck: encoder compresses entire sentence into
#      one vector — loses information
#
#   Transformers solved ALL THREE:
#   1. Parallel: all tokens processed simultaneously
#   2. Direct: any token can attend to any other token directly
#   3. No bottleneck: attention uses all encoder outputs
#
# INTERVIEW ANSWER:
#   "RNNs process tokens sequentially, can't parallelize, and
#    struggle with long-range dependencies due to vanishing
#    gradients. Transformers use attention to let every token
#    directly relate to every other token in parallel."
# --------------------------------------------------------------

print("=" * 60)
print("CONCEPT 1: Why Transformers?")
print("=" * 60)
print("""
RNN problems:
  - Sequential: token[t] needs token[t-1] -> slow training
  - Vanishing gradient: early tokens forgotten in long sequences
  - Bottleneck: whole sentence compressed into 1 hidden vector

Transformer solutions:
  - Parallel: ALL tokens processed simultaneously
  - Attention: token[i] directly attends to token[j] regardless of distance
  - No bottleneck: uses ALL encoder outputs via cross-attention
""")


# --------------------------------------------------------------
# CONCEPT 2: Self-Attention — Built from Scratch
# --------------------------------------------------------------
# THEORY:
#   Every token creates 3 vectors from its embedding:
#   - Q (Query):  "What information am I looking for?"
#   - K (Key):    "What information do I contain?"
#   - V (Value):  "What will I actually give if selected?"
#
#   The formula:
#   Attention(Q, K, V) = softmax( Q @ K.T / sqrt(d_k) ) @ V
#
#   Step by step:
#   1. Q @ K.T  → similarity score between every pair of tokens
#   2. / sqrt(d_k) → scale down to prevent softmax saturation
#   3. softmax → convert scores to probabilities (sum=1)
#   4. @ V → weighted sum of values based on attention weights
#
# WHY sqrt(d_k)?
#   For large d_k, dot products grow large -> softmax becomes
#   very "peaked" (one weight near 1, rest near 0) -> gradients
#   vanish. Dividing by sqrt(d_k) keeps values in stable range.
#
# INTERVIEW ANSWER:
#   "Each token projects into Q, K, V vectors. We compute dot
#    products of Q with all Ks to get similarity scores, scale
#    by sqrt(d_k) for numerical stability, apply softmax to get
#    attention weights, then take a weighted sum of Vs.
#    Result: each token's output is a blend of ALL other tokens,
#    weighted by relevance."
# --------------------------------------------------------------

print("=" * 60)
print("CONCEPT 2: Self-Attention From Scratch")
print("=" * 60)

# Simulate a sentence: 4 tokens, each with 8-dim embedding
# In real GPT-2: 768-dim. We use 8 for clarity.
np.random.seed(42)
seq_len = 4          # "The cat sat on"
d_model = 8          # embedding dimension
d_k     = 8          # key/query dimension (= d_model for single head)

# Simulated token embeddings (input to transformer layer)
X = np.random.randn(seq_len, d_model)
print(f"\nInput X shape: {X.shape}  (seq_len=4 tokens, d_model=8)")

# Weight matrices — LEARNED during training
# In real models these are huge: (768, 64) per head
W_Q = np.random.randn(d_model, d_k) * 0.1
W_K = np.random.randn(d_model, d_k) * 0.1
W_V = np.random.randn(d_model, d_k) * 0.1

# Step 1: Project input into Q, K, V
Q = X @ W_Q     # shape: (4, 8)
K = X @ W_K     # shape: (4, 8)
V = X @ W_V     # shape: (4, 8)
print(f"\nQ shape: {Q.shape}  (each of 4 tokens has a 8-dim query vector)")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")

# Step 2: Compute attention scores = Q @ K.T
scores = Q @ K.T          # shape: (4, 4) — every token vs every token
print(f"\nRaw scores shape: {scores.shape}  (4x4: token[i] vs token[j])")
print("Raw scores (before scaling):")
print(np.round(scores, 3))

# Step 3: Scale by sqrt(d_k) — CRITICAL for stability
scores_scaled = scores / np.sqrt(d_k)
print(f"\nScaled scores (divided by sqrt({d_k})={np.sqrt(d_k):.2f}):")
print(np.round(scores_scaled, 3))

# Step 4: Softmax — convert to probabilities
def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x = x - x.max(axis=axis, keepdims=True)  # subtract max for stability
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

attention_weights = softmax(scores_scaled)   # shape: (4, 4)
print(f"\nAttention weights (after softmax) — each ROW sums to 1:")
print(np.round(attention_weights, 3))
print(f"Row sums: {attention_weights.sum(axis=1).round(4)}  <- all 1.0")

# Step 5: Weighted sum of Values
output = attention_weights @ V   # shape: (4, 8)
print(f"\nAttention output shape: {output.shape}")
print("Each token's output is now a BLEND of all other tokens' values,")
print("weighted by how much attention was paid to each.")


# --------------------------------------------------------------
# CONCEPT 3: Causal Masking — Why GPT Needs It
# --------------------------------------------------------------
# THEORY:
#   GPT is AUTOREGRESSIVE: it generates the next token using
#   only PAST tokens. During training, we feed the full sequence
#   but must prevent each position from seeing FUTURE positions.
#
#   "The cat sat on the mat"
#   When predicting "sat", model should only see "The cat"
#   When predicting "on", model should only see "The cat sat"
#
#   Implementation: set future positions to -infinity BEFORE softmax
#   -inf -> softmax -> 0 (completely ignored)
#
# INTERVIEW ANSWER:
#   "Causal masking prevents each token from attending to future
#    tokens during training. We set those attention scores to -inf
#    before softmax, making them 0 after softmax. This ensures
#    GPT learns true next-token prediction, not cheating by
#    looking ahead. At inference, future tokens don't exist yet."
# --------------------------------------------------------------

print("\n" + "=" * 60)
print("CONCEPT 3: Causal Masking (GPT-style)")
print("=" * 60)

def causal_mask(seq_len):
    """
    Creates upper-triangular mask of -inf.
    Token i can only attend to tokens 0..i (not i+1..N).
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)  # upper triangle
    mask[mask == 1] = -np.inf   # blocked positions -> -inf -> 0 after softmax
    return mask

mask = causal_mask(seq_len)
print(f"\nCausal mask ({seq_len}x{seq_len}):")
print(mask)
print("0 = can attend, -inf = BLOCKED (future token)")

# Apply mask to scores
masked_scores = scores_scaled + mask
print(f"\nMasked scores (blocked positions are -inf):")
print(np.round(masked_scores, 3))

# After softmax: -inf becomes 0
masked_weights = softmax(masked_scores)
print(f"\nMasked attention weights (upper triangle is all 0):")
print(np.round(masked_weights, 3))
print("Token 0 only attends to itself.")
print("Token 3 attends to tokens 0,1,2,3 (all past + current).")


# --------------------------------------------------------------
# CONCEPT 4: Multi-Head Attention
# --------------------------------------------------------------
# THEORY:
#   Instead of ONE attention operation, run H in parallel.
#   Each head uses DIFFERENT W_Q, W_K, W_V projections.
#   Each head learns to attend to DIFFERENT relationships:
#   - Head 1: syntactic (subject-verb agreement)
#   - Head 2: semantic (word meaning)
#   - Head 3: positional (nearby tokens)
#   etc.
#
#   d_model is split across heads: d_k = d_model / num_heads
#   GPT-2 small: d_model=768, 12 heads -> d_k=64 per head
#
#   After all heads: concatenate + project back to d_model
#   MultiHead(Q,K,V) = Concat(head_1,...,head_h) @ W_O
#
# INTERVIEW ANSWER:
#   "Multi-head attention runs H attention operations in parallel,
#    each with different learned projections. We split d_model into
#    H heads of size d_k=d_model/H. Each head captures different
#    relationships. Outputs are concatenated and projected back.
#    More heads = richer representations at same compute cost."
# --------------------------------------------------------------

print("\n" + "=" * 60)
print("CONCEPT 4: Multi-Head Attention")
print("=" * 60)

def multi_head_attention(X, num_heads=2, causal=False):
    """
    Full multi-head attention implementation.
    X shape: (seq_len, d_model)
    """
    seq_len, d_model = X.shape
    assert d_model % num_heads == 0
    d_k = d_model // num_heads

    outputs = []
    for head in range(num_heads):
        # Each head has its own weight matrices (different projections)
        np.random.seed(head)
        W_Q_h = np.random.randn(d_model, d_k) * 0.1
        W_K_h = np.random.randn(d_model, d_k) * 0.1
        W_V_h = np.random.randn(d_model, d_k) * 0.1

        Q_h = X @ W_Q_h   # (seq_len, d_k)
        K_h = X @ W_K_h
        V_h = X @ W_V_h

        scores_h = Q_h @ K_h.T / np.sqrt(d_k)

        if causal:
            scores_h = scores_h + causal_mask(seq_len)

        weights_h = softmax(scores_h)
        out_h = weights_h @ V_h   # (seq_len, d_k)
        outputs.append(out_h)
        print(f"  Head {head+1}: Q@K.T shape={scores_h.shape}, output shape={out_h.shape}")

    # Concatenate all heads: (seq_len, d_model)
    concat = np.concatenate(outputs, axis=-1)

    # Final projection W_O: (d_model, d_model)
    W_O = np.random.randn(d_model, d_model) * 0.1
    return concat @ W_O   # (seq_len, d_model)

print(f"\nRunning Multi-Head Attention (2 heads, d_model=8, d_k=4 each):")
mha_output = multi_head_attention(X, num_heads=2, causal=True)
print(f"\nFinal MHA output shape: {mha_output.shape}  <- same as input!")
print("Each token's representation is now enriched from all other tokens.")


# --------------------------------------------------------------
# CONCEPT 5: Positional Encoding
# --------------------------------------------------------------
# THEORY:
#   Attention is PERMUTATION INVARIANT — it doesn't care about order.
#   "The cat sat on" = "sat cat on The" to attention alone!
#   We need to inject position information.
#
#   Original (sinusoidal) positional encoding:
#   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
#   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
#
#   Modern: RoPE (Rotary Position Embedding) — used in LLaMA
#   Instead of adding position to embedding, it ROTATES Q and K
#   vectors in a way that encodes RELATIVE position.
#   Much better at generalizing to longer sequences than trained on.
#
# INTERVIEW ANSWER:
#   "Self-attention is permutation-invariant — it has no notion of
#    order. Positional encoding injects position info. Original
#    transformers used sinusoidal encoding added to embeddings.
#    Modern LLMs (LLaMA, Mistral) use RoPE which encodes relative
#    position in the Q/K rotation — better for long contexts."
# --------------------------------------------------------------

print("\n" + "=" * 60)
print("CONCEPT 5: Positional Encoding")
print("=" * 60)

def sinusoidal_positional_encoding(seq_len, d_model):
    """
    Original Transformer positional encoding (Vaswani et al. 2017).
    Returns shape: (seq_len, d_model)
    """
    PE = np.zeros((seq_len, d_model))
    positions = np.arange(seq_len).reshape(-1, 1)        # (seq_len, 1)
    dims      = np.arange(0, d_model, 2)                 # even indices
    div_term  = np.power(10000, dims / d_model)           # 10000^(2i/d)

    PE[:, 0::2] = np.sin(positions / div_term)  # even dims -> sin
    PE[:, 1::2] = np.cos(positions / div_term)  # odd dims  -> cos
    return PE

PE = sinusoidal_positional_encoding(seq_len=4, d_model=8)
print(f"\nPositional encoding shape: {PE.shape}")
print("Positional encoding values (each row = one position):")
print(np.round(PE, 3))

# Add to token embeddings — this is what actually goes into the transformer
X_with_position = X + PE
print(f"\nFinal input to transformer (embedding + position): {X_with_position.shape}")
print("Now the model knows BOTH the meaning AND the position of each token.")


# --------------------------------------------------------------
# CONCEPT 6: KV Cache — Why It's Critical in Production
# --------------------------------------------------------------
# THEORY:
#   During inference (generating tokens one by one):
#   - Token 1: compute K1, V1
#   - Token 2: compute K1, V1 AGAIN + K2, V2
#   - Token 3: compute K1, V1, K2, V2 AGAIN + K3, V3
#   This is O(n^2) compute per step — extremely wasteful!
#
#   KV Cache: store K and V for all past tokens.
#   - Token 1: compute K1, V1 -> CACHE them
#   - Token 2: REUSE K1,V1 from cache + compute K2, V2 -> cache
#   - Token N: only compute KN, VN. Reuse everything else.
#   This is O(n) per step!
#
#   COST: GPU memory. A 7B model with 32 layers, 32 heads,
#   d_k=128, for 4096 tokens:
#   32 layers x 2 (K+V) x 32 heads x 4096 tokens x 128 dims x 2 bytes
#   = ~2GB just for KV cache!
#
#   PagedAttention (vLLM): manages KV cache in non-contiguous pages
#   like OS virtual memory -> 20-100x more efficient serving.
#
# INTERVIEW ANSWER:
#   "KV cache stores the Key and Value matrices for all previous
#    tokens during autoregressive generation. Without it: O(n^2)
#    compute per token. With it: O(n). In production, the KV cache
#    can be GBs for long contexts. vLLM's PagedAttention manages
#    KV cache in non-contiguous pages, eliminating fragmentation
#    and enabling much higher throughput."
# --------------------------------------------------------------

print("\n" + "=" * 60)
print("CONCEPT 6: KV Cache — Production Memory Impact")
print("=" * 60)

def kv_cache_memory_gb(
    num_layers, num_heads, seq_len, d_k, bytes_per_param=2
):
    """Calculate KV cache memory in GB for a given model config."""
    # K cache: num_layers x num_heads x seq_len x d_k
    # V cache: same size
    total_elements = 2 * num_layers * num_heads * seq_len * d_k
    total_bytes = total_elements * bytes_per_param
    return total_bytes / (1024 ** 3)

print("\nKV Cache memory for different models at different context lengths:")
print(f"{'Model':<20} {'Layers':<8} {'Heads':<8} {'d_k':<6} {'2K ctx':<10} {'8K ctx':<10} {'32K ctx'}")
print("-" * 75)

configs = [
    ("GPT-2 small",  12, 12, 64),
    ("LLaMA-2 7B",   32, 32, 128),
    ("LLaMA-2 70B",  80, 64, 128),
    ("GPT-4 (est.)", 96, 96, 128),
]

for name, layers, heads, dk in configs:
    mem_2k  = kv_cache_memory_gb(layers, heads, 2048,  dk)
    mem_8k  = kv_cache_memory_gb(layers, heads, 8192,  dk)
    mem_32k = kv_cache_memory_gb(layers, heads, 32768, dk)
    print(f"{name:<20} {layers:<8} {heads:<8} {dk:<6} {mem_2k:<10.2f} {mem_8k:<10.2f} {mem_32k:.2f} GB")

print("\nThis is why long contexts are expensive!")
print("vLLM's PagedAttention solves KV cache fragmentation for production.")


# ==============================================================
# INTERVIEW CHEAT SHEET — Transformers & Attention
# ==============================================================
print("\n" + "=" * 60)
print("INTERVIEW CHEAT SHEET — Transformers & Attention")
print("=" * 60)
print("""
THE FORMULA (say this in your interview):
  Attention(Q,K,V) = softmax( Q @ K.T / sqrt(d_k) ) @ V

WHY sqrt(d_k)?
  Large d_k -> large dot products -> peaked softmax -> vanishing
  gradients. Scaling by sqrt(d_k) keeps values stable.

CAUSAL MASK:
  GPT sets future positions to -inf before softmax -> 0 after.
  Prevents cheating during training (can't see future tokens).

MULTI-HEAD:
  H heads, each d_k = d_model/H. Different heads learn different
  relationships (syntactic, semantic, positional).
  Output: Concat(heads) @ W_O -> (seq_len, d_model)

POSITIONAL ENCODING:
  Attention is order-blind. PE injects position info.
  Original: sinusoidal. Modern: RoPE (LLaMA, Mistral).
  RoPE encodes RELATIVE position by rotating Q/K vectors.

KV CACHE:
  Cache K,V for past tokens during generation.
  Without: O(n^2) per step. With: O(n) per step.
  vLLM PagedAttention: KV cache in non-contiguous pages.

BERT vs GPT:
  BERT = encoder-only, bidirectional attention, MLM training
  GPT  = decoder-only, CAUSAL attention, next-token prediction
""")
