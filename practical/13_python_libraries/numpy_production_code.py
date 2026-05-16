"""
PRODUCTION NUMPY — How it's actually used in LLM/ML jobs
==========================================================
Real scenario: You just retrieved 5 document chunks from a vector DB.
You have their embeddings. You need to rank them by similarity to
the user's query. This is what you write at work every day.

Run this file: python 01_numpy_production.py
pip install numpy
"""

import numpy as np

print("=" * 65)
print("SCENARIO 1: Cosine Similarity — The Heart of RAG Retrieval")
print("=" * 65)

# In production, your embedding model gives you vectors like these
# (in real life these are 768 or 1024 dimensional, we use 4 for clarity)

query_embedding = np.array([0.2, 0.8, 0.1, 0.5])

document_embeddings = np.array([
    [0.21, 0.79, 0.12, 0.48],   # doc 0: very similar to query
    [0.9,  0.1,  0.8,  0.2 ],   # doc 1: not similar
    [0.19, 0.82, 0.09, 0.51],   # doc 2: most similar
    [0.5,  0.5,  0.5,  0.5 ],   # doc 3: moderately similar
    [0.8,  0.15, 0.9,  0.1 ],   # doc 4: not similar
])

# --- PRODUCTION WAY: vectorized, no loops ---
def cosine_similarity_batch(query, docs):
    """
    Compute cosine similarity between one query and many documents.
    This is the EXACT operation your vector DB runs internally.
    """
    # Normalize query
    query_norm = query / np.linalg.norm(query)

    # Normalize all documents at once (axis=1 = row-wise)
    docs_norm = docs / np.linalg.norm(docs, axis=1, keepdims=True)

    # Dot product: shape (num_docs,)
    scores = docs_norm @ query_norm
    return scores

scores = cosine_similarity_batch(query_embedding, document_embeddings)

print("\nSimilarity scores:")
for i, score in enumerate(scores):
    print(f"  Doc {i}: {score:.4f}")

# Sort and get top-3 (this is what vector DB returns as top-K)
top_k_indices = np.argsort(scores)[::-1][:3]   # descending, top 3
print(f"\nTop-3 most relevant doc indices: {top_k_indices}")
print(f"Top-3 scores: {scores[top_k_indices]}")

# KEY INTERVIEW POINT:
# "In RAG, after chunking and embedding documents, we run cosine
#  similarity between the query embedding and all document embeddings.
#  np.argsort gives us the ranking. In production, a vector DB like
#  FAISS or Qdrant does this efficiently for millions of vectors."


print("\n" + "=" * 65)
print("SCENARIO 2: Softmax — What Happens Inside Attention")
print("=" * 65)

# Raw attention scores (QK^T / sqrt(d_k)) before softmax
# In real GPT-2 with 12 heads, d_k=64 — here we use 5 tokens
raw_attention_scores = np.array([2.1, 0.3, -1.2, 0.8, 1.5])

def softmax(x):
    """
    Numerically stable softmax.
    WHY subtract max? Large values cause overflow in exp().
    e^1000 = inf. e^(1000-1000) = e^0 = 1. Same result, no overflow.
    """
    x_shifted = x - np.max(x)          # numerical stability trick
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum()

attention_weights = softmax(raw_attention_scores)

print(f"\nRaw scores:        {raw_attention_scores}")
print(f"Attention weights: {np.round(attention_weights, 4)}")
print(f"Sum (must be 1.0): {attention_weights.sum():.6f}")

# Effect of temperature on softmax
print("\n--- Temperature Effect on Attention (same concept as LLM sampling) ---")
for temp in [0.1, 0.5, 1.0, 2.0]:
    weights = softmax(raw_attention_scores / temp)
    print(f"  temp={temp}: {np.round(weights, 3)}  ← {'peaked/focused' if temp < 1 else 'flat/random' if temp > 1 else 'normal'}")

# KEY INTERVIEW POINT:
# "Softmax converts raw attention scores into a probability distribution
#  that sums to 1. Temperature < 1 sharpens it (model focuses on top tokens),
#  temperature > 1 flattens it (more random generation)."


print("\n" + "=" * 65)
print("SCENARIO 3: Batch Matrix Multiply — Multi-Head Attention Shape")
print("=" * 65)

# In real transformer:
# Input shape: (batch_size, seq_len, d_model)
# After QKV projections in multi-head attention

batch_size = 2
seq_len    = 10
d_model    = 512
num_heads  = 8
d_k        = d_model // num_heads   # = 64

# Simulate Q, K, V after projection and reshape
Q = np.random.randn(batch_size, num_heads, seq_len, d_k)
K = np.random.randn(batch_size, num_heads, seq_len, d_k)
V = np.random.randn(batch_size, num_heads, seq_len, d_k)

print(f"\nQ shape: {Q.shape}  (batch, heads, seq_len, d_k)")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")

# Attention scores: Q @ K^T / sqrt(d_k)
# np.einsum('bhqd,bhkd->bhqk', Q, K) = for each batch, each head:
#   multiply Q (seq_len, d_k) @ K^T (d_k, seq_len) = (seq_len, seq_len)
scores = np.einsum('bhqd,bhkd->bhqk', Q, K) / np.sqrt(d_k)
print(f"\nAttention scores shape: {scores.shape}  (batch, heads, seq, seq)")

# Apply softmax over last dimension (key dimension)
def softmax_axis(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

attention_weights_mha = softmax_axis(scores)
print(f"Attention weights shape: {attention_weights_mha.shape}")

# Weighted sum of values
output = np.einsum('bhqk,bhkd->bhqd', attention_weights_mha, V)
print(f"Attention output shape: {output.shape}  (batch, heads, seq, d_k)")

# Concatenate heads → back to (batch, seq_len, d_model)
output_concat = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
print(f"After concat heads: {output_concat.shape}  (batch, seq_len, d_model)")

# KEY INTERVIEW POINT:
# "Multi-head attention runs h attention operations in parallel.
#  We split d_model into h heads of size d_k = d_model/h.
#  Each head learns different relationships. Then we concatenate
#  the outputs back to d_model."


print("\n" + "=" * 65)
print("SCENARIO 4: Dataset Statistics — Real Pre-Processing Work")
print("=" * 65)

# Before training, you ALWAYS analyze your dataset.
# This is what you do at the start of every fine-tuning job.

# Simulated token counts of your training dataset
np.random.seed(42)
token_counts = np.random.exponential(scale=300, size=10000).astype(int) + 10

print(f"\nDataset: {len(token_counts):,} samples")
print(f"  Mean tokens:   {token_counts.mean():.1f}")
print(f"  Median tokens: {np.median(token_counts):.1f}")
print(f"  Max tokens:    {token_counts.max()}")
print(f"  Min tokens:    {token_counts.min()}")
print(f"  Std dev:       {token_counts.std():.1f}")

# Check: how many samples fit in a 2048-token context window?
within_2048 = (token_counts <= 2048).sum()
print(f"\n  Samples within 2048 tokens: {within_2048:,} ({100*within_2048/len(token_counts):.1f}%)")
print(f"  Samples truncated if max_len=2048: {len(token_counts)-within_2048:,}")

# Percentiles — crucial for deciding max_seq_length
percentiles = [50, 75, 90, 95, 99]
pct_values = np.percentile(token_counts, percentiles)
print("\n  Percentiles:")
for p, v in zip(percentiles, pct_values):
    print(f"    p{p}: {v:.0f} tokens")

# KEY INTERVIEW POINT:
# "Before fine-tuning, I always analyze token length distribution.
#  Setting max_seq_length at p95 covers 95% of samples without
#  wasting memory on padding. Samples above that get truncated."


print("\n" + "=" * 65)
print("SCENARIO 5: Embedding Normalization — Critical for Production RAG")
print("=" * 65)

# Many embedding models return un-normalized vectors.
# Cosine similarity REQUIRES normalized (unit) vectors.
# If not normalized: dot product ≠ cosine similarity.

raw_embeddings = np.random.randn(5, 768)   # 5 docs, 768-dim embeddings

print(f"Before normalization, norms: {np.linalg.norm(raw_embeddings, axis=1).round(3)}")

# Normalize to unit vectors
norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
normalized_embeddings = raw_embeddings / norms

print(f"After normalization, norms:  {np.linalg.norm(normalized_embeddings, axis=1).round(3)}")
print("(All should be 1.0)")

# Now dot product = cosine similarity
similarity_matrix = normalized_embeddings @ normalized_embeddings.T
print(f"\nSelf-similarity matrix shape: {similarity_matrix.shape}")
print("(diagonal should be 1.0 — each doc is identical to itself)")
print(np.round(np.diag(similarity_matrix), 4))

# KEY INTERVIEW POINT:
# "In production RAG, I always normalize embeddings before storing
#  in the vector DB. This makes dot product equivalent to cosine
#  similarity, which many ANN libraries (FAISS) optimize for."


print("\n" + "=" * 65)
print("COMMON NUMPY MISTAKES IN INTERVIEWS")
print("=" * 65)
print("""
MISTAKE 1: Using Python loops instead of vectorized ops
  BAD:   scores = [np.dot(q, d) for d in docs]  # slow
  GOOD:  scores = docs @ query                   # fast

MISTAKE 2: Not handling numerical stability in softmax
  BAD:   np.exp(x) / np.exp(x).sum()   # overflows for large x
  GOOD:  x -= x.max(); np.exp(x) / np.exp(x).sum()

MISTAKE 3: Wrong axis in normalization
  BAD:   x / np.linalg.norm(x)               # treats as flat vector
  GOOD:  x / np.linalg.norm(x, axis=1, keepdims=True)  # row-wise

MISTAKE 4: Forgetting shape rules in matmul
  (10, 768) @ (768,)    → (10,)     ✓
  (10, 768) @ (10, 768) → ERROR     ✗ (need transpose)
  (10, 768) @ (768, 10) → (10, 10)  ✓
""")
