# =====================================================================
# PYTHON FOR GenAI ENGINEERS: NUMPY DEEP-DIVE MASTERCLASS
# =====================================================================
# This script covers the ultimate NumPy concepts required for 
# Transformers, Embeddings, RAG, and Deep Learning engineering.
# We focus on the exact patterns used in PyTorch, JAX, and vLLM.
# 
# Run: python practical/13_python_libraries/python_masterclass_numpy.py
# =====================================================================

import numpy as np
import time

def heading(title):
    print("\n" + "=" * 65)
    print(f"[{title}]")
    print("=" * 65)

# ---------------------------------------------------------------------
# 1. VECTORIZATION VS LOOPS (THE PYTHONDIC BOTTLENECK)
# ---------------------------------------------------------------------
# THE PROBLEM:
#    In python, standard loops are slow due to dynamic typing, 
#    overhead, and lack of compiler optimization. In ML, iterating 
#    over millions of embeddings with nested loops will kill your throughput.
#
# THE SOLUTION:
#    Vectorization. NumPy performs operations in highly optimized C code,
#    using SIMD (Single Instruction Multiple Data) at the hardware level.
#
# INTERVIEW QUESTION:
#    "Why is NumPy vectorization faster than Python loops?"
#    "Python loops incur interpreter overhead, dynamic type checking, 
#     and boxing/unboxing for every element. NumPy arrays are contiguous
#     blocks of memory containing uniform types (e.g. float32). Operations 
#     are performed in compiled C code, leveraging SIMD register instructions
#     and cache locality."
# ---------------------------------------------------------------------
heading("1. Vectorization vs Loops (Performance Benchmark)")

# Create two lists of 1 million floats
size = 1_000_000
list_a = [float(x) for x in range(size)]
list_b = [float(x * 1.5) for x in range(size)]

# NumPy arrays
array_a = np.array(list_a, dtype=np.float32)
array_b = np.array(list_b, dtype=np.float32)

# Benchmark 1: Python Loop
start = time.time()
loop_result = []
for i in range(size):
    loop_result.append(list_a[i] + list_b[i])
loop_time = time.time() - start
print(f"[Python] Loop time        : {loop_time:.6f} seconds")

# Benchmark 2: NumPy Vectorized Addition
start = time.time()
vectorized_result = array_a + array_b
numpy_time = time.time() - start
print(f"[NumPy] Vectorization time: {numpy_time:.6f} seconds")
print(f"[Speedup] Factor          : {loop_time / numpy_time:.2f}x faster!")


# ---------------------------------------------------------------------
# 2. MATRIX MULTIPLICATION & TRANSPOSE IN DEEP LEARNING
# ---------------------------------------------------------------------
# THE PROBLEM:
#    In self-attention: Attention(Q, K, V) = Softmax(QK^T / sqrt(d_k))V
#    We must compute:
#    1. Q of shape (SeqLen, d_k) multiplied by K transposed of shape (d_k, SeqLen).
#    2. Slicing multi-head shapes during intermediate projection passes.
#
# KEY OPERATORS:
#    - @ or np.matmul: Standard matrix multiplication.
#    - .T or np.transpose: Flips array dimensions.
#    - .reshape: Changes shape without copying data in memory (if possible).
# ---------------------------------------------------------------------
heading("2. Matrix Operations in Transformer Self-Attention")

# Simulate a batch of 1 sequence, length=3 tokens, hidden_dim=4
seq_len = 3
d_model = 4
np.random.seed(42)

# Simulated input token embeddings
X = np.random.randn(seq_len, d_model)
print("Input Token Embeddings (X):\n", np.round(X, 3))
print("Shape of X:", X.shape)

# Weights to project into Query and Key spaces
W_Q = np.random.randn(d_model, d_model)
W_K = np.random.randn(d_model, d_model)

# Compute Query (Q) and Key (K) representations
Q = X @ W_Q
K = X @ W_K
print("\nQuery Representation (Q) Shape:", Q.shape)
print("Key Representation (K) Shape:", K.shape)

# Calculate Raw Attention Weights (Q @ K^T)
# K has shape (3, 4). K.T has shape (4, 3)
attention_scores = Q @ K.T
print("\nAttention Scores (Q @ K.T) Shape:", attention_scores.shape)
print("Attention Scores Matrix:\n", np.round(attention_scores, 3))


# ---------------------------------------------------------------------
# 3. BROADCASTING (ELIMINATING EXPLICIT TILES & LOOPS)
# ---------------------------------------------------------------------
# THE PROBLEM:
#    Suppose you have a batch of sentence embeddings of shape (Batch, Dim)
#    and you want to add a single bias vector of shape (Dim) to all of them,
#    or subtract the mean vector. Writing loops is slow.
#
# THE RULE OF BROADCASTING:
#    NumPy compares shapes element-wise starting from right to left.
#    Two dimensions are compatible if:
#    1. They are equal.
#    2. One of them is 1.
# ---------------------------------------------------------------------
heading("3. Broadcasting in Deep Learning LayerNorm / Bias Insertion")

# Batch of 3 token vectors, dimension 4
tokens = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0]
]) # Shape: (3, 4)

# Bias vector to add to each token
bias = np.array([0.1, -0.2, 0.5, 1.0]) # Shape: (4,) -> treated as (1, 4)

# Broadcasted addition:
outputs_with_bias = tokens + bias
print("Original Tokens Shape:", tokens.shape)
print("Bias Vector Shape    :", bias.shape)
print("Broadcasted Sum Shape:", outputs_with_bias.shape)
print("Resulting Array:\n", outputs_with_bias)

# LayerNorm-like step: Center each token vector around its own mean
# Mean along the last dimension (axis=1), keepdims=True maintains shape (3, 1)
mean = tokens.mean(axis=1, keepdims=True)
variance = tokens.var(axis=1, keepdims=True)
print("\nMean per Token (keepdims=True) shape:", mean.shape)
print("Mean per Token:\n", mean)

# Subtract mean (broadcasting shape (3, 4) with shape (3, 1))
centered_tokens = tokens - mean
print("\nCentered Tokens:\n", centered_tokens)


# ---------------------------------------------------------------------
# 4. NP.EINSUM (EINSTEIN SUMMATION) - THE DL SUPERPOWER
# ---------------------------------------------------------------------
# THE PROBLEM:
#    Transformer Multi-head Attention involves complex multidimensional products.
#    E.g. Batch matrix multiplication, Batch transposes, dot products.
#    Writing this using matmul and transpose requires multiple reshaping steps
#    which makes code hard to read and prone to bugs.
#
# THE SOLUTION:
#    np.einsum allows you to specify tensor contractions using simple string labels.
#    
#    E.g. Dot Product of 2 vectors a and b:
#    - np.einsum('i,i->', a, b) -> sum over index i.
#
#    E.g. Matrix Multiply A (i, j) by B (j, k):
#    - np.einsum('ij,jk->ik', A, B)
#
#    E.g. Batch Matrix Multiply (BMM) used in Attention:
#    Q (Batch, Heads, SeqLen, HeadDim) and K (Batch, Heads, SeqLen, HeadDim)
#    We want raw weights of shape (Batch, Heads, SeqLen, SeqLen):
#    - np.einsum('bhqd,bhkd->bhqk', Q, K) -> Sum over the head dimension d!
# ---------------------------------------------------------------------
heading("4. Einstein Summation (einsum) for Multi-Head Attention")

# Define inputs
batch_size = 2
num_heads = 4
seq_len = 5
head_dim = 16

# Simulated Multi-Head Query and Key matrices
Q_mha = np.random.randn(batch_size, num_heads, seq_len, head_dim)
K_mha = np.random.randn(batch_size, num_heads, seq_len, head_dim)

# Method 1: Using Einstein Summation (Elegant, robust, self-documenting)
start_einsum = time.time()
scores_einsum = np.einsum('bhqd,bhkd->bhqk', Q_mha, K_mha)
einsum_elapsed = time.time() - start_einsum

# Method 2: Standard implementation using matmul & transpose (messy, error-prone)
start_classic = time.time()
# We must transpose K_mha dimensions from (B, H, S, D) to (B, H, D, S) to do matmul
K_mha_T = np.transpose(K_mha, (0, 1, 3, 2))
scores_classic = Q_mha @ K_mha_T
classic_elapsed = time.time() - start_classic

print(f"einsum Attention Weights Shape: {scores_einsum.shape}")
print(f"classic Attention Weights Shape: {scores_classic.shape}")
print("Check identity of outputs     :", np.allclose(scores_einsum, scores_classic))
print(f"einsum execution time        : {einsum_elapsed:.6f}s")
print(f"classic matmul execution time: {classic_elapsed:.6f}s")


# ---------------------------------------------------------------------
# 5. NUMERICAL STABILITY: PREVENTING OVERFLOW & UNDERFLOW
# ---------------------------------------------------------------------
# THE PROBLEM:
#    Calculating Softmax: Softmax(x_i) = exp(x_i) / sum(exp(x_j))
#    If the inputs x_i are large (e.g. 1000.0), np.exp(1000) will overflow to inf
#    resulting in inf / inf = nan (Not a Number). This breaks your neural network.
#
# THE SOLUTION:
#    Subtract the maximum value from all inputs:
#    exp(x_i - max(x)) / sum(exp(x_j - max(x)))
#    This has the exact same mathematical output but keeps numbers <= 0,
#    restricting output of exp() strictly between 0 and 1.
# ---------------------------------------------------------------------
heading("5. Numerical Stability (Stable Softmax Demo)")

# Unstable array (simulating raw logit inputs before softmax)
logits_unstable = np.array([1000.0, 1001.0, 999.0])

def unstable_softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def stable_softmax(x):
    # Subtract max element for stability
    shift_x = x - np.max(x)
    exp_x = np.exp(shift_x)
    return exp_x / np.sum(exp_x)

print("Raw inputs:", logits_unstable)

try:
    res = unstable_softmax(logits_unstable)
    print("Unstable Softmax Output:", res)
except Exception as e:
    print("Unstable Softmax failed!")

res_stable = stable_softmax(logits_unstable)
print("Stable Softmax Output  :", res_stable)
print("Proof it sums to 1.0   :", np.sum(res_stable))


# =====================================================================
# INTERVIEW PRACTICE QUESTIONS (NUMPY)
# =====================================================================
print("\n" + "=" * 65)
print("[PRO-LEVEL INTERVIEW DRILLS (NUMPY FOR LLMs)]")
print("=" * 65)
print("""
Q1: When computing cosine similarity between a 768-dim query embedding and 
    10,000 document embeddings, what is the most vectorized way to do this in NumPy?
    
    -> Answer: Normalize both query and document embeddings along their respective 
       axes first. Then use matrix-vector multiplication: 'scores = docs @ query'. 
       This avoids computing norms inside loops and calculates all similarities 
       simultaneously in single SIMD-optimized instruction.

Q2: Why do we use keepdims=True when computing means or norms in neural network layers?
    
    -> Answer: 'keepdims=True' preserves the dimensions of the reduced axes as size 1. 
       This ensures the output shape remains compatible with original shapes, 
       allowing automatic broadcasting to function without manually adding axes using 
       'np.newaxis'.

Q3: What is the benefit of einsum in writing deep learning architectures?
    
    -> Answer: einsum replaces combinations of 'transpose', 'reshape', 'matmul', 
       and element-wise multiplication with a single concise Einstein index string. 
       It is highly readable, prevents reshaping errors, and allows the underlying 
       BLAS library to optimize memory footprint and computations under the hood.
""")
print("=" * 65 + "\n")
