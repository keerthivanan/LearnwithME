# 02 — Transformers & Attention Mechanisms

> The core architecture behind every modern LLM. You MUST know this deeply.

---

## 1. Why Transformers? (The Problem They Solved)

**What it is:** The reason Transformers were invented. Understanding the "why" makes the "what" easier to remember.

Before Transformers (2017), NLP used **RNNs and LSTMs**:
- Processed tokens **sequentially** (one by one) — could not parallelize
- Struggled with **long-range dependencies** (forgot early context after many steps)
- Could not be **parallelized** during training (each step depends on previous step's output)

**Transformers** solved all three:
- Process all tokens **in parallel** — use all GPU cores at once
- Use **attention** to relate any token to any other token directly — no forgetting
- Scale efficiently to billions of parameters because training is fully parallel

**Analogy:** RNNs are like a relay race — each runner must wait for the previous one to finish and hand off the baton before they can start. Transformers are like a 100-meter sprint — all runners go at once, but each runner can radio any other runner in real time. Same distance, drastically different speed.

> "Attention Is All You Need" — Vaswani et al., 2017 (Google Brain)

**WHY this matters:** Every LLM you'll work with — GPT-4, LLaMA, Claude, Gemini — is a Transformer. Understanding this architecture is not optional for a GenAI engineer. This is your foundation.

---

## 2. High-Level Transformer Architecture

**What it is:** The complete bird's-eye view of what a Transformer looks like, before diving into each piece.

```
Input Tokens
     ↓  Raw text is broken into integer IDs (e.g., "cat" → 3797)
     ↓  These IDs are meaningless numbers to the model — just indices
Token Embeddings + Positional Encoding
     ↓  Each ID becomes a learned vector; position info added on top
     ↓  Now the model has meaning (embedding) + position (PE) for each token
┌─────────────────────────┐
│   Transformer Block     │  ×N layers   ← stacked N times (e.g., 12 for GPT-2 small)
│  ┌───────────────────┐  │             ← each block adds one level of understanding
│  │  Multi-Head       │  │  ← attention: every token looks at every other token
│  │  Self-Attention   │  │  ← decides how much to "borrow" from each other token
│  └────────┬──────────┘  │
│           │ + Residual  │  ← input added to output: prevents vanishing gradients
│      Layer Norm         │  ← normalizes scale before next component
│  ┌───────────────────┐  │
│  │  Feed-Forward     │  │  ← 2-layer MLP: processes each token independently
│  │  Network (FFN)    │  │  ← stores factual knowledge, adds non-linearity
│  └────────┬──────────┘  │
│           │ + Residual  │  ← second gradient highway
│      Layer Norm         │  ← second normalization
└─────────────────────────┘
     ↓  After all N blocks, each token has a rich contextual representation
     ↓  Lower layers = syntax; middle layers = semantics; upper layers = task reasoning
Output (logits over vocabulary)
     ↓  Project to vocab size → softmax → probability of each possible next word
```

**WHY is it stacked N times?** Each block adds one layer of understanding. Lower layers capture syntax (is "bank" a noun?), middle layers capture semantics (is "bank" financial or geographical?), upper layers capture task-specific reasoning. More layers = more abstraction levels. This is why a 96-layer model (GPT-3) is so much more capable than a 12-layer model (GPT-2 small).

---

## 3. Attention Mechanism — The Heart of Transformers

### What is Attention?

**What it is:** A mechanism that lets each token "look at" all other tokens and decide how much to borrow from each one's representation.

A mechanism that lets each token **look at all other tokens** and decide how much to "attend" to each one.

Think of it as: *"When processing the word 'bank', should I pay more attention to 'river' or 'money' in this sentence?"*

**Analogy:** When proofreading a sentence, your eye doesn't read linearly — it jumps back and forth between related words. "He gave it to her" → your eye connects "he" to its antecedent and "it" to what was given. Attention formalizes this jumping using math instead of instinct.

### Scaled Dot-Product Attention

**What it is:** The specific mathematical formula for computing attention — the core equation of the entire Transformer.

Every token produces three vectors:
- **Q (Query)**: What am I looking for? ("I'm looking for my subject")
- **K (Key)**: What do I contain? ("I am the subject of this sentence")
- **V (Value)**: What do I actually give? ("Here's all my linguistic information")

**WHY three vectors?** Separating "what to search for" (Q), "what to advertise" (K), and "what to share" (V) allows the model to decouple different roles a word plays. A word can be searched for one thing, advertised as another, and provide yet another type of information. It's a powerful separation of concerns.

```
Attention(Q, K, V) = softmax( QK^T / √d_k ) * V
```

```python
import math
import torch
import torch.nn.functional as F

# Q: Query matrix, shape (seq_len, d_k) — what each token is "searching for"
# K: Key matrix, shape (seq_len, d_k) — what each token "advertises" as containing
# V: Value matrix, shape (seq_len, d_v) — what each token "shares" when attended to
# d_k: dimension of keys (e.g., 64 for GPT-2 small)

# Step 1: Compute raw similarity scores between every query and every key
scores = Q @ K.transpose(-2, -1)
# Q @ K.T: dot product of every query with every key
# Shape: (seq_len, seq_len) — entry [i, j] = how much token i "wants" token j
# High score = "I'm looking for something and this token has it"
# Low score = "this token is irrelevant to what I need"

# Step 2: Scale down to prevent softmax saturation
scores = scores / math.sqrt(d_k)
# WHY scale by sqrt(d_k)?
# For d_k=64, random vectors have dot products with std = sqrt(64) = 8
# Without scaling: values ±8 → softmax → one value near 1, all others near 0
# That near-one-hot distribution has near-zero gradients → no learning
# With scaling: values ±1 → softmax → smooth probability distribution
# Smooth distribution → useful gradients → model actually learns

# Step 3: Convert scores to probabilities that sum to 1
attention_weights = F.softmax(scores, dim=-1)
# softmax: exponentiates each score, then divides by sum
# e^2 / (e^2 + e^1 + e^0 + ...) — high scores dominate
# High scores become high probabilities; low scores become near-zero
# Each row sums to 1 → "token i distributes 100% of its attention across all tokens"
# Example: token "it" → {"bank": 0.72, "deposits": 0.18, "costs": 0.06, ...}

# Step 4: Weighted sum of Value vectors
output = attention_weights @ V
# attention_weights[i] is token i's probability distribution over all tokens
# @ V: weighted sum → output[i] = sum(attention_weights[i][j] * V[j]) for all j
# Token "it" gets 72% of bank's info, 18% of deposits' info, etc.
# The result: token "it" now contains a blend of all relevant context
# Output shape: (seq_len, d_v) — each token now carries contextual information
```

Step by step:
1. Compute similarity scores: `QK^T` (dot product of query with all keys)
2. Scale: divide by `√d_k` to prevent very large values
3. Softmax: convert scores to probabilities (sum to 1)
4. Weighted sum: multiply probabilities by V to get output

### Why Scale by √d_k?

**What it is:** A normalization trick that prevents the softmax from becoming too "sharp," which would kill gradients.

For large `d_k`, dot products grow large → softmax becomes very "peaky" (near 0 or 1) → vanishing gradients. Scaling prevents this.

**Mathematical detail:** d_k-dimensional vectors have dot products with expected variance = d_k. Dividing by √d_k normalizes variance to 1. This keeps the softmax input in a range where it produces useful probability distributions rather than near-one-hot vectors.

---

## 4. Multi-Head Attention

**What it is:** Running several independent attention mechanisms in parallel, each learning to capture a different type of relationship between tokens.

Instead of one attention, run **h attention heads in parallel**, each with different learned Q, K, V projections.

**Analogy:** A panel of expert judges watching the same courtroom. One judge takes notes on legal precedents, one on body language, one on logical consistency. No single judge catches everything. Their combined judgment is more reliable than any one alone. Multi-head attention is the same idea — many specialized "judges" working simultaneously.

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O

head_i = Attention(Q*W_Q_i, K*W_K_i, V*W_V_i)
```

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model        # total model dimension, e.g. 768
        self.num_heads = num_heads    # number of parallel heads, e.g. 12
        self.d_k = d_model // num_heads  # dimension per head, e.g. 64
        # Each head gets a 64-dimensional "slice" of the 768-dimensional space

        # Each head has its own projection matrices (learned independently)
        # W_Q, W_K, W_V: project d_model → d_k for each head
        self.W_Q = Linear(d_model, d_model)  # projects to all heads' Q simultaneously
        # Shape: (768, 768) — will be reshaped to give 12 separate 64-dim projections
        self.W_K = Linear(d_model, d_model)  # projects to all heads' K simultaneously
        self.W_V = Linear(d_model, d_model)  # projects to all heads' V simultaneously
        # W_O: final projection to combine all heads' outputs back to d_model
        self.W_O = Linear(d_model, d_model)  # concatenated heads → single output

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        # x shape: (batch_size, sequence_length, model_dim) e.g. (32, 100, 768)

        # Create Q, K, V for ALL heads at once (more efficient than looping per head)
        Q = self.W_Q(x)  # shape: (batch, seq_len, d_model) = (32, 100, 768)
        K = self.W_K(x)  # shape: (batch, seq_len, d_model) = (32, 100, 768)
        V = self.W_V(x)  # shape: (batch, seq_len, d_model) = (32, 100, 768)

        # Reshape to separate out the individual heads
        # Each head gets its own 64-dimensional slice of the 768-dimensional space
        Q = Q.reshape(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # After reshape: (32, 100, 12, 64) — 12 heads, each 64-dim
        # After transpose: (32, 12, 100, 64) — heads in dim 1 for parallel compute
        K = K.reshape(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.reshape(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Shape now: (batch, num_heads, seq_len, d_k) — each head operates independently

        # Run scaled dot-product attention for all heads in parallel
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Q @ K.T shape: (batch, num_heads, seq_len, seq_len) — attention matrix per head
        # Head 1 computes its own attention, head 2 its own, completely independent
        weights = F.softmax(scores, dim=-1)  # normalize across key dimension
        head_outputs = weights @ V           # shape: (batch, num_heads, seq_len, d_k)
        # Each head produces a 64-dim output for every token

        # Concatenate all heads back together
        combined = head_outputs.transpose(1, 2).reshape(batch, seq_len, d_model)
        # transpose back: (batch, seq_len, num_heads, d_k)
        # reshape to join: (batch, seq_len, d_model) — 12 × 64 = 768
        # Shape: (batch, seq_len, d_model) — all heads' outputs joined side by side

        # Final linear projection to mix information across heads
        output = self.W_O(combined)
        # W_O lets the model learn how to combine insights from different heads
        # Without W_O, heads couldn't interact — each head's output would be independent
        return output  # shape: (batch, seq_len, d_model)
```

Why multiple heads?
- Each head can attend to **different aspects** of the input simultaneously
- Head 1 might learn syntactic relations (subject-verb), Head 2 semantic relations (coreference)
- Richer, more expressive representations than any single head could provide

**WHY project with W_O at the end?** The concatenated heads each "speak a different language." W_O is a learned mixer that combines these different perspectives into a single coherent output representation. Without it, heads couldn't interact with each other's findings.

**Typical values:**
| Model | d_model | Heads | d_k |
|-------|---------|-------|-----|
| GPT-2 small | 768 | 12 | 64 |
| BERT base | 768 | 12 | 64 |
| GPT-3 | 12288 | 96 | 128 |

---

## 5. Types of Attention

**What it is:** The three variations of attention used in different Transformer architectures, each with a different rule about which tokens can look at which.

### Self-Attention

**What it is:** Attention where a sequence attends to itself — every token can look at every other token in the same sequence.

Q, K, V all come from the **same sequence**. Each token attends to all other tokens in the same sequence.
→ Used in encoder (BERT) and decoder (GPT)

**WHY call it "self" attention?** Because the query and the keys come from the same source. The sequence is "looking at itself" to understand its own context. Contrast with cross-attention where you look at a different sequence.

### Cross-Attention

**What it is:** Attention where one sequence queries another — the decoder looks at the encoder's output to decide what source information to use while generating.

Q comes from one sequence, K and V from **another sequence**.
→ Used in encoder-decoder models (T5) between encoder output and decoder

```python
# Cross-attention in a translation decoder:
# Q: from decoder (what the decoder is currently generating)
# K, V: from encoder (the full encoded source sentence)

decoder_Q = decoder_layer(target_so_far)    # "what does the decoder need?"
# decoder is generating the French translation — what context does it need?

encoder_K = encoder_output                   # "what does the source contain?"
# the full English sentence has been encoded — these are all its "labels"

encoder_V = encoder_output                   # "what info does the source provide?"
# when the decoder attends to an English word, it gets that word's full representation

# The decoder asks: "given what I've generated so far, which source words should I look at?"
cross_attn_output = attention(decoder_Q, encoder_K, encoder_V)
# Each decoder position gets a weighted blend of encoder positions
# Generating "chat" in French → attends heavily to "cat" in English source
# The model figures out this alignment automatically during training
```

**WHY cross-attention for translation?** When generating each target word, the model needs to refer back to the source. Cross-attention provides this reference mechanism, letting the decoder "read" the source at each generation step.

### Causal (Masked) Self-Attention

**What it is:** Self-attention with a constraint: each token can only look backwards (at itself and previous tokens), never forwards.

Each token can only attend to **previous tokens** (not future ones).
→ Used in GPT-style decoders (autoregressive generation)
Implemented by masking future positions to -∞ before softmax.

```python
# Create the causal mask for a sequence of length 4
seq_len = 4
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
# torch.triu: upper triangular matrix (True above the diagonal)
# diagonal=1 means we keep 0s on and below the diagonal, 1s above
# Visualized:
# [[False, True,  True,  True ],   ← token 0: True positions are MASKED (invisible)
#  [False, False, True,  True ],   ← token 1: can see 0 and 1, not 2 or 3
#  [False, False, False, True ],   ← token 2: can see 0, 1, 2, not 3
#  [False, False, False, False]]   ← token 3: can see all (no future to mask)

# Apply mask to scores before softmax
scores = scores.masked_fill(mask, float('-inf'))
# Positions marked True get -inf → after softmax, e^(-inf) = exactly 0
# This makes those attention weights exactly zero — those positions are invisible
# It's as if those future tokens don't exist yet
weights = F.softmax(scores, dim=-1)
# Future positions have weight 0 → can't influence the current token
# Token 2 generating output → only tokens 0, 1, 2 influence it, never token 3
```

```
Token 1: can attend to [Token 1]
Token 2: can attend to [Token 1, Token 2]
Token 3: can attend to [Token 1, Token 2, Token 3]
```

**WHY is causal masking used during training?** During training, all tokens exist simultaneously. But we're training the model to predict each next token. If token 3 could see token 4 during training, it would just copy it — no learning. The mask forces each position to predict its successor from only what came before.

**WHY -inf specifically?** After softmax, e^(-∞) = 0 exactly. The masked positions contribute zero to the weighted sum — they're truly invisible. Using a large negative number like -10000 would give a tiny non-zero weight, which could cause subtle training instabilities.

---

## 6. Positional Encoding

**What it is:** A mechanism to inject sequence-order information into token representations, because self-attention itself has no concept of position.

Transformers have no inherent sense of order (unlike RNNs which process sequentially). Positional encoding adds position information so the model knows token order.

**Analogy:** Imagine giving every word a colored sticker based on its position — word 1 gets red, word 2 gets blue, word 3 gets green. Attention can now tell that "cat" (blue sticker) comes after "The" (red sticker). Without stickers, "cat The" looks identical to "The cat." Positional encoding is that sticker system — but in 768 dimensions.

### Sinusoidal Positional Encoding (Original)

**What it is:** A fixed mathematical formula using sine and cosine waves to generate a unique "fingerprint" for each position.

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

```python
import numpy as np

def sinusoidal_positional_encoding(max_seq_len, d_model):
    # Creates a matrix of shape (max_seq_len, d_model) with position encodings
    # This is NOT learned — it's a fixed mathematical formula
    PE = np.zeros((max_seq_len, d_model))  # start with zeros, fill below

    positions = np.arange(max_seq_len)[:, np.newaxis]  # shape: (max_seq_len, 1)
    # positions = [[0], [1], [2], ..., [max_seq_len-1]]
    # Each row is the position index — we'll compute sin/cos for each

    i_values = np.arange(0, d_model, 2)  # every other dimension index: 0, 2, 4, ...
    # The formula 10000^(2i/d_model) creates different frequencies for each dimension
    # Low dimensions (i=0): fast oscillation (changes significantly between nearby positions)
    # High dimensions (i=d_model/2): slow oscillation (changes gradually between positions)
    # This multi-scale encoding lets the model detect both local and global position patterns
    # Think: like a clock with fast-moving seconds hand and slow-moving hour hand
    div_term = 10000 ** (i_values / d_model)
    # div_term shape: (d_model/2,) — one divisor per dimension pair

    PE[:, 0::2] = np.sin(positions / div_term)  # even dimensions → sine wave
    PE[:, 1::2] = np.cos(positions / div_term)  # odd dimensions → cosine wave
    # WHY both sin and cos? For any offset k, PE(pos+k) = linear_function(PE(pos))
    # This means the model can learn "look 3 positions back" as a learned linear op
    # With only sin, you couldn't express this transformation cleanly

    return PE  # shape: (max_seq_len, d_model), added to token embeddings
```
Added directly to token embeddings.

**WHY sinusoidal?**
- Unique pattern for every position — no two positions have the same encoding
- Works for any sequence length — the formula extends beyond training length
- Distance consistency — nearby positions have similar encodings
- The model can infer relative positions from the sin/cos patterns

### Learned Positional Embeddings

**What it is:** Simply treating positions as tokens and learning an embedding for each one, just like word embeddings.

Learn a separate embedding for each position (used in BERT, GPT-2).

```python
# Learned positional embeddings — simple but effective
max_positions = 512        # maximum sequence length (fixed at training time)
d_model = 768              # embedding dimension
# Trainable parameter: one vector per position, all learned from scratch
pos_embedding_table = nn.Embedding(max_positions, d_model)  
# Shape: (512, 768) — 512 learned position vectors
# These 512 vectors are just parameters like any other — trained via backprop

# At runtime: look up the embedding for each position
positions = torch.arange(seq_len)        # [0, 1, 2, ..., seq_len-1]
pos_embeddings = pos_embedding_table(positions)  # shape: (seq_len, d_model)
# Look up the learned vector for each position index
# Add to token embeddings: token gets both its meaning and its position
x = token_embeddings + pos_embeddings   # shape: (seq_len, d_model)
# Each token now carries: "I am the word X (from embedding) at position Y (from PE)"
```

**WHY learned embeddings can be better:** The model discovers the best position representation for its task, rather than using a fixed formula. But there's a downside: position 513 was never trained, so it has no embedding — can't generalize beyond training length.

### RoPE (Rotary Position Embedding)

**What it is:** A modern positional encoding that directly modifies Query and Key vectors by rotating them, so relative position is naturally encoded in attention scores.

- Used in modern LLMs (LLaMA, GPT-NeoX, Mistral)
- Encodes relative position directly in attention computation
- Better at generalizing to longer sequences

```python
def apply_rope(q, k, positions, d_model):
    # q, k: shape (batch, heads, seq_len, d_head)
    # positions: [0, 1, 2, ..., seq_len-1]
    # We'll rotate q and k vectors — the rotation angle encodes position

    # Compute rotation angles: different frequencies for each dimension pair
    # Low dimensions: fast rotation (changes quickly between positions)
    # High dimensions: slow rotation (changes slowly between positions)
    # Same multi-frequency idea as sinusoidal encoding, but applied as rotation
    theta = 1.0 / (10000 ** (torch.arange(0, d_model, 2) / d_model))
    # theta shape: (d_model/2,) — one frequency per dimension pair

    angles = positions[:, None] * theta[None, :]  # shape: (seq_len, d_model/2)
    # Each position gets rotated by a different angle per dimension
    # Position 0 → angles near 0 (almost no rotation)
    # Position 100 → larger angles (significant rotation)

    # Apply rotation: pair up dimensions and rotate each pair
    cos_angles = torch.cos(angles)  # cosine component of rotation matrix
    sin_angles = torch.sin(angles)  # sine component of rotation matrix

    # Rotate q: split into pairs (q0, q1), (q2, q3), ...
    # Rotation formula: [x, y] → [x*cos - y*sin, x*sin + y*cos]
    q_rotated = q * cos_angles - q_shifted * sin_angles  # rotation formula applied
    k_rotated = k * cos_angles - k_shifted * sin_angles  # same for keys

    return q_rotated, k_rotated
    # WHY rotation? When you compute q_rotated · k_rotated (the attention dot product),
    # the result depends only on the RELATIVE angle between positions (pos_q - pos_k)
    # — not on their absolute positions. This gives the model relative position for free.
    # Key property: rotating both Q and K by their positions → dot product encodes (pos_q - pos_k)
```

**WHY RoPE beats learned embeddings?** The dot product Q·K naturally encodes relative position (m-n), not absolute. The model learns "how far apart" without needing absolute positions. Also: rotation preserves vector magnitude, so it doesn't distort the content information.

### ALiBi (Attention with Linear Biases)

**What it is:** An alternative to positional embeddings that adds a distance penalty directly to attention scores — tokens far apart get penalized.

- Used in MPT, BLOOM
- Adds a linear bias to attention scores based on distance
- Extrapolates better to longer sequences

```python
def apply_alibi_bias(attention_scores, num_heads):
    # attention_scores: (batch, heads, seq_len, seq_len) — raw QK^T scores

    seq_len = attention_scores.shape[-1]

    # Create distance matrix: entry [i,j] = how far apart tokens i and j are
    distance_matrix = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
    # entry [i,j] = j - i (positive means j comes after i, negative means before)
    # Example: distance_matrix[3][1] = 1-3 = -2 (token 1 is 2 positions before token 3)

    # Negative because farther = more penalty (more negative = less attention)
    # m is a head-specific slope — different heads penalize distance differently
    # Head 1: steep slope → strong preference for nearby tokens (local attention)
    # Head 8: gentle slope → can still attend to distant tokens (global attention)
    # This allows different heads to have different "attention ranges"
    m_values = get_slopes(num_heads)  # pre-determined set of slopes, one per head
    # Slopes are geometric series: 2^(-8/h), 2^(-16/h), ... for h heads

    # Add linear bias: entry [i,j] = m * (j - i), clipped to 0 for j < i (causal)
    alibi_bias = m_values[:, None, None] * distance_matrix.clamp(max=0)
    # clamp(max=0): only penalize attending to past tokens (j < i)
    # Tokens 1 position apart: small penalty (e.g., -0.5)
    # Tokens 100 positions apart: large penalty (e.g., -50) → near-zero attention weight

    return attention_scores + alibi_bias  # bias added BEFORE softmax
    # WHY before softmax? So the bias affects the probability distribution directly
    # After softmax: far-away tokens get exponentially lower weights
```

**Advantage:** Excellent length extrapolation — train on 2K tokens, works on 100K.
**Disadvantage:** Not as strong as RoPE on absolute position tasks.

**WHY ALiBi works at long context?** The bias is linear in distance, so it extends naturally to any distance without needing special scaling tricks. RoPE needs explicit "NTK-aware scaling" to extend; ALiBi extends automatically.

---

## 7. Feed-Forward Network (FFN)

**What it is:** A two-layer neural network applied independently to each token after attention — it stores factual knowledge and adds non-linear transformation power.

After attention, each position passes through a **2-layer MLP** independently:

**Analogy:** Attention is the "gather information from others" step. FFN is the "process that information privately" step. Like a committee meeting (attention) followed by each member going home to think and write their report (FFN). The meeting decides what context matters; the private work retrieves knowledge about it.

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, ffn_dim):
        super().__init__()
        self.d_model = d_model   # input/output size, e.g. 768
        self.ffn_dim = ffn_dim   # hidden size = 4 × d_model = 3072 for GPT-2

        # Two learned weight matrices
        self.W_1 = nn.Linear(d_model, ffn_dim)   # expand: 768 → 3072
        # This first layer "asks many questions" about the token's representation
        self.W_2 = nn.Linear(ffn_dim, d_model)   # contract: 3072 → 768
        # This second layer "summarizes the answers" back into the model dimension

    def forward(self, x):
        # x: shape (batch, seq_len, d_model)
        # Applied identically and independently to every position — fully parallel
        # All 100 tokens in the sequence are processed at the same time (no dependencies)

        hidden = self.W_1(x)
        # Linear expansion: each 768-dim token representation → 3072-dim intermediate
        # This creates a larger "workspace" for complex pattern matching
        # Like expanding a formula before simplifying: more room = more operations
        # Shape: (batch, seq_len, 3072)

        hidden = F.gelu(hidden)
        # GELU activation: Gaussian Error Linear Unit
        # Like ReLU (zero for negatives) but smooth — no hard cutoff at zero
        # GELU approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x³)))
        # Allows gradients to flow even for slightly negative values
        # WHY not ReLU? ReLU's hard zero creates "dead neurons" — once negative, stays zero
        # GELU's smooth approximation avoids this, better gradient flow through training

        output = self.W_2(hidden)
        # Contract back to model dimension: 3072 → 768
        # Shape: (batch, seq_len, d_model)
        # Each position is processed independently — no cross-position mixing here
        # (Cross-position mixing already happened in attention — this is individual processing)

        return output
```

```
FFN(x) = GELU(x * W_1 + b_1) * W_2 + b_2
```

- Typically 4× wider than d_model
- Example: d_model=768 → FFN hidden dim = 3072
- Acts as "memory" — stores factual knowledge about the world

**WHY no interaction between positions in FFN?** After attention, each token's representation already contains mixed information from all relevant other tokens. The FFN's job is to process that mixed representation — there's nothing more to mix. Keeping positions independent here also makes computation fully parallelizable across the sequence.

**Interview answer:** "The FFN is a position-wise 2-layer MLP applied independently to each token after attention. It expands to 4× d_model (giving more processing capacity), applies GELU activation, then contracts back. Research by Geva et al. shows FFN layers function as key-value memories storing factual knowledge. Attention handles contextual mixing; FFN handles knowledge retrieval."

---

## 8. Residual Connections & Layer Norm

### Residual Connection (Skip Connection)

**What it is:** Adding a sublayer's input directly to its output, creating a "shortcut" that lets gradients flow backwards without passing through the sublayer.

```
output = LayerNorm(x + sublayer(x))
```

**Analogy:** A mountain hiking trail with a shortcut path. Even if the main trail (the sublayer) is steep and treacherous (gradients shrink), hikers (gradients) can take the direct shortcut home. The shortcut ensures they always make progress regardless of how difficult the main path becomes.

```python
class TransformerBlock(nn.Module):
    def forward(self, x):
        # WRONG (no residual — gradients vanish in deep networks):
        # x = self.attention(x)    # gradient must propagate through all of attention
        # x = self.ffn(x)          # gradient must propagate through all of FFN
        # After 96 layers: gradient ≈ near zero → early layers learn nothing

        # CORRECT (with residual):
        # ATTENTION sublayer with residual
        attn_output = self.attention(self.layer_norm_1(x))  # pre-norm style
        # attn_output: what attention learned to ADD to the current representation
        # The attention output might be small, but it adds something useful
        x = x + attn_output
        # x + attn_output: the shortcut! Input x flows through unchanged
        # Gradient flows through: d(x + attn_output)/dx = 1 + d(attn_output)/dx
        # The "1" is the shortcut — even if d(attn_output)/dx → 0, total gradient = 1
        # This prevents gradient vanishing at any depth — even layer 1 of a 96-layer model

        # FFN sublayer with residual
        ffn_output = self.ffn(self.layer_norm_2(x))  # pre-norm style
        # ffn_output: what the FFN learned to ADD after attention
        x = x + ffn_output
        # Same shortcut here — gradient highway from this layer to all previous layers
        # The model learns to use FFN output as small corrections, not full rewrites

        return x
```

- Adds the input directly to the output of each sub-layer
- Prevents vanishing gradients in deep networks (96 layers in GPT-3)
- Allows gradients to flow directly to early layers via the shortcut

**WHY is this so important for large models?** GPT-3 has 96 Transformer blocks. Without residuals, gradients would multiply through 192 sublayers, shrinking exponentially. With residuals, there's always a direct path — gradients remain meaningful even at layer 1.

### Pre-Norm vs Post-Norm

**What it is:** Whether normalization happens before or after a sublayer. This changes training stability significantly.

| Type | Formula | Used In |
|------|---------|---------|
| Post-Norm | LayerNorm(x + sublayer(x)) | Original Transformer, BERT |
| Pre-Norm | x + sublayer(LayerNorm(x)) | GPT-3, LLaMA (more stable) |

```python
# Post-Norm: normalize AFTER the residual sum
def post_norm_block(x, sublayer, layer_norm):
    result = layer_norm(x + sublayer(x))
    # Problem: x (residual) is unnormalized when added to sublayer output
    # The sublayer might see large inputs if x has grown over many layers
    # At large scale, unnormalized residuals can accumulate and cause instability
    # The normalization only happens AFTER the damage — too late
    return result

# Pre-Norm: normalize BEFORE the sublayer
def pre_norm_block(x, sublayer, layer_norm):
    result = x + sublayer(layer_norm(x))
    # The sublayer always receives a normalized input — stable throughout training
    # layer_norm(x) is always well-behaved (mean≈0, std≈1) regardless of depth
    # The residual x bypasses normalization — the direct highway stays clean
    # WHY this is more stable: sublayer never sees "wild" values, even in deep networks
    # Like always handing a chef ingredients that are at the right temperature
    return result
```

**WHY Pre-Norm dominates modern LLMs?** With Pre-Norm, the sublayer always receives a well-normalized input with controlled scale. With Post-Norm, if residuals accumulate over many layers, the sublayer might receive very large or very small inputs, causing unstable training. At GPT-3 scale (96 layers), this matters a lot.

---

## 9. Encoder vs Decoder vs Encoder-Decoder

**What it is:** Three architectural configurations of Transformers, each optimized for a different type of task.

| Architecture | Attention Type | Examples | Best For |
|-------------|----------------|---------|---------|
| **Encoder-only** | Bidirectional self-attention | BERT, RoBERTa | Classification, NER, embeddings |
| **Decoder-only** | Causal (masked) self-attention | GPT-2, GPT-3, LLaMA | Text generation, completion |
| **Encoder-Decoder** | Bidirectional encoder + causal decoder + cross-attention | T5, BART | Translation, summarization, QA |

```python
# Encoder (BERT style): every token sees everything
# Attention mask: all ones (no masking)
encoder_mask = torch.ones(seq_len, seq_len)  # all attention allowed
# Token 50 can look at token 1 AND token 100 simultaneously
# WHY bidirectional? Understanding a word's meaning requires full sentence context
# "The bank was robbed" — you need to see all words to know what kind of "bank"

# Decoder (GPT style): causal mask — only look backwards
decoder_mask = torch.tril(torch.ones(seq_len, seq_len))  # lower triangle only
# Token 50 can only look at tokens 1-50, not 51-100
# WHY causal? Generation is left-to-right — the model hasn't written tokens 51-100 yet
# During training: masking forces real prediction instead of copying future tokens

# Encoder-Decoder (T5 style): encoder is bidirectional, decoder is causal + cross-attention
# Encoder: processes source text with full bidirectional attention
encoder_output = transformer_encoder(source_tokens, mask=None)  # sees all source tokens
# encoder_output contains rich representations of every source token
# Each source token knows about all other source tokens

# Decoder: generates output tokens causally while cross-attending to encoder
for step in range(target_length):
    decoder_output = transformer_decoder(
        target_tokens_so_far,  # Q: what has the decoder generated so far
        encoder_memory=encoder_output,  # K, V: full source representation
        causal_mask=True       # can't see future target tokens (must predict them)
    )
    next_token = project_to_vocab(decoder_output[-1])  # predict next word
    # The decoder "reads" the source via cross-attention at every generation step
```

**WHY decoder-only (GPT) dominates today?** GPT-3 proved that one decoder, trained on enough data, can do translation, summarization, classification, and QA — all via prompting. No need for separate architectures per task. Scaling a single decoder architecture proved more efficient than maintaining multiple architectures.

---

## 10. KV Cache

**What it is:** A memory optimization for text generation that saves computed Key and Value matrices from previous steps so they don't need to be recomputed.

During inference (text generation), the model generates one token at a time.
Re-computing K and V for all previous tokens every step is wasteful.

**KV Cache**: Store the K and V matrices from past tokens. Only compute for the new token.

**Analogy:** Writing a book chapter by chapter. Without a cache, before writing chapter 10, you re-read chapters 1-9 from scratch every time. With a cache, you keep chapters 1-9 open on your desk — you only need to read the newest chapter you just wrote. Your effort stays constant regardless of how long the book gets.

```python
class GenerationWithKVCache:
    def __init__(self):
        self.kv_cache = {}  # dictionary to store K, V for each layer
        # Separate cache per layer because each layer has different K, V projections

    def generate_next_token(self, new_token, layer_id):
        # Compute K, V only for the NEW token — not for all previous tokens
        new_k = compute_key(new_token)    # shape: (1, d_k) — just one token's key
        new_v = compute_value(new_token)  # shape: (1, d_k) — just one token's value

        if layer_id in self.kv_cache:
            # Retrieve cached K, V from all PREVIOUS tokens (already computed)
            past_k = self.kv_cache[layer_id]['k']  # shape: (past_len, d_k)
            past_v = self.kv_cache[layer_id]['v']  # shape: (past_len, d_k)
            # These were computed in previous generation steps and saved

            # Append new token's K, V to the cache
            all_k = torch.cat([past_k, new_k], dim=0)  # shape: (past_len + 1, d_k)
            all_v = torch.cat([past_v, new_v], dim=0)  # shape: (past_len + 1, d_k)
            # Now we have K and V for ALL tokens (old cached + new computed)
        else:
            all_k, all_v = new_k, new_v  # first token — nothing cached yet

        # Store the updated cache for the next generation step
        self.kv_cache[layer_id] = {'k': all_k, 'v': all_v}

        # Attention for new token: query is ONLY the new token
        q_new = compute_query(new_token)   # shape: (1, d_k)
        # But K, V include ALL past tokens (from cache + new)
        output = attention(q_new, all_k, all_v)  # attends to entire history efficiently
        return output
        # WHY only query the new token? It's the only one we don't know the output for yet
        # Past tokens' outputs don't change — they were computed and used in earlier steps
```

```
Without KV cache: O(n²) computation per step  ← must recompute K,V for all n past tokens
With KV cache: O(n) computation per step       ← only compute K,V for 1 new token
```

Critical for efficient inference. Used in all production LLM serving systems.

**WHY does KV cache work?** In autoregressive generation, past tokens' representations never change — their context is fixed. The K and V matrices for "The cat sat" don't change when you're predicting "on." Caching them avoids redundant computation.

**Memory trade-off:** KV cache stores O(n × layers × heads × d_k) values. For LLaMA 3 8B at 4096 context: ~536MB per sequence. This limits batch size in production.

---

## 11. Context Window & Attention Complexity

**What it is:** The maximum number of tokens a model can process at once, and why making it larger is expensive.

**Context window**: Maximum number of tokens the model can process at once.

Attention has **O(n²) memory and compute** with sequence length n.
- This is why long contexts are expensive.
- Active research area: Flash Attention, Sparse Attention, Linear Attention.

```python
# Why n² is a problem — concrete numbers:
sequence_lengths = [512, 2048, 8192, 128_000]
for n in sequence_lengths:
    attention_matrix_size = n * n  # number of entries in the attention matrix
    # This is how many attention scores we need to compute (and store)
    print(f"n={n:,}: {attention_matrix_size:,} attention pairs")

# Output:
# n=512:      262,144 attention pairs      ← manageable (BERT)
# n=2,048:  4,194,304 attention pairs      ← expensive but doable (GPT-3)
# n=8,192:  67,108,864 attention pairs     ← expensive (LLaMA 3)
# n=128,000: 16,384,000,000 attention pairs ← impossible without FlashAttention
# Each entry is a float (4 bytes) — 128K context = 65 GB just for the attention matrix
```

### Flash Attention

**What it is:** A hardware-aware reimplementation of attention that achieves the same math result but reads/writes GPU memory far less often.

- Reorders attention computation to minimize memory I/O (the real bottleneck)
- Same mathematical result, much faster and memory-efficient
- Flash Attention 2 / 3 are standard in modern training

```python
# The core insight of FlashAttention — tiling:

# Standard attention (SLOW — many reads/writes to slow GPU memory called HBM/DRAM):
S = Q @ K.T          # compute full (n×n) matrix, write entire thing to GPU DRAM (slow)
# DRAM is like a hard drive — large but slow. n×n can be gigabytes.
P = softmax(S)       # read from DRAM, compute softmax, write back to DRAM (slow)
O = P @ V            # read P and V from DRAM, compute, write O (slow)
# Every step forces a round-trip to slow memory
# Total DRAM operations: O(n²) — very slow for large n

# FlashAttention (FAST — stays in fast on-chip memory called SRAM):
# SRAM is like RAM — small (~20MB) but 10-100x faster than DRAM
# Split Q, K, V into blocks that fit in GPU SRAM (fast cache, ~20MB)
for q_block in blocks(Q):
    for kv_block in blocks(K, V):
        # Compute partial attention within SRAM — no DRAM access needed
        partial_scores = q_block @ kv_block.T  # tiny block computation in SRAM
        # Accumulate result using online softmax (no need to see full row at once)
        # Key trick: keep running max and sum → equivalent to full softmax
        # Update running max and sum incrementally as you see each block
# Write final result O to DRAM just once at the end
# Total DRAM operations: O(n) — much faster! Same answer, far fewer slow memory accesses
```

**WHY does FlashAttention achieve the same result?** It uses the "online softmax" algorithm — you can compute softmax of [a, b, c, d] incrementally as you see each value, keeping only a running maximum and running sum. You get the same answer without ever materializing the full array.

---

## 12. Transformer Scaling

**What it is:** The empirical finding that transformer performance improves predictably as you increase size, data, and compute — enabling the entire LLM industry.

Key insight from "Scaling Laws" (Kaplan et al., 2020):
- Model performance improves predictably with:
  - More **parameters** (bigger model)
  - More **data** (more training tokens)
  - More **compute** (more GPU-hours)

**WHY this is important:** It means you can predict in advance how good a model will be given a compute budget. This made the LLM industry investable — you can say "GPT-3 sized model will be X% better than GPT-2" before spending money.

```python
# Scaling law (simplified Chinchilla formula):
# Optimal training: num_tokens ≈ 20 × num_parameters
# GPT-3: 175B params → should train on ~3.5T tokens
# LLaMA 3: 8B params → trained on 15T tokens (over-trained intentionally for better inference efficiency)

# Why over-train smaller models?
# Inference cost = num_active_params × tokens_generated
# Smaller model = cheaper inference per query
# If you can get a 7B model to match an undertrained 70B model through more data,
# the 7B model will be 10× cheaper to serve — huge production advantage
# Training is a one-time cost; inference is paid per query (billions of queries)

# The business logic:
# 10× more training compute on a 7B model → 10× training cost (paid once)
# 10× cheaper inference → savings of 10× on every single query forever
# At scale (1B queries/month), this is easily justified
```

Larger models → better performance on all tasks (emergent abilities appear at scale).

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
> During autoregressive generation, Key and Value matrices from previous tokens don't change. Caching them avoids recomputation and reduces inference time from O(n²) to O(n) per generation step. Critical for production serving — without KV cache, every generation step would be O(n²).

**Q: What is Flash Attention?**
> A hardware-aware attention algorithm that rewrites the attention computation to minimize GPU memory reads/writes (I/O bound operations), achieving the same mathematical result 2-4× faster with significantly less memory.

---

## Quick Reference Cheat Sheet

```
Core Formula:    Attention(Q,K,V) = softmax(QK^T / √d_k) * V
                 Q = what I'm looking for, K = what I have, V = what I give

Heads:           Multiple parallel attention heads (different projections)
                 d_k = d_model / num_heads (each head gets a slice)

Self-Attention:  Q,K,V from same sequence — every token attends to all others
Causal Mask:     Prevents attending to future tokens (GPT) — -inf before softmax
Cross-Attention: Q from decoder, K,V from encoder (T5) — decoder reads source

Residuals:       x + sublayer(x) — gradient highway, prevents vanishing gradients
                 Gradient = 1 always exists, even if sublayer gradient → 0

FFN:             2-layer MLP after attention, 4x wider — stores factual knowledge
                 Applied independently per position — parallel across sequence

Positional Enc:  Sinusoidal (original), Learned (BERT/GPT-2), RoPE (LLaMA/Mistral)
                 RoPE: rotate Q,K vectors → dot product encodes relative position

KV Cache:        Cache K,V for efficient inference — O(n²) → O(n) per step
Flash Attention: Tile attention in SRAM, avoid n² DRAM reads — same math, faster
```
