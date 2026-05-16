# Attention Is All You Need — The Paper That Changed Everything

> Google Brain, 2017. Vaswani et al.
> This single paper killed RNNs, LSTMs, and CNNs for NLP.
> Every LLM today — GPT-4, LLaMA, Gemini, Claude — is built on THIS architecture.

---

## WHY THIS PAPER EXISTS — The Problem Before 2017

Before Transformers, NLP used **RNNs (Recurrent Neural Networks)** and **LSTMs**.

### How RNNs worked:
```
"The cat sat on the mat"

Step 1: process "The"   → hidden state h1
Step 2: process "cat"   → hidden state h2 (uses h1)
Step 3: process "sat"   → hidden state h3 (uses h2)
...
Step 6: process "mat"   → final hidden state h6
```

**3 massive problems with this:**

### Problem 1: Sequential = Slow
Words must be processed ONE BY ONE. You can't process "cat" until "The" is done.
On a GPU with 10,000 cores — you're only using 1 at a time. Criminal waste.

### Problem 2: Long-Range Forgetting
To translate "The animal didn't cross the street because **it** was too tired" —
the model needs to know "it" refers to "animal" (6 words back).
By step 6, the hidden state has forgotten what happened at step 1.
This is called the **vanishing gradient problem**.

### Problem 3: Sequential Training = Weeks
Training on 100 billion words, one word at a time = took weeks even on 100 GPUs.

### What Vaswani et al. said:
> "What if we got rid of recurrence entirely and just used attention?"

Result: Transformers. All words processed IN PARALLEL. Any word can directly
attend to any other word. Training that took weeks now takes days.

---

## THE CORE IDEA — What Is Attention?

Imagine you're reading: **"The bank can guarantee deposits will eventually cover
future tuition costs because it was constructed specifically to do so."**

What does **"it"** refer to? The bank? The deposits? Future costs?

Attention is the mechanism that lets the model answer this.

### Human Intuition:
When you read "it", your brain automatically goes back and checks:
- "bank" — possible antecedent
- "deposits" — possible
- "tuition costs" — less likely

You assign mental "weights" to each word. "Bank" gets high weight, "tuition" gets low.
**That's exactly what attention does — but with math.**

### The Three Vectors — Q, K, V

Every word in the sentence gets transformed into 3 vectors:

| Vector | Full Name | Intuition | Question it answers |
|--------|-----------|-----------|---------------------|
| **Q** | Query | What am I looking for? | "I'm the word 'it' — what should I look at?" |
| **K** | Key | What do I represent? | "I'm the word 'bank' — this is what I contain" |
| **V** | Value | What do I actually give? | "If you attend to me, here's my actual information" |

### The Attention Formula:
```
Attention(Q, K, V) = softmax( Q × Kᵀ / √d_k ) × V
```

**Step by step in plain English:**

**Step 1:** `Q × Kᵀ` — Every word's Query dot products with every word's Key.
This gives a raw similarity score. High score = "these two words are related."

**Step 2:** `÷ √d_k` — Divide by square root of the key dimension.
Why? If d_k = 512, dot products become huge numbers → softmax becomes too "peaky"
(one value close to 1, all others close to 0) → gradients vanish.
Dividing by √512 ≈ 22.6 keeps values in a healthy range.

**Step 3:** `softmax(...)` — Turn raw scores into probabilities that sum to 1.
Now each word has a probability distribution over all other words.
"it" → {bank: 0.72, deposits: 0.18, costs: 0.06, other: 0.04}

**Step 4:** `× V` — Weighted sum of Value vectors.
The output for "it" = 0.72 × V_bank + 0.18 × V_deposits + 0.06 × V_costs + ...
"it" now carries 72% of bank's information in its representation.

**This is the magic.** Every word's final representation is a weighted blend
of ALL other words in the sentence, weighted by relevance.

---

## MULTI-HEAD ATTENTION — Why One Attention Isn't Enough

Single attention can only learn one type of relationship at a time.

But language has MULTIPLE simultaneous relationships:
- Syntactic: "bank" is the subject of "can guarantee"
- Semantic: "it" refers back to "bank"
- Positional: "bank" is near "deposits"
- Coreference: "it" and "animal" are the same entity

### Solution: Run h attention mechanisms in parallel

```
Input X (seq_len × d_model)
    ↓ split into h heads
Head 1: learns syntactic relationships
Head 2: learns coreference
Head 3: learns positional relationships
Head 4: learns semantic similarity
...
Head h: learns some other relationship
    ↓ concatenate all heads
Output: (seq_len × d_model)
```

### Each head has its own projection matrices:
- W_Q^i, W_K^i, W_V^i — learned independently per head
- d_k = d_model / h (each head gets a slice of the full dimension)

### Real numbers from GPT-2:
- d_model = 768
- h = 12 heads
- d_k = 768 / 12 = 64 per head

BERT base uses the same. GPT-3 uses 96 heads with d_model = 12288.

### What did researchers find when they visualized attention heads?
- Some heads learned to attend to the previous word (positional)
- Some heads learned subject-verb agreement
- Some heads learned coreference resolution
- Some heads seemed to learn "rare token attention" (unknown purpose)

Each head specializes automatically during training. No human designs this.

---

## POSITIONAL ENCODING — Teaching Order to a Orderless Model

### The Problem:
Self-attention is **permutation invariant**.
"cat sat mat" and "mat sat cat" produce the exact same attention scores
if the words are the same — because attention only cares about content, not position.

But order matters! "Dog bites man" ≠ "Man bites dog."

### The Solution: Add position information to embeddings

**Sinusoidal Encoding (Original Paper):**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where `pos` = position in sequence, `i` = dimension index.

**Why sinusoidal?**
- Unique pattern for every position
- Works for any sequence length (even longer than training)
- Distance between positions is consistent
- The model can learn to attend to "relative positions" from these patterns

**Learned Positional Embeddings (BERT, GPT-2):**
Instead of a formula, just learn a separate embedding for each position.
More flexible, works better in practice, but can't generalize beyond training length.

**Modern: RoPE (Rotary Position Embedding) — LLaMA, Mistral, GPT-NeoX:**
Encodes position directly into the Q and K vectors using rotation matrices.
```
Key insight: instead of adding position to token embeddings,
rotate Q and K vectors by an angle proportional to their position.
The dot product Q·K then naturally encodes relative position.
```
Why better? Generalizes to longer sequences. LLaMA-3 trained on 8K context
but works on 128K with RoPE scaling (NTK-aware scaling).

---

## THE FEED-FORWARD NETWORK — The "Memory" of the Transformer

After attention, each position independently passes through a 2-layer MLP:

```
FFN(x) = GELU( x × W₁ + b₁ ) × W₂ + b₂
```

### Key facts:
- W₁ expands: d_model → 4 × d_model (e.g., 768 → 3072)
- W₂ contracts: 4 × d_model → d_model (e.g., 3072 → 768)
- Applied **identically and independently** to each position
- No interaction between positions here (that already happened in attention)

### What does FFN actually do?
Research (Geva et al., 2021) showed FFN layers act as **key-value memories**.
The neurons store factual knowledge:
- "Paris is the capital of France" — stored in FFN neurons
- "Python is a programming language" — stored in FFN neurons

When you ask an LLM a factual question, it's the FFN layers "remembering."
Attention figures out context. FFN provides facts. That's the division of labor.

### Modern variant: SwiGLU (used in LLaMA, PaLM):
```
SwiGLU(x) = (x × W₁) ⊙ SiLU(x × W₂)
```
Two parallel projections with element-wise multiplication.
Empirically better than standard FFN. PaLM/LLaMA use 8/3 × d_model instead of 4×.

---

## RESIDUAL CONNECTIONS — Why Deep Networks Can Be Trained

### The Problem:
A 96-layer Transformer (GPT-3) has 96 layers of attention + 96 FFNs.
During backpropagation, gradients must pass through all 192 operations.
By layer 10, gradient is near zero. Layers 1-96 learn nothing. This is
**vanishing gradients** — killed all deep networks before ResNet (2015).

### The Solution: Add input directly to output of every sublayer
```
output = LayerNorm( x + Attention(x) )
output = LayerNorm( x + FFN(x) )
```

The `+ x` creates a **direct highway for gradients**.
Gradient of loss can flow directly from layer 96 to layer 1 through the residuals.
Early layers now receive meaningful gradients and learn.

### Why this works mathematically:
```
∂L/∂x = ∂L/∂output × ∂output/∂x
       = ∂L/∂output × (1 + ∂Sublayer/∂x)
```
The `1` ensures gradient never completely vanishes, even if ∂Sublayer/∂x → 0.

---

## LAYER NORMALIZATION — Stabilizing Training

### Why normalize?
After each sublayer, activations can have varying scales.
Large activations → large gradients → unstable training.
Normalization keeps values in a healthy range throughout training.

### Layer Norm formula:
```
LayerNorm(x) = γ × (x - μ) / √(σ² + ε) + β
```
Where μ = mean, σ² = variance (computed over the feature dimension).
γ and β are learned parameters (scale and shift).

### Pre-Norm vs Post-Norm (Critical for stability):

**Post-Norm (Original Paper):**
```
x = LayerNorm(x + Sublayer(x))
```
Can be unstable at large scale. Used in original BERT.

**Pre-Norm (Modern LLMs — GPT-3, LLaMA, Mistral):**
```
x = x + Sublayer(LayerNorm(x))
```
Normalize BEFORE the sublayer. Much more stable for very deep/large models.
All modern LLMs use Pre-Norm.

### RMSNorm (LLaMA, T5):
```
RMSNorm(x) = x / RMS(x) × γ,   RMS(x) = √(mean(x²))
```
Simpler than LayerNorm — no mean subtraction.
Faster, similar performance. LLaMA uses this instead of LayerNorm.

---

## ENCODER vs DECODER vs ENCODER-DECODER

### Encoder Only (BERT family):
```
Input tokens → [Bidirectional Self-Attention] × N → Output representations
```
- Every token attends to EVERY other token (both left and right)
- Not for generation — for UNDERSTANDING
- Used for: classification, NER, embeddings, semantic search
- Examples: BERT, RoBERTa, DeBERTa, ALBERT

### Decoder Only (GPT family):
```
Input tokens → [Causal Self-Attention] × N → Next token prediction
```
- Each token attends ONLY to previous tokens (causal mask)
- Perfect for generation — predicts one token at a time
- Used for: text generation, code, chat, completion
- Examples: GPT-2, GPT-3, GPT-4, LLaMA, Mistral, Falcon

### Encoder-Decoder (T5, BART family):
```
Source → [Encoder: Bidirectional Attention] × N → Context
Context + Target → [Decoder: Causal + Cross-Attention] × N → Output
```
- Encoder reads source fully (bidirectional)
- Decoder generates output while attending to encoder (cross-attention)
- Used for: translation, summarization, question answering
- Examples: T5, BART, mT5, FLAN-T5

### Which is best?
```
Understanding tasks        → Encoder (BERT)
Generation tasks           → Decoder (GPT/LLaMA)
Sequence-to-sequence tasks → Encoder-Decoder (T5)
General purpose/chat/code  → Decoder wins (GPT-4, LLaMA dominate)
```

---

## THE FULL TRANSFORMER — How It All Connects

```
INPUT: "The cat sat on the mat"
    ↓
[Tokenizer]: "The"=464, "cat"=3797, "sat"=3332, ...
    ↓
[Token Embeddings]: each ID → 768-dim vector
    ↓
[+ Positional Encoding]: add position information
    ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRANSFORMER BLOCK × N (12 for GPT-2 small):
    ↓
  [Pre-LayerNorm]
    ↓
  [Multi-Head Self-Attention]
    - Split into 12 heads (64-dim each)
    - Each head: Q × Kᵀ / √64 → softmax → × V
    - Concatenate 12 heads → 768-dim
    ↓
  [+ Residual connection]
    ↓
  [Pre-LayerNorm]
    ↓
  [Feed-Forward Network]
    - Linear: 768 → 3072
    - GELU activation
    - Linear: 3072 → 768
    ↓
  [+ Residual connection]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓
[Final LayerNorm]
    ↓
[Linear: 768 → 50257 (vocab size)]
    ↓
[Softmax → probability over all words]
    ↓
OUTPUT: probability distribution → sample next token
```

---

## COMPLEXITY ANALYSIS — Why Long Context Is Expensive

| Operation | Time Complexity | Memory Complexity |
|-----------|----------------|-------------------|
| Self-Attention | O(n² × d) | O(n²) |
| FFN | O(n × d²) | O(n × d) |
| Total per layer | O(n² × d + n × d²) | O(n²) |

**n** = sequence length, **d** = model dimension

The `n²` in attention is the problem:
- n=512 (BERT): 512² = 262K attention pairs — manageable
- n=2048 (GPT-3): 2048² = 4M attention pairs — expensive
- n=128K (GPT-4): 128000² = 16 BILLION attention pairs — requires FlashAttention

This is why **FlashAttention** was a revolution — same math, 100× less memory I/O.

---

## KEY NUMBERS TO MEMORIZE

| Model | Layers | Heads | d_model | FFN dim | Params | Context |
|-------|--------|-------|---------|---------|--------|---------|
| GPT-2 small | 12 | 12 | 768 | 3072 | 117M | 1024 |
| GPT-2 large | 36 | 20 | 1280 | 5120 | 774M | 1024 |
| BERT base | 12 | 12 | 768 | 3072 | 110M | 512 |
| BERT large | 24 | 16 | 1024 | 4096 | 340M | 512 |
| GPT-3 | 96 | 96 | 12288 | 49152 | 175B | 2048 |
| LLaMA 3 8B | 32 | 32 | 4096 | 14336 | 8B | 8192 |
| LLaMA 3 70B | 80 | 64 | 8192 | 28672 | 70B | 8192 |

---

## WHAT CHANGED FROM 2017 TO NOW

| Component | Original (2017) | Modern LLMs (2024) |
|-----------|----------------|-------------------|
| Activation | ReLU | SwiGLU / GELU |
| Normalization | Post-Norm, LayerNorm | Pre-Norm, RMSNorm |
| Positional Enc. | Sinusoidal | RoPE (rotary) |
| Attention | Multi-Head (MHA) | Grouped Query (GQA) |
| Context | 512 tokens | 128K–1M tokens |
| Precision | FP32 | BF16 |
| Optimizer | Adam | AdamW |

---

## INTERVIEW BLAST — Say Exactly This

**"Explain transformers in 60 seconds"**
> "The transformer replaced RNNs by processing all tokens in parallel using self-attention.
> Every token creates Query, Key, and Value vectors. Attention scores are computed as
> scaled dot products of Q with all Ks, softmaxed to get weights, then applied to Vs.
> This lets every token directly attend to any other token regardless of distance —
> solving the long-range dependency problem. Multiple heads run in parallel to capture
> different relationships. Residual connections prevent vanishing gradients in deep networks.
> Every modern LLM — GPT-4, LLaMA, Gemini — uses this exact architecture."

**"Why is Q divided by √d_k?"**
> "For large d_k, dot products grow large and push softmax into a region with near-zero
> gradients — the distribution becomes too peaked. Dividing by √d_k keeps dot products
> in a stable variance range."

**"What's the difference between encoder and decoder?"**
> "Encoder uses bidirectional attention — every token sees all others. Best for
> understanding tasks like classification. Decoder uses causal attention — each token
> only sees previous ones. Best for generation. T5 uses both: encoder reads source,
> decoder generates output while cross-attending to encoder."
