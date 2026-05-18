# Open Source LLMs — Complete Production Guide

> Every major open-source LLM, what makes it different, when to use it,
> and what changed from the original Transformer. Nothing missing.

---

## WHY OPEN SOURCE MATTERS IN PRODUCTION

**What it is:** The business and technical case for using open-source models instead of paid APIs — understanding this makes you sound like an engineer, not just a hobbyist.

**Closed source (GPT-4, Claude, Gemini):**
- Your data goes to OpenAI/Anthropic/Google servers — privacy risk for sensitive data
- Pay per token — expensive at scale (millions of queries/day = large bills)
- No customization of weights — can't fine-tune on your exact data
- Dependent on API availability — if the API goes down, you go down

**Open source (LLaMA, Mistral, Falcon):**
- Run on YOUR servers — full data privacy, no data leaves your infrastructure
- One-time GPU cost vs recurring API cost — amortized over time
- Fine-tune on YOUR data — specialized models for your exact use case
- Control everything — version, deployment, updates, customization

**At $10M revenue, most companies switch from GPT-4 API to self-hosted open-source.**
That's why you need to know this as a production engineer. At scale, the cost savings are enormous.

**Analogy:** Renting a car (API) vs buying a car (open-source). Renting is great for occasional use. Once you're driving 500 miles a day, buying is cheaper.

---

## THE LLAMA FAMILY (Meta) — The Most Important Open-Source LLMs

**What it is:** Meta's series of open-weight models that created the open-source LLM ecosystem — each version significantly improved on the last.

### LLaMA 1 (February 2023) — The Spark

**What it is:** The first truly capable open-source LLM — the model that proved you didn't need to be OpenAI to build state-of-the-art language models.

Meta released weights for 7B, 13B, 33B, and 65B parameter models.
First time a truly capable open-source LLM was available.

**Key architecture choices:**
- Decoder-only (GPT style) — generation, not understanding
- RoPE positional encoding — relative positions, better generalization
- RMSNorm instead of LayerNorm — 15% faster, same quality
- SwiGLU activation — better than GELU in practice
- Pre-normalization (normalize before sublayer, not after) — more stable training
- No bias terms in linear layers (faster matmuls, reduces parameters slightly)

**Limitation:** Non-commercial license. Research only. Companies couldn't ship products with it.

---

### LLaMA 2 (July 2023) — The Game Changer

**What it is:** The version that enabled an industry — a commercial license meant companies could actually deploy it in products.

**What changed:**
- **Commercial license** — companies could use it in products (huge deal)
- Context window: 4096 tokens (doubled from 2048) — handles longer documents
- Grouped Query Attention (GQA) in 34B and 70B models — 8× KV cache reduction
- Trained on 2 TRILLION tokens (vs 1.4T for LLaMA 1) — more knowledge
- Chat variants: fine-tuned with RLHF + safety filters — safe for deployment

**Sizes:** 7B, 13B, 34B, 70B (+ chat variants of each)

**GQA in LLaMA 2:**
```python
# LLaMA 2 model sizes and their attention configuration:

# LLaMA 2 7B: standard MHA (no GQA)
num_query_heads = 32   # 32 query heads
num_kv_heads    = 32   # 32 KV heads — same as query heads (full MHA)
# KV cache per token: 32 K-heads + 32 V-heads

# LLaMA 2 70B: GQA (massive KV cache reduction)
num_query_heads = 64   # 64 query heads
num_kv_heads    = 8    # only 8 KV heads (8 query heads share 1 KV head)
# KV cache per token: 8 K-heads + 8 V-heads (vs 64+64 for full MHA)
# Compression: 64/8 = 8× smaller KV cache

# Why does this matter?
# At 4096 context, 70B model KV cache:
# Without GQA: 2 × 80 × 64 × 128 × 4096 × 2 bytes = 85 GB
# With GQA:   2 × 80 × 8  × 128 × 4096 × 2 bytes = 10.7 GB
# GQA makes the model deployable on reasonable hardware
```
Why? 70B model's KV cache at 4096 context = 80GB without GQA. GQA reduces to 10GB.

---

### LLaMA 3 (April 2024) — State of the Art

**What it is:** A massive generational leap — a 128K vocabulary, 15 trillion training tokens, and performance that surpassed many closed models.

**Major improvements:**
- **New tokenizer: 128K vocabulary** (vs 32K) — better for code, multilingual text, special characters
- Trained on **15 TRILLION tokens** (vs 2T for LLaMA 2) — 7.5× more data
- Context: 8192 tokens natively
- All sizes use GQA now (even 8B) — consistent memory efficiency
- Better instruction following, reasoning, coding

**Sizes:** 8B, 70B (+ instruct variants)

**LLaMA 3 8B beats LLaMA 2 70B on most benchmarks.**
8× fewer parameters, better performance. This is the power of data quality and quantity.

**WHY 128K vocabulary matters:** More vocabulary means common phrases become single tokens. "astronaut" might be 1 token (in the 128K vocab) vs 3 tokens ("astro" + "na" + "ut" in 32K vocab). Fewer tokens per sentence = longer effective context for the same context window.

---

### LLaMA 3.1 (July 2024) — Long Context + Multilingual

**What it is:** The version that brought LLaMA to GPT-4-competing quality while adding 128K context and multilingual capability.

**What changed:**
- Context: **128K tokens** (16× increase from 3.0) — entire codebases, long papers
- Added 405B parameter model (largest open-source model) — GPT-4-class quality
- Multilingual: 8 languages
- Better tool use and function calling
- Uses NTK-aware RoPE scaling for 128K context

**Sizes:** 8B, 70B, 405B (+ instruct variants)

**LLaMA 3.1 405B competes with GPT-4.**
This changed the entire industry. Enterprise-grade quality, open weights.
A company can now run a GPT-4-quality model on their own infrastructure.

---

### LLaMA 3.2 (September 2024) — Multimodal

**What it is:** LLaMA's expansion into image understanding and tiny models for edge devices.

- 1B and 3B models for edge/mobile (fits on a phone)
- 11B and 90B vision models (image + text understanding)
- First LLaMA with image understanding — analyze charts, photos, diagrams

---

## MISTRAL FAMILY (Mistral AI) — Efficiency Champions

**What it is:** Models from the French startup Mistral AI that consistently punch above their weight class — achieving quality far beyond what their parameter count would suggest.

### Mistral 7B (September 2023)

**Beat LLaMA 2 13B on every benchmark at half the size. How?**

**Analogy:** A Ferrari beating an SUV in a race — not by being bigger, but by being more engineered.

**Two key innovations:**

**1. Sliding Window Attention (SWA):**

**What it is:** Instead of every token attending to all previous tokens, each token only attends to the nearest W tokens. Information from farther back still reaches the model through multiple layers.

Instead of attending to all previous tokens, each token attends only to
the nearest W tokens (window = 4096).

```
Standard attention:
  token 500 attends to tokens 1-499 (499 lookups)
  ← all past tokens are consulted

Sliding window:
  token 500 attends to tokens 497-500 (4 lookups, window=4)
  ← only recent neighbors consulted
```

Memory: O(n²) → O(n × W). Handle 32K context with 4K window size.
Much cheaper to compute: 4K×4K matrix instead of 32K×32K matrix.

But wait — doesn't this miss long-range context?
No, because through multiple layers, information propagates:

```
Layer 1: token 500 sees tokens 496-500           (window=4 example)
Layer 2: token 500 now has info from tokens 492-500
         (because in layer 1, token 496 already saw tokens 492-496)
Layer 3: token 500 effectively has info from tokens 488-500
         Each layer extends the effective receptive field by W tokens
Layer N: token 500 effectively has info from far back
         (effective receptive field = W × N layers)
```

**WHY this works:** Each layer's sliding window extends reach by W more tokens. With 32 layers and window=4096, effective receptive field = 32 × 4096 = 131K tokens. Information propagates like ripples — not as far as full attention, but far enough for most tasks.

**2. Grouped Query Attention (GQA):**
8 query heads share 1 KV head. 8× smaller KV cache than standard attention.

**3. Rolling Buffer Cache:**

**What it is:** A fixed-size KV cache that evicts old tokens when full — enabling infinite generation without growing memory.

KV cache has fixed size. Old tokens are evicted. Enables infinite generation
without growing memory.
```
Buffer size = 4096 tokens (fixed)
When buffer is full: oldest tokens are dropped
New tokens always fit in the buffer
← Memory usage stays constant regardless of generation length
← Trade-off: can't recall exact content from very early in the conversation
```

---

### Mistral 8×7B — Mixtral (December 2023)

**What it is:** Mistral's Mixture-of-Experts model — the model that brought MoE to the open-source world and showed you can get 65B-quality at 13B inference cost.

**The Mixture of Experts (MoE) revolution.**

**How MoE works:**
```
Standard FFN: ALL parameters are active for every token
              Compute: full FFN cost for every single token

MoE:          A router decides which 2 of 8 experts handle this token
              Compute: 2 × FFN cost (instead of 8 × FFN cost)
              Parameters: 8 × FFN count (all 8 stored in memory)
```

```python
# MoE routing in Mixtral:
# For each token, the router picks 2 of 8 experts:
router_scores = linear_router(token)  # shape: (8,) — one score per expert
# Higher score = this expert is more relevant for this token type

top2_experts = router_scores.topk(2)  # pick the 2 highest-scoring experts
# Example: token "def" (Python keyword) → experts [1, 5] activated
# Example: token "Paris" (geography) → experts [2, 7] activated
# Different token types route to different expert combinations

# Only run the 2 selected experts (6 are completely idle for this token):
expert_1_output = expert_networks[top2_experts[0]](token)  # one FFN forward pass
expert_2_output = expert_networks[top2_experts[1]](token)  # one FFN forward pass
# Experts 0, 2, 3, 4, 5, 6 (the other 6): skipped entirely, no compute

# Weighted combination:
output = gate_weight_1 * expert_1_output + gate_weight_2 * expert_2_output
# gate weights come from the router scores (normalized)
# Expert with higher score contributes more to the output
```

Mixtral 8×7B has 46.7B parameters total, but only uses 12.9B per token.
**Speed of 12.9B. Quality of ~65B. Genius design.**

Beats LLaMA 2 70B on most benchmarks.
Matches GPT-3.5 on many tasks.

**Why it matters for production:**
- Cheaper inference than 70B (fewer active params = faster per-token generation)
- Higher quality than 13B (more total capacity = more learned knowledge)
- First major MoE that was accessible to the open-source community

---

### Mistral 7B v0.3 and Mistral Large
- v0.3: function calling support, 32K context
- Mistral Large: closed source, competes with GPT-4 Turbo
- Codestral: code-specialized Mistral for programming tasks

---

## FALCON FAMILY (Technology Innovation Institute, UAE)

**What it is:** UAE's contribution to open-source LLMs — notable for aggressive KV cache compression and high-quality training data.

### Falcon 7B / 40B (May 2023)

**Key innovation: Multi-Query Attention (MQA)**

**What it is:** The most aggressive form of KV cache compression — ALL query heads share a single K and V head.

```
Multi-Head (standard): 71 Q heads, 71 K heads, 71 V heads
                       ← full separate K,V per query head
Multi-Query:           71 Q heads, 1 K head,  1 V head
                       ← ALL 71 query heads share ONE K and ONE V head
                       ← KV cache is 71× smaller than MHA
```

Even more aggressive than GQA. Extreme KV cache compression.

**Training data: RefinedWeb** — high-quality filtered web data.
The filtering strategy was as important as the architecture.

**WHY MQA works despite sharing one K/V:** The model learns to extract sufficient information from the shared key and value. The expressiveness comes from 71 different query projections asking 71 different questions of the same K/V representation. Quality drops somewhat vs MHA, but the memory savings are extreme.

---

## PHI FAMILY (Microsoft) — Small but Mighty

**What it is:** Microsoft Research's exploration of a radical hypothesis — that data quality matters more than data quantity, enabling small models to match much larger ones.

### Phi-1 (2023) — Code focused, 1.3B parameters
**Key insight:** Quality of training data > quantity.
Trained on "textbook quality" synthetic data. Matched much larger models on code.

### Phi-2 (December 2023) — 2.7B parameters
Outperformed 7B models on reasoning benchmarks.
Showed you don't need billions of parameters if data quality is right.

### Phi-3 Mini (April 2024) — 3.8B parameters

**What it is:** A 3.8B model trained on GPT-4-generated "textbook" content that competes with 7B models — essentially knowledge distillation at the data level.

**Trained on GPT-4 generated "textbook" data** (essentially knowledge distillation via data).
Competes with Mistral 7B and LLaMA 3 8B at 3.8B parameters.
Runs on a phone (4-bit quantized = ~2GB).

```python
# Why Phi-3 works — data quality hypothesis:
# Standard LLM training: scrape all the internet
# 1 trillion tokens including: spam, arguments, poorly written text, jokes, noise

# Phi-3 approach: filter for "textbook quality"
# Train on: textbooks, structured tutorials, well-reasoned explanations, worked examples
# Also add: GPT-4-generated synthetic textbooks on each topic

# Result: 3.8B model trained on "only the good stuff"
# ≈ 7B model trained on mixed internet data
# Intuition: a student who reads 100 great books learns more than one who reads
#            10,000 random webpages
```

**Phi-3's lesson:** LLM capability = Architecture × Data Quality × Scale.
Microsoft found a point where data quality compensated for scale.

---

## GEMMA FAMILY (Google DeepMind)

**What it is:** Google's open-source models based on their internal Gemini research — bringing Google's architectural innovations to the open-source community.

### Gemma 2B / 7B (February 2024)
- Based on Gemini architecture research
- Multi-query attention — aggressive KV compression
- RoPE positional encoding
- GeGLU activation (variant of SwiGLU using GELU instead of SiLU)
- Strong on math and reasoning for their size

### Gemma 2 (June 2024)

**What it is:** Google's improved open-source models using alternating attention patterns and knowledge distillation from larger models.

- 9B and 27B models
- Sliding window attention alternated with global attention
  (some layers use window, some use full — balances efficiency and quality)
- Knowledge distillation from larger Gemma models
- 27B model competes with 70B models

**WHY alternating attention?** Window attention is cheap but misses long-range context. Full attention is expensive but captures everything. Alternating (say, window for 3 layers, full for 1 layer) gives most of the quality benefit at a fraction of the cost.

---

## QWEN FAMILY (Alibaba) — Best Multilingual

**What it is:** Alibaba's series of models with particular strength in Chinese + English and strong multilingual capability.

### Qwen 2 (June 2024)
- 0.5B, 1.5B, 7B, 57B-A14B (MoE), 72B
- **Best Chinese + English bilingual models** — trained with equal Chinese/English data
- 128K context window
- Strong on coding and math
- 72B model competes with LLaMA 3 70B

---

## DEEPSEEK FAMILY (DeepSeek AI, China)

**What it is:** Chinese AI lab's models that attracted global attention by matching frontier models at fraction of training cost.

### DeepSeek-V2 (May 2024) — MoE Architecture
- 236B total params, 21B active (MoE) — speed of 21B, quality of 236B
- Multi-head Latent Attention (MLA) — new attention variant reducing KV cache by 93%
- Competes with GPT-4 at fraction of inference cost

### DeepSeek-R1 (January 2025) — Reasoning Champion

**What it is:** The first fully open-source reasoning model matching OpenAI's o1 — a landmark for AI transparency.

- Reasoning model trained with RL (similar to OpenAI o1)
- Chain-of-thought reasoning baked into the model
- Matched OpenAI o1 on math/coding benchmarks
- **Fully open source** — weights AND training methodology published
- Community immediately created distilled versions (7B-70B from DeepSeek-R1's reasoning)

---

## CODE-SPECIALIZED MODELS

**What it is:** Models specifically optimized for programming tasks — fine-tuned on code repositories with code-specific training procedures.

| Model | Base | Strength |
|-------|------|---------|
| Code LLaMA 7B/13B/34B | LLaMA 2 | Code generation, infilling (fill in the middle) |
| StarCoder 2 | Custom | 600+ programming languages — broadest coverage |
| DeepSeek-Coder | DeepSeek | Strong on competitive programming problems |
| Codestral | Mistral | Code + 80+ languages — fast inference |
| Qwen2.5-Coder | Qwen | Best open-source code model as of 2024 |

**WHY code-specialized models exist:** Code has different statistical properties than English — variable names, syntax rules, indentation patterns. Fine-tuning on code teaches the model these patterns. A general model knows English well but not necessarily Python syntax conventions.

---

## EMBEDDING MODELS (Used in RAG)

**What it is:** Models that convert text into vector representations (embeddings) for semantic search — not for generation, but for finding similar documents.

These don't generate text — they convert text to vectors for semantic search.

| Model | Dimensions | Notes |
|-------|-----------|-------|
| BAAI/bge-large-en-v1.5 | 1024 | Best English embedding (open) |
| BAAI/bge-m3 | 1024 | Multilingual, long context (8192 tokens) |
| nomic-embed-text-v1.5 | 768 | 8192 context, fully open (training code published) |
| E5-large-v2 | 1024 | Microsoft, strong performance |
| text-embedding-ada-002 | 1536 | OpenAI, paid API |
| text-embedding-3-large | 3072 | OpenAI, best quality |

```python
# Using an embedding model for semantic search:
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-large-en-v1.5')
# This model converts text → 1024-dimensional vectors
# Similar texts will have similar (close) vectors

documents = [
    "The cat sat on the mat",
    "A feline rested upon the rug",
    "The stock market crashed today"
]
embeddings = model.encode(documents)
# embeddings shape: (3, 1024) — one 1024-dim vector per document
# documents 0 and 1 will have high cosine similarity (same meaning)
# document 2 will be far from documents 0 and 1 (different topic)

# For a query:
query = "Where is the cat?"
query_embedding = model.encode([query])  # shape: (1, 1024)
# Compute similarity to find relevant documents
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(query_embedding, embeddings)
# similarities: [[0.92, 0.88, 0.21]] — cat documents much more similar than stock market
```

**In production RAG:** BAAI/bge-large-en-v1.5 is the default choice.

---

## HOW TO CHOOSE AN LLM FOR PRODUCTION

**What it is:** A decision tree for selecting the right model for your use case — this is what interviewers want to hear when they ask "how would you approach this?"

```
Task: General chat / instruction following
  → LLaMA 3.1 8B Instruct (best quality per compute — first choice)
  → LLaMA 3.1 70B Instruct (if budget allows and quality matters more than speed)

Task: Code generation
  → Qwen2.5-Coder 7B/32B (best open-source code model 2024)
  → DeepSeek-Coder-V2 (strong alternative)

Task: Long documents (RAG over large docs)
  → LLaMA 3.1 (128K context native)
  → Mistral 7B with extended context (cost-efficient)

Task: Edge / mobile deployment (runs on-device)
  → Phi-3 Mini 3.8B (small but capable)
  → LLaMA 3.2 1B/3B (Meta's edge models)

Task: High throughput, cost-sensitive
  → Mixtral 8×7B (MoE — fast inference, 46.7B quality at 12.9B cost)
  → Mistral 7B (fast, lightweight)

Task: Best possible quality (open source, no budget constraint)
  → LLaMA 3.1 405B (GPT-4-class quality, open weights)
  → DeepSeek-V2 (236B MoE, very capable)

Task: Multilingual (non-English languages)
  → Qwen2 72B (best Chinese+English bilingual)
  → BAAI/bge-m3 for multilingual embeddings
```

---

## KEY ARCHITECTURAL COMPARISON TABLE

**What it is:** A side-by-side comparison of every architectural choice across major models — memorize this for interviews.

| Model | Attention | Norm | Activation | Position | Context |
|-------|-----------|------|-----------|----------|---------|
| GPT-2 | MHA | Post-LN | GELU | Learned | 1024 |
| BERT | MHA | Post-LN | GELU | Learned | 512 |
| LLaMA 1/2 | MHA/GQA | Pre-RMSNorm | SwiGLU | RoPE | 2K/4K |
| LLaMA 3.1 | GQA | Pre-RMSNorm | SwiGLU | RoPE | 128K |
| Mistral 7B | GQA | Pre-RMSNorm | SwiGLU | RoPE | 32K |
| Mixtral | GQA+MoE | Pre-RMSNorm | SwiGLU | RoPE | 32K |
| Gemma 2 | GQA+SWA | Pre-RMSNorm | GeGLU | RoPE | 8K |
| Phi-3 | MHA | Pre-LN | SwiGLU | RoPE | 128K |

**Pattern to notice:** Every modern model has converged to Pre-norm + SwiGLU/GeGLU + RoPE. The choice of attention type (MHA vs GQA vs MoE) is the main differentiator.

---

## MIXTURE OF EXPERTS (MoE) — DEEP DIVE

**What it is:** The architectural pattern that decouples model quality (total parameters) from inference cost (active parameters) — the key innovation for scaling efficiently.

### Why MoE exists:
Scaling laws say: 2× parameters = better model.
But 2× parameters = 2× inference cost. Not sustainable.

### MoE insight:
Train a large model but only use a fraction of it per token.

```
Router Network → for each token, choose top-K experts

Token "Paris" → Router → [Expert 3, Expert 7] activated
Token "def"   → Router → [Expert 1, Expert 12] activated
← Different content → different experts → specialization emerges
← Content that appears together in training routes to the same experts
```

### Sparse MoE vs Dense:
```
Dense 7B:    ALL 7B params active for every token
             7B × (cost per param per forward pass) = full compute

MoE 47B:     47B total params, 12.9B active per token
             12.9B × (cost per param per forward pass) = 27% of dense 47B cost
```

### Why this works:
Different experts specialize in different types of tokens/knowledge.
- Some experts become "code experts" — see lots of code during training, specialize
- Some become "math experts" — handle equations and numerical reasoning
- Some become "language-specific experts" — French, German, Chinese routing

This happens naturally during training — not designed by humans.
The load balancing loss ensures all experts train; specialization happens organically.

### Load Balancing Problem:

**What it is:** A training failure mode where all tokens collapse onto the same 1-2 experts, leaving the rest permanently undertrained.

Without constraints, ALL tokens route to the same 2 experts → other experts never train.
Solution: add auxiliary load balancing loss to force even distribution.

```python
# Load balancing loss calculation:
def load_balancing_loss(router_probs, alpha=0.01):
    # router_probs: (batch, seq, num_experts) — soft probabilities before top-k selection
    # We want: each expert handles approximately equal fraction of tokens

    # Fraction of tokens routed to each expert (hard, top-k decision):
    # Count how many tokens each expert actually processed
    expert_counts = (top_k_assignments == expert_id).float().mean()
    # Shape: (num_experts,) — fraction of tokens going to each expert
    # Perfect balance: each expert = 1/num_experts

    # Average router probability for each expert:
    expert_avg_prob = router_probs.mean(dim=[0, 1])
    # Shape: (num_experts,) — average soft probability for each expert
    # Perfect balance: each = 1/num_experts

    # Loss: high when routing is concentrated (rich-get-richer)
    # Dot product of counts × probs: minimized when both are uniform
    loss = num_experts * (expert_counts * expert_avg_prob).sum()
    return alpha * loss  # alpha scales the weight vs main task loss
    # alpha=0.01: auxiliary loss is 1% of main loss — influences without dominating
```

---

## INTERVIEW BLAST — Open Source LLMs

**"Which open-source LLM would you use in production and why?"**
> "For most production use cases, LLaMA 3.1 8B Instruct is my go-to. It has 128K context,
> strong instruction following, runs on a single A100 40GB in INT8, and outperforms
> LLaMA 2 70B. For higher quality with more budget, LLaMA 3.1 70B. For cost-sensitive
> high-throughput, Mixtral 8×7B — 47B params but only 12.9B active per token."

**"What is Grouped Query Attention and why does LLaMA use it?"**
> "GQA shares KV heads across multiple query heads. LLaMA 3 70B has 64 query heads
> but only 8 KV heads — 8× smaller KV cache. At 128K context with batch size 32,
> this is the difference between fitting in memory and not."

**"What is Mixture of Experts?"**
> "MoE trains many specialized sub-networks (experts) but only routes each token
> to a few of them. Mixtral 8×7B has 8 expert FFNs per layer but uses 2 per token.
> You get the quality of a 47B model at the inference cost of a 12.9B model."
