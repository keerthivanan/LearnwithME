# Open Source LLMs — Complete Production Guide

> Every major open-source LLM, what makes it different, when to use it,
> and what changed from the original Transformer. Nothing missing.

---

## WHY OPEN SOURCE MATTERS IN PRODUCTION

**Closed source (GPT-4, Claude, Gemini):**
- Your data goes to OpenAI/Anthropic/Google servers
- Pay per token — expensive at scale
- No customization of weights
- Dependent on API availability

**Open source (LLaMA, Mistral, Falcon):**
- Run on YOUR servers — full data privacy
- One-time GPU cost vs recurring API cost
- Fine-tune on YOUR data
- Control everything

**At $10M revenue, most companies switch from GPT-4 API to self-hosted open-source.**
That's why you need to know this as a production engineer.

---

## THE LLAMA FAMILY (Meta) — The Most Important Open-Source LLMs

### LLaMA 1 (February 2023) — The Spark

Meta released weights for 7B, 13B, 33B, and 65B parameter models.
First time a truly capable open-source LLM was available.

**Key architecture choices:**
- Decoder-only (GPT style)
- RoPE positional encoding
- RMSNorm instead of LayerNorm
- SwiGLU activation
- Pre-normalization (normalize before sublayer)
- No bias terms in linear layers (faster)

**Limitation:** Non-commercial license. Research only.

---

### LLaMA 2 (July 2023) — The Game Changer

**What changed:**
- Commercial license — companies could use it in products
- Context window: 4096 tokens (doubled from 2048)
- Grouped Query Attention (GQA) in 34B and 70B models
- Trained on 2 TRILLION tokens (vs 1.4T for LLaMA 1)
- Chat variants: fine-tuned with RLHF + safety filters

**Sizes:** 7B, 13B, 34B, 70B (+ chat variants of each)

**GQA in LLaMA 2:**
```
LLaMA 2 7B:  32 query heads, 32 KV heads (standard MHA)
LLaMA 2 70B: 64 query heads, 8 KV heads  (GQA - 8x smaller KV cache)
```
Why? 70B model's KV cache at 4096 context = 80GB. GQA reduces to 10GB.

---

### LLaMA 3 (April 2024) — State of the Art

**Major improvements:**
- New tokenizer: 128K vocabulary (vs 32K) — better for code, multilingual
- Trained on 15 TRILLION tokens (vs 2T for LLaMA 2)
- Context: 8192 tokens natively
- All sizes use GQA now (even 8B)
- Better instruction following, reasoning, coding

**Sizes:** 8B, 70B (+ instruct variants)

**LLaMA 3 8B beats LLaMA 2 70B on most benchmarks.**
8× fewer parameters, better performance. Efficiency of scale.

---

### LLaMA 3.1 (July 2024) — Long Context + Multilingual

**What changed:**
- Context: **128K tokens** (16× increase from 3.0)
- Added 405B parameter model (largest open-source model)
- Multilingual: 8 languages
- Better tool use and function calling
- Uses NTK-aware RoPE scaling for 128K context

**Sizes:** 8B, 70B, 405B (+ instruct variants)

**LLaMA 3.1 405B competes with GPT-4.**
This changed the entire industry. Enterprise-grade quality, open weights.

---

### LLaMA 3.2 (September 2024) — Multimodal

- 1B and 3B models for edge/mobile
- 11B and 90B vision models (image + text)
- First LLaMA with image understanding

---

## MISTRAL FAMILY (Mistral AI) — Efficiency Champions

### Mistral 7B (September 2023)

**Beat LLaMA 2 13B on every benchmark at half the size. How?**

**Two key innovations:**

**1. Sliding Window Attention (SWA):**
Instead of attending to all previous tokens, each token attends only to
the nearest W tokens (window = 4096).

```
Standard:   token 500 attends to tokens 1-499 (499 lookups)
Sliding:    token 500 attends to tokens 497-500 (4 lookups, window=4)
```

Memory: O(n²) → O(n × W). Handle 32K context with 4K window size.

But wait — doesn't this miss long-range context?
No, because through multiple layers, information propagates:
- Layer 1: token 500 sees tokens 496-500
- Layer 2: token 500 now has info from tokens 492-500 (through layer 1's output)
- Layer N: token 500 effectively has info from far back

**2. Grouped Query Attention (GQA):**
8 query heads share 1 KV head. 8× smaller KV cache than standard attention.

**3. Rolling Buffer Cache:**
KV cache has fixed size. Old tokens are evicted. Enables infinite generation
without growing memory.

---

### Mistral 8×7B — Mixtral (December 2023)

**The Mixture of Experts (MoE) revolution.**

**How MoE works:**
```
Standard FFN: ALL 8 experts process every token
              Compute: 8 × FFN cost

MoE:          Router decides which 2 experts handle this token
              Compute: 2 × FFN cost
              Parameters: 8 × FFN count (all 8 stored in memory)
```

Mixtral 8×7B has 46.7B parameters total, but only uses 12.9B per token.
**Speed of 12.9B. Quality of ~65B. Genius design.**

Beats LLaMA 2 70B on most benchmarks.
Matches GPT-3.5 on many tasks.

**Why it matters for production:**
- Cheaper inference than 70B (fewer active params)
- Higher quality than 13B (more total capacity)
- First major MoE that was accessible

---

### Mistral 7B v0.3 and Mistral Large

- v0.3: function calling support, 32K context
- Mistral Large: closed source, competes with GPT-4
- Codestral: code-specialized Mistral

---

## FALCON FAMILY (Technology Innovation Institute, UAE)

### Falcon 7B / 40B (May 2023)

**Key innovation: Multi-Query Attention (MQA)**
```
Multi-Head (standard): 71 Q heads, 71 K heads, 71 V heads
Multi-Query:           71 Q heads, 1 K head,  1 V head
```
Even more aggressive than GQA. Extreme KV cache compression.

**Training data: RefinedWeb** — high-quality filtered web data.
The filtering strategy was as important as the architecture.

---

## PHI FAMILY (Microsoft) — Small but Mighty

### Phi-1 (2023) — Code focused, 1.3B parameters
**Key insight:** Quality of training data > quantity.
Trained on "textbook quality" synthetic data. Matched much larger models on code.

### Phi-2 (December 2023) — 2.7B parameters
Outperformed 7B models on reasoning benchmarks.
Showed you don't need billions of parameters if data quality is right.

### Phi-3 Mini (April 2024) — 3.8B parameters
**Trained on GPT-4 generated "textbook" data** (essentially knowledge distillation).
Competes with Mistral 7B and LLaMA 3 8B at 3.8B parameters.
Runs on a phone (4-bit quantized = ~2GB).

**Phi-3's lesson:** LLM capability = Architecture × Data Quality × Scale.
Microsoft found a point where data quality compensated for scale.

---

## GEMMA FAMILY (Google DeepMind)

### Gemma 2B / 7B (February 2024)
- Based on Gemini architecture research
- Multi-query attention
- RoPE positional encoding
- GeGLU activation
- Strong on math and reasoning for their size

### Gemma 2 (June 2024)
- 9B and 27B models
- Sliding window attention alternated with global attention
- Knowledge distillation from larger Gemma models
- 27B model competes with 70B models

---

## QWEN FAMILY (Alibaba) — Best Multilingual

### Qwen 2 (June 2024)
- 0.5B, 1.5B, 7B, 57B-A14B (MoE), 72B
- **Best Chinese + English bilingual models**
- 128K context window
- Strong on coding and math
- 72B model competes with LLaMA 3 70B

---

## DEEPSEEK FAMILY (DeepSeek AI, China)

### DeepSeek-V2 (May 2024) — MoE Architecture
- 236B total params, 21B active (MoE)
- Multi-head Latent Attention (MLA) — new attention variant
- Competes with GPT-4 at fraction of inference cost

### DeepSeek-R1 (January 2025) — Reasoning Champion
- Reasoning model trained with RL (similar to OpenAI o1)
- Chain-of-thought reasoning baked into the model
- Matched OpenAI o1 on math/coding benchmarks
- **Fully open source** — huge deal

---

## CODE-SPECIALIZED MODELS

| Model | Base | Strength |
|-------|------|---------|
| Code LLaMA 7B/13B/34B | LLaMA 2 | Code generation, infilling |
| StarCoder 2 | Custom | 600+ programming languages |
| DeepSeek-Coder | DeepSeek | Strong on competitive programming |
| Codestral | Mistral | Code + 80+ languages |
| Qwen2.5-Coder | Qwen | Best open-source code model (2024) |

---

## EMBEDDING MODELS (Used in RAG)

These don't generate text — they convert text to vectors for semantic search.

| Model | Dimensions | Notes |
|-------|-----------|-------|
| BAAI/bge-large-en-v1.5 | 1024 | Best English embedding (open) |
| BAAI/bge-m3 | 1024 | Multilingual, long context |
| nomic-embed-text-v1.5 | 768 | 8192 context, fully open |
| E5-large-v2 | 1024 | Microsoft, strong performance |
| text-embedding-ada-002 | 1536 | OpenAI, paid API |
| text-embedding-3-large | 3072 | OpenAI, best quality |

**In production RAG:** BAAI/bge-large-en-v1.5 is the default choice.

---

## HOW TO CHOOSE AN LLM FOR PRODUCTION

```
Task: General chat / instruction following
  → LLaMA 3.1 8B Instruct (best quality per compute)
  → LLaMA 3.1 70B Instruct (if budget allows)

Task: Code generation
  → Qwen2.5-Coder 7B/32B
  → DeepSeek-Coder-V2

Task: Long documents (RAG over large docs)
  → LLaMA 3.1 (128K context)
  → Mistral 7B with extended context

Task: Edge / mobile deployment
  → Phi-3 Mini 3.8B
  → LLaMA 3.2 1B/3B

Task: High throughput, cost-sensitive
  → Mixtral 8×7B (MoE - fast inference)
  → Mistral 7B

Task: Best possible quality (open source)
  → LLaMA 3.1 405B
  → DeepSeek-V2

Task: Multilingual
  → Qwen2 72B
  → BAAI/bge-m3 for embeddings
```

---

## KEY ARCHITECTURAL COMPARISON TABLE

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

---

## MIXTURE OF EXPERTS (MoE) — DEEP DIVE

### Why MoE exists:
Scaling laws say: 2× parameters = better model.
But 2× parameters = 2× inference cost. Not sustainable.

### MoE insight:
Train a large model but only use a fraction of it per token.

```
Router Network → for each token, choose top-K experts

Token "Paris" → Router → [Expert 3, Expert 7] activated
Token "def"   → Router → [Expert 1, Expert 12] activated
```

### Sparse MoE vs Dense:
```
Dense 7B:    ALL 7B params active for every token
MoE 47B:     47B total params, 12.9B active per token
```

### Why this works:
Different experts specialize in different types of tokens/knowledge.
- Some experts become "code experts"
- Some become "math experts"
- Some become "language-specific experts"

This happens naturally during training — not designed by humans.

### Load Balancing Problem:
Without constraints, ALL tokens route to the same 2 experts → other experts never train.
Solution: add auxiliary load balancing loss to force even distribution.

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
