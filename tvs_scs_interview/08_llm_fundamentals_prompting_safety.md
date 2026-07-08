# LLM Fundamentals, Prompt Engineering & AI Safety (Pillar)

The core knowledge under everything else. Know these cold — they're asked in every GenAI interview.

---

## 1. LLM fundamentals

- **LLM:** a transformer neural network trained to predict the next token; text in → text out; stateless per call.
- **Token:** a chunk of text (~4 chars / ~¾ word). You pay per token (input + output).
- **Context window:** max tokens per call (input + output). GPT-4o ~128K, Claude ~200K. Manage it.
- **Temperature:** randomness. 0-0.2 = deterministic/factual (extraction, classification); 0.7+ = creative. Production automations → low.
- **Top-p (nucleus sampling):** sample from the smallest set of tokens covering probability p.
- **Max tokens:** cap output length (cost + latency control).
- **Embeddings:** vectors capturing meaning (for RAG/search) — different from generation.
- **Parameters:** the model's learned weights (e.g., 8B, 70B). More ≈ more capable, costlier.

---

## 2. Prompt engineering (know every technique)

| Technique | What |
|-----------|------|
| **System prompt** | Sets role, rules, format, guardrails — the agent's "constitution" |
| **Zero-shot** | Just the instruction, no examples |
| **Few-shot** | 2-5 input→output examples to steer format/behavior |
| **Chain-of-Thought (CoT)** | "Think step by step" before answering → better reasoning |
| **ReAct** | Reason + act with tools interleaved |
| **Grounding** | "Answer ONLY from the context below; if absent, say you don't know" → cuts hallucination |
| **Structured output** | Force JSON via schema / function calling → reliable parsing |
| **Role prompting** | "You are an expert logistics analyst..." |
| **Delimiters** | Use ``` or tags to separate instructions from data (also anti-injection) |
| **Decomposition** | Break a complex task into sub-prompts |

**Context window management (JD lists):**
- Trim/summarize old conversation; retrieve only relevant chunks (don't stuff everything)
- Put the most important instructions at the start AND end (models weight edges)
- Watch total tokens = system + few-shot + history + retrieved context + question

**Prompt production practices:**
- **Version prompts** like code (they're config that changes behavior)
- **Test prompts** against an eval set before shipping
- Temperature 0-0.1 for consistency in automations

---

## 3. Cost optimization (the JD cares — say numbers)
- **Right-size the model:** cheap model (gpt-4o-mini) for classification/extraction; frontier model only for hard reasoning (you did this)
- **Cache:** repeated prompts/embeddings → Redis; semantic cache for similar queries
- **Trim tokens:** concise prompts, summarize history, retrieve fewer/better chunks (re-ranking helps)
- **Cap output:** max_tokens
- **Batch** embeddings
- **Route + fallback:** multi-LLM routing by cost/quality
- **Track cost per request** (observability) to catch runaway loops
> "I cut cost by routing simple tasks to a mini model, caching embeddings and repeated answers, trimming context with re-ranking, and monitoring token cost per request to catch loops."

---

## 4. Hallucination (guaranteed question)
**What:** the model states false things confidently.
**Why:** it predicts plausible text, not truth; gaps in training/context.
**Reduce it:**
- **Grounding** (answer only from provided context/tools)
- **RAG** (give it real, current data)
- **Tool use** (fetch facts instead of guessing) — your tracking agent never invents status
- **Structured output + validation**
- **Lower temperature**
- **Ask it to cite sources / say "I don't know"**
- **Factuality/faithfulness checks** in eval + guardrails

---

## 5. AI Safety, Guardrails & Responsible AI (JD section)

### Guardrails
- **Input:** prompt-injection detection, PII scrubbing, off-topic/abuse filtering, jailbreak detection
- **Output:** schema/JSON validation, content/safety moderation, groundedness/factuality check, refuse-if-unsupported, no-competitor/brand rules
- **Tools:** NeMo Guardrails, Llama Guard, Guardrails AI, Pydantic validators, provider moderation APIs

### Responsible AI / Governance (the 5 pillars to name)
1. **Fairness / bias** — test across demographics/languages; mitigate biased outputs
2. **Transparency / explainability** — traceable decisions, audit logs, model cards
3. **Privacy** — PII handling, data residency, GDPR; don't leak sensitive data to the model
4. **Accountability** — human-in-the-loop for high-stakes, audit trails, who approved what
5. **Safety / robustness** — guardrails, adversarial testing, graceful failure

### Compliance terms
- **SOC 2** — security/availability controls audit (enterprise trust)
- **GDPR** — EU data protection (consent, right to deletion, data minimization)
- **PII** — personally identifiable information (mask before sending to LLM APIs)

### Prompt injection (know this specifically)
An attacker puts instructions in the input/data ("ignore previous instructions and...") to hijack the model. **Defenses:** separate instructions from data with delimiters, input filtering, least-privilege tools, output validation, never blindly trust retrieved/user content.

---

## 6. Evaluation (how you know it works)
- **LLM-as-a-judge:** use a strong model to grade outputs against criteria
- **Golden/eval datasets:** fixed input→expected sets you run on every change
- **RAG eval (RAGAS):** faithfulness, context precision/recall, answer relevance
- **Regression evals in CI:** don't ship a prompt/model change that lowers scores
- **Human review sampling** in production
> "I build an eval set before shipping, run it in CI on every prompt/model change, use LLM-as-a-judge and RAGAS for RAG, and sample production outputs for human review."

---

## The sentence that proves you get LLM fundamentals
> "LLMs are next-token predictors — stateless, priced per token, bounded by a context window. I engineer prompts with system rules, few-shot, chain-of-thought, and strict grounding to cut hallucination, enforce structured output for reliability, manage context by retrieving only what's relevant, and optimize cost by right-sizing models and caching. I wrap it in guardrails and evals for a responsible, factual system."

---

## Reuse your existing deep material
Your `practical/` folder already has deep dives on transformers, tokenization, sampling, fine-tuning, RAG internals, quantization, and deployment. Skim those for extra depth — but this file is the interview-ready core.
