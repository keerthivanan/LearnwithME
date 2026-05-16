# Interview Master Sheet — 50 Production-Level Q&A

> Study this on Day 7. Every answer is written the way you should SAY it in the interview.
> Short. Confident. With real production context.

---

## SECTION 1: TRANSFORMERS & ARCHITECTURE (Most Asked)

---

### Q1. Explain the Transformer architecture.

> A Transformer has two core parts per layer: **Multi-Head Self-Attention** and a **Feed-Forward Network**, with residual connections and layer norm around each.
>
> The attention mechanism lets each token look at every other token and decide how much to attend to it — computed as `softmax(QKᵀ / √d_k) × V`. Running this in multiple heads in parallel lets the model capture different types of relationships simultaneously.
>
> This replaced RNNs because it parallelizes over the entire sequence (no sequential dependency), handles long-range dependencies directly, and scales to billions of parameters efficiently.

---

### Q2. What is self-attention? Walk me through the math.

> Every token projects itself into three vectors — **Query** (what am I looking for?), **Key** (what do I contain?), and **Value** (what do I give?).
>
> ```
> Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V
> ```
>
> Step by step: compute dot products of Q with all Ks (similarity scores), divide by √d_k to prevent large values from saturating softmax, apply softmax to get a probability distribution, then take a weighted sum of Vs.
>
> The result: each token's output is a blend of all other tokens, weighted by relevance.

---

### Q3. Why do we divide by √d_k in attention?

> For large d_k, dot products grow large in magnitude. This pushes softmax into regions with near-zero gradients (very peaked distribution). Dividing by √d_k keeps the values in a stable range for softmax to work properly.

---

### Q4. What is causal masking? Why does GPT use it?

> Causal masking prevents each token from attending to **future** tokens. During training, we mask out future positions by setting those attention scores to -∞ before softmax (which becomes 0 after softmax).
>
> GPT needs this because it's **autoregressive** — it generates the next token based only on past tokens. If it could see future tokens during training, it would "cheat" and not learn to predict them. At inference time, future tokens don't exist yet anyway.

---

### Q5. What is the difference between BERT and GPT?

> | | BERT | GPT |
> |-|------|-----|
> | Architecture | Encoder-only | Decoder-only |
> | Attention | Bidirectional (sees all tokens) | Causal (left-to-right only) |
> | Training | Masked Language Modeling | Next-token prediction |
> | Best for | Classification, NER, embeddings | Text generation |
>
> BERT sees full context so it understands meaning better. GPT generates coherently because it's trained to predict what comes next. You can't directly generate text with BERT, and BERT isn't naturally suited for open-ended generation.

---

### Q6. What is T5 and what makes it different?

> T5 (Text-to-Text Transfer Transformer) uses an **encoder-decoder** architecture and frames **every** NLP task as text-to-text — translation, summarization, classification all use the same format:
> ```
> Input:  "summarize: [article]"
> Output: "[summary]"
> ```
> This unified interface means one model, one training objective, works for everything. It uses span corruption (mask spans of text, not individual tokens) as its pre-training objective.

---

### Q7. What is a KV cache and why does it matter in production?

> During autoregressive generation, the Key and Value matrices for all previous tokens don't change — only the new token's K,V are new. KV cache stores the previous K,V so we don't recompute them every step.
>
> Without it: O(n²) compute per generation step. With it: O(n) per step.
>
> In production this is critical. A KV cache for a long conversation (4096 tokens) with a 7B model can be several GBs. Managing KV cache memory efficiently (PagedAttention in vLLM) is what enables high-throughput serving.

---

### Q8. What is Flash Attention?

> Flash Attention rewrites the attention computation to minimize GPU memory reads/writes (I/O bound, not compute bound). Standard attention materializes the full N×N attention matrix in HBM (high bandwidth memory) — Flash Attention tiles the computation so it stays in fast SRAM.
>
> Same mathematical result. 2-4× faster. Enables much longer context lengths. Flash Attention 2/3 is now standard in all serious LLM training and inference.

---

### Q9. What improvements does LLaMA have over the original Transformer?

> Four key changes:
> 1. **RoPE** (Rotary Positional Encoding) instead of sinusoidal — better at length generalization
> 2. **SwiGLU** activation instead of ReLU/GELU — empirically better
> 3. **RMSNorm** instead of LayerNorm — simpler (no mean subtraction), faster
> 4. **Pre-norm** (normalize before sublayer, not after) — more training stable
> 5. **GQA** (Grouped Query Attention) in larger models — reduces KV cache size significantly

---

### Q10. What is Grouped Query Attention (GQA)?

> In standard multi-head attention, every query head has its own K,V heads — this makes the KV cache scale linearly with number of heads.
>
> GQA groups multiple query heads to share a single K,V head. LLaMA 2 70B uses 8 K,V groups for 64 query heads — 8× smaller KV cache. This allows higher batch sizes and longer contexts in production without running out of memory.

---

## SECTION 2: TRAINING & FINE-TUNING

---

### Q11. What is LoRA and how does it work mathematically?

> LoRA (Low-Rank Adaptation) keeps the pre-trained weights **frozen** and adds trainable low-rank matrices alongside them:
>
> ```
> W' = W + ΔW = W + B × A
> ```
> where W is d×k (frozen), A is r×k, B is d×r (both trainable), and r << d,k.
>
> The hypothesis: weight updates during fine-tuning have low **intrinsic rank** — they don't need a full d×k matrix to represent useful adaptations. Using r=16 on a 7B model gives only ~4M trainable parameters (0.06% of total).
>
> At inference, you can merge: `W' = W + BA` — zero overhead compared to base model.

---

### Q12. What is QLoRA and when do you use it?

> QLoRA applies LoRA on top of a **4-bit quantized** base model (NF4 format). The base model is quantized and frozen; LoRA adapters train in BF16.
>
> Result: fine-tune a 7B model on a **single 8GB GPU**. Without QLoRA you'd need 56GB+ for full fine-tuning. Quality is close to full fine-tuning for most tasks.
>
> Use QLoRA when: you have limited GPU budget, you want to run fine-tuning on a single GPU or colab, you're prototyping. For production-grade fine-tuning with budget, full fine-tuning or LoRA on full-precision model is better.

---

### Q13. LoRA hyperparameters — what do r and alpha do?

> - **r (rank)**: controls how many parameters are in the adapter. Higher r = more capacity = more memory and compute. r=8 is lightweight, r=64 is high-capacity. Start with r=16.
> - **alpha**: scaling factor applied to LoRA output: `output = W_base + (alpha/r) * BA`. Effectively controls the "strength" of the adapter. A common rule: set alpha = 2×r.
> - **target_modules**: which weight matrices to apply LoRA to. At minimum: `q_proj, v_proj`. For more capacity: add `k_proj, o_proj, gate_proj, up_proj, down_proj`.

---

### Q14. What is RLHF? Walk me through the 3 stages.

> **Stage 1 — SFT**: Fine-tune the base model on high-quality human-written demonstrations. Teaches basic instruction following.
>
> **Stage 2 — Reward Model**: Collect pairs of model outputs for the same prompt, have humans rank them. Train a reward model (usually a smaller LLM with a regression head) to predict human preference scores.
>
> **Stage 3 — PPO**: Use RL (Proximal Policy Optimization) to optimize the SFT model to maximize the reward model's score. Add a KL divergence penalty to prevent the model from drifting too far from the SFT model (avoids reward hacking).
>
> This is how ChatGPT, GPT-4, and Claude were aligned to be helpful and safe.

---

### Q15. What is DPO and why is it replacing RLHF?

> DPO (Direct Preference Optimization) achieves the same alignment goal as RLHF but without training a separate reward model and without RL.
>
> It directly optimizes the LLM on (chosen, rejected) response pairs using a simple classification loss:
> ```
> L = -log σ(β * log(π_chosen/π_ref) - β * log(π_rejected/π_ref))
> ```
>
> Why it's better in practice: simpler pipeline, no reward model to maintain, more stable training, comparable or better results. Most modern fine-tuned models (Mistral, LLaMA fine-tunes) now use DPO.

---

### Q16. What is instruction tuning?

> Fine-tuning a base LLM on a large set of (instruction, response) pairs formatted in natural language. This teaches the model to follow user instructions, not just complete text.
>
> Before instruction tuning: "Summarize this article:" → model might just continue the sentence.
> After instruction tuning: model actually summarizes the article.
>
> Key datasets: FLAN (1000+ tasks), Alpaca (52K GPT-4 generated), ShareGPT (real conversations). Quality beats quantity — LIMA showed 1000 carefully chosen examples can match SFT on much larger datasets.

---

### Q17. When would you choose fine-tuning over RAG?

> Choose **fine-tuning** when:
> - You need to change the model's **behavior, style, or reasoning patterns**
> - You want to teach **domain-specific knowledge** that won't change frequently
> - You need **latency improvements** (no retrieval overhead)
> - The task has a specific format the base model doesn't follow well
>
> Choose **RAG** when:
> - Knowledge **updates frequently** (news, product info, live data)
> - Data is **too large** to bake into weights
> - You need **source attribution** (cite which document answered the question)
> - You don't have training compute budget
>
> **Best in production**: combine both — fine-tune for behavior, RAG for knowledge.

---

### Q18. What is catastrophic forgetting in fine-tuning?

> When you fine-tune a model on a new task, it can "forget" capabilities learned during pre-training. The model updates weights to minimize loss on the new task, overwriting general knowledge.
>
> Mitigations:
> - **LoRA**: adapters don't touch base weights — no forgetting possible
> - **Low learning rate**: small updates preserve original knowledge
> - **Replay**: mix original pre-training data with fine-tuning data
> - **EWC** (Elastic Weight Consolidation): penalize changes to important weights

---

## SECTION 3: RAG (Retrieval Augmented Generation)

---

### Q19. What is RAG? Why is it used?

> RAG (Retrieval Augmented Generation) augments an LLM with an external retrieval system. Instead of relying on what the model memorized during training, we retrieve relevant documents at query time and inject them into the context.
>
> Why: LLMs have static knowledge (training cutoff), hallucinate when they don't know something, and can't access private/internal documents. RAG fixes all three without retraining.
>
> In production: used in enterprise Q&A systems, documentation search, customer support, knowledge bases.

---

### Q20. Walk me through a RAG pipeline from scratch.

> **Indexing (one-time):**
> 1. Load documents (PDFs, Confluence, etc.)
> 2. Chunk into ~512-token segments with overlap
> 3. Embed each chunk using an embedding model (BAAI/bge-large)
> 4. Store vectors in a vector DB (Pinecone, Qdrant, FAISS)
>
> **Query time:**
> 5. Embed the user query with the same embedding model
> 6. Run ANN search in vector DB → retrieve top-K chunks (k=5)
> 7. (Optional) Re-rank with cross-encoder for better precision
> 8. Build prompt: system prompt + retrieved chunks + user query
> 9. Send to LLM → generate grounded answer

---

### Q21. What is the difference between dense retrieval and sparse retrieval?

> **Dense (semantic)**: Use neural embeddings. Similar meaning → similar vectors → found via cosine similarity. Handles paraphrases, synonyms.
> `"car accident" matches "vehicle collision"` ✓
>
> **Sparse (BM25/TF-IDF)**: Keyword frequency-based. Must match exact terms.
> `"car accident" misses "vehicle collision"` ✗
>
> **Hybrid**: Combine both scores. Best in production.
> ```
> final_score = α * dense_score + (1-α) * bm25_score
> ```
> Hybrid catches both semantic and exact-match cases. Most production RAG systems use hybrid.

---

### Q22. What is re-ranking in RAG and why does it matter?

> Vector search (bi-encoder) is fast but approximate — it embeds query and docs independently and compares. A cross-encoder jointly processes (query, doc) pairs for a much more accurate relevance score.
>
> **Two-stage retrieval:**
> 1. Retrieve top-50 with fast bi-encoder (vector search)
> 2. Re-rank top-50 with cross-encoder → take top-5
> 3. Send top-5 to LLM
>
> This is the production pattern. Direct cross-encoder search over all docs is too slow; direct vector search for top-5 is imprecise. Two-stage gives you speed + accuracy.

---

### Q23. How do you evaluate a RAG system?

> Use **RAGAS** framework with 4 metrics:
>
> - **Context Precision**: Of the retrieved chunks, what fraction are actually relevant?
> - **Context Recall**: Were all relevant chunks retrieved?
> - **Faithfulness**: Does the answer stay grounded in the retrieved context? (No hallucination)
> - **Answer Relevance**: Does the answer actually address the question?
>
> In production, also monitor: retrieval latency, end-to-end latency, and user feedback signals (thumbs up/down, rephrases).

---

### Q24. What chunk size do you use and why?

> Depends on the use case, but typical starting point: **512 tokens with 50-token overlap**.
>
> - Too small (< 128): lacks context, LLM can't answer from the chunk alone
> - Too large (> 1024): retrieves too much noise, hits context window limits
>
> In production we often use **parent-child chunking**: index small chunks (128 tokens) for precise retrieval, but return the parent chunk (512 tokens) to the LLM for full context.

---

## SECTION 4: GENERATIVE AI & LLMs

---

### Q25. What is temperature in LLM generation?

> Temperature scales the logits before softmax: `probs = softmax(logits / T)`
>
> - `T < 1`: sharpens the distribution → model picks the most likely token more often → focused, repetitive
> - `T = 1`: unchanged distribution
> - `T > 1`: flattens distribution → more random, diverse, creative
>
> **Production settings**: factual QA → T=0.1-0.3. Conversational → T=0.7. Creative writing → T=1.0-1.2.

---

### Q26. What is the difference between top-K and top-P sampling?

> **Top-K**: always sample from exactly the K most probable tokens. Problem: K=50 might include very low-probability tokens when the model is confident, or exclude many good tokens when it's uncertain.
>
> **Top-P (nucleus sampling)**: sample from the smallest set of tokens whose cumulative probability exceeds P. If model is confident, this might be 2 tokens; if uncertain, 50 tokens. Adapts to the actual distribution.
>
> **In production**: use top-P=0.9 (standard). Optionally combine with top-K=50 as a safety cap.

---

### Q27. What is hallucination and how do you mitigate it in production?

> Hallucination: LLM generates factually incorrect but confident-sounding text. Happens because LLMs optimize for fluency and coherence, not factual accuracy.
>
> **Production mitigations (in order of effectiveness):**
> 1. **RAG**: ground responses in retrieved documents — if it's not in the context, don't say it
> 2. **Low temperature (0.1-0.3)** for factual tasks
> 3. **Self-consistency**: sample 5 outputs, take majority vote
> 4. **LLM-as-judge**: use GPT-4 to verify factual claims
> 5. **Prompt engineering**: "Only use information from the provided context. If you don't know, say so."
> 6. **Fine-tune with factual preference data** (DPO on accurate vs hallucinated pairs)

---

### Q28. What is the context window and why does it matter?

> The maximum number of tokens a model can process in one forward pass. Everything outside the context window is invisible to the model.
>
> **Why it matters in production:**
> - Long documents must be chunked (RAG)
> - Long conversations need summarization or compression
> - KV cache grows with context — GPU memory bottleneck
> - Inference cost scales quadratically with sequence length (standard attention)
>
> **Current limits**: GPT-4 (128K), Claude 3 (200K), LLaMA 3.1 (128K), Gemini 1.5 (1M).

---

### Q29. What is Chain-of-Thought (CoT) prompting?

> Prompting the model to generate **intermediate reasoning steps** before the final answer.
>
> ```
> Without CoT: "Q: 23 × 47 = ? A: 1061"  (likely wrong)
> With CoT:    "Q: 23 × 47 = ? Let's think step by step.
>              A: 23 × 40 = 920, 23 × 7 = 161, 920 + 161 = 1081"
> ```
>
> Why it works: forces the model to decompose the problem, each step conditions the next, reduces errors.
>
> **In production**: useful for complex reasoning, multi-step math, code debugging. Not needed for simple factual lookup. Increases output tokens (latency + cost).

---

### Q30. What is RLHF and why did it transform LLMs?

> Before RLHF (2022): LLMs were good at text completion but bad at following instructions, often harmful or unhelpful.
>
> RLHF (InstructGPT, 2022) fine-tuned GPT-3 to **optimize for what humans actually want**. Human raters ranked outputs, a reward model learned their preferences, PPO optimized the LLM to maximize that reward.
>
> Result: ChatGPT. The gap between GPT-3 (confusing, inconsistent) and ChatGPT (helpful, coherent) is almost entirely RLHF. It transformed LLMs from academic curiosities into products.

---

## SECTION 5: MODEL OPTIMIZATION

---

### Q31. What is quantization and what are the trade-offs?

> Quantization reduces numerical precision of weights:
> ```
> FP32 (4 bytes) → FP16 (2 bytes) → INT8 (1 byte) → INT4 (0.5 bytes)
> ```
>
> A 7B model goes from 28GB → 14GB → 7GB → 3.5GB.
>
> **Trade-offs:**
> - FP16: near-zero quality loss, 2× memory saving — always use this
> - INT8: minor quality loss (<1%), 4× saving — production-safe
> - INT4 (GPTQ/AWQ): noticeable quality loss on hard tasks, 8× saving — good for consumer deployment
>
> **In production**: INT8 for most models, INT4 with AWQ for resource-constrained deployments.

---

### Q32. What is the difference between GPTQ and AWQ?

> Both are 4-bit post-training quantization methods.
>
> **GPTQ**: minimizes quantization error layer by layer using second-order (Hessian) information. Works well but can miss important weights.
>
> **AWQ** (Activation-Aware Weight Quantization): identifies which weights are **most important** based on activation patterns, and protects them from quantization errors. Result: better quality at 4-bit.
>
> **In production**: prefer AWQ for quality. GPTQ when AWQ isn't available for your model.

---

### Q33. What is knowledge distillation?

> Train a small **student** model to mimic the outputs of a large **teacher** model. The student learns from the teacher's full probability distribution (soft labels), not just the ground truth label.
>
> Why soft labels are better: `[0, 0.02, 0.95, 0.03]` is more informative than `[0, 0, 1, 0]` — it tells you "cat is most likely, but maybe kitten."
>
> **Production example**: DistilBERT is 40% smaller and 60% faster than BERT with 97% of its performance. Phi-3 mini (3.8B) performs like GPT-3.5 because it was distilled on GPT-4 outputs.

---

### Q34. What is speculative decoding?

> Use a small, fast **draft model** to generate multiple candidate tokens, then verify them all with the large model in **one forward pass**.
>
> ```
> Draft (fast): generates [" The", " cat", " sat", " on"]
> Large model: verifies all 4 in parallel
> Accepts: [" The", " cat"]  → rejects " sat" → resamples from here
> ```
>
> Result: same quality as the large model alone, but **2-3× faster**. The large model verifies in parallel what would have taken 4 serial steps.
>
> Used in production at Google, Meta. The draft model must share vocabulary with the large model.

---

### Q35. What is ZeRO and what problem does it solve?

> In standard data parallelism, every GPU stores the full model + gradients + optimizer states.
> For Adam: model(2) + grad(4) + optimizer states(8) ≈ 18 bytes/param → 7B model = 126GB per GPU.
>
> **ZeRO (DeepSpeed) partitions these across GPUs:**
> - ZeRO-1: shard optimizer states → 4× memory reduction
> - ZeRO-2: shard optimizer states + gradients → 8× reduction
> - ZeRO-3: shard all three → N× reduction (N = num GPUs)
>
> With 8 GPUs and ZeRO-3: 126GB → ~16GB per GPU. Now you can train a 7B model on 8× 24GB GPUs.

---

## SECTION 6: DEPLOYMENT & PRODUCTION SYSTEMS

---

### Q36. How would you deploy a 7B LLM to serve 1000 concurrent users?

> **Architecture:**
> 1. **Model**: 7B in INT8 (7GB) or 4-bit (3.5GB) — fits on 1 A100 40GB with room for KV cache
> 2. **Serving**: vLLM with PagedAttention + continuous batching — maximizes GPU utilization
> 3. **Scaling**: multiple vLLM instances behind NGINX/K8s load balancer
> 4. **Cache**: Redis for repeated prompts (semantic cache)
> 5. **API**: FastAPI with async endpoints + streaming (SSE)
> 6. **Monitoring**: Prometheus for TTFT, throughput, GPU util
>
> **Scaling math**: vLLM on 1 A100 handles ~100-200 concurrent users for typical chat workloads. For 1000 users: 5-10 A100 instances.

---

### Q37. What is PagedAttention and why is vLLM so much faster?

> KV cache memory for a single sequence is allocated upfront in contiguous GPU memory. Problem: different requests have different lengths → internal fragmentation → 60-80% of GPU memory wasted.
>
> **PagedAttention** (inspired by OS virtual memory) stores KV cache in non-contiguous **pages** (like memory pages in an OS). Pages are allocated on demand and freed when a sequence completes.
>
> Result: near-zero memory waste → more concurrent requests on same hardware → higher throughput.
>
> Combined with **continuous batching** (insert new requests as slots open), vLLM achieves 20-100× higher throughput than naive HuggingFace generation.

---

### Q38. What is continuous batching?

> Traditional batching: wait for all requests in a batch to finish → then start next batch. GPU idles waiting for slow requests.
>
> **Continuous batching**: as soon as any request in the batch completes its generation, immediately insert a new request into that slot. No idle time.
>
> Real-world impact: if 1 request generates 10 tokens and another generates 100 tokens, you start a new request after 10 tokens instead of waiting for 100. GPU utilization goes from 50-60% to 90%+.

---

### Q39. What is TTFT and why is it important?

> **TTFT (Time to First Token)**: the delay between the user submitting a request and seeing the first token of the response.
>
> This is the **main driver of perceived responsiveness**. Users are much more tolerant of slow generation if they see output immediately.
>
> That's why streaming (SSE/WebSocket) is essential — send tokens as generated, don't wait for full response.
>
> **Production targets**: TTFT < 500ms for interactive chat. TPS (tokens/second) > 30 for smooth streaming.

---

### Q40. How do you monitor an LLM in production?

> **Key metrics to track:**
>
> | Metric | What it tells you |
> |--------|------------------|
> | TTFT | User-perceived latency |
> | Token throughput (TPS) | Serving efficiency |
> | GPU utilization | Are you using hardware well? |
> | Request queue depth | Scaling trigger |
> | Error rate | Guardrails / model issues |
> | Hallucination rate | Output quality (LLM-as-judge) |
>
> **Tools**: Prometheus + Grafana for metrics. Langfuse or Arize for LLM-specific observability (trace prompts, responses, latency per call). Alert on P99 latency > 5s or GPU util < 60%.

---

## SECTION 7: DISTRIBUTED TRAINING

---

### Q41. What is the difference between data parallelism and model parallelism?

> **Data parallelism (DDP)**: Every GPU has the full model. Each GPU trains on a different batch. Gradients are averaged (AllReduce) after each step. Scales batch size, not model size.
>
> **Model parallelism**: The model itself is split across GPUs. Necessary when the model doesn't fit on one GPU.
> - *Tensor parallelism*: split weight matrices horizontally (within a layer)
> - *Pipeline parallelism*: split layers vertically (different GPUs hold different layers)
>
> **In practice**: use data parallelism + ZeRO for most fine-tuning. Use 3D parallelism (DP + TP + PP) for training 100B+ parameter models from scratch.

---

### Q42. What is gradient checkpointing and when do you use it?

> During the backward pass, PyTorch needs the intermediate activations computed in the forward pass. For a 32-layer model, that's 32 layers × batch × seq_len × d_model of memory — can easily be 40-60GB.
>
> **Gradient checkpointing**: only save activations at every N-th layer (checkpoints). During backward pass, recompute the missing activations on the fly.
>
> Trade-off: ~33% more compute, but ~√N memory reduction.
>
> **When to use**: always for fine-tuning large models. `model.gradient_checkpointing_enable()` in HuggingFace — one line, big memory saving.

---

### Q43. What is gradient accumulation?

> Instead of doing an optimizer step after every batch, accumulate gradients across N batches, then do one step.
>
> `effective_batch_size = per_device_batch × num_gpus × accumulation_steps`
>
> Why: a batch size of 1024 trains better than 8 (more stable gradients, better generalization). But you can't fit 1024 samples in GPU memory. With `accumulation_steps=128` and `per_device_batch=8`, you get effective batch of 1024 with 8 samples in memory at once.

---

### Q44. What is FSDP?

> PyTorch's native equivalent of DeepSpeed ZeRO-3. Fully Sharded Data Parallel — shards model weights, gradients, AND optimizer states across all GPUs.
>
> Each GPU only holds `1/N` of the model in memory. Before each forward pass, AllGather to reconstruct the full layer. After backward, ReduceScatter to keep only the local gradient shard.
>
> Use FSDP when: you want ZeRO-3 behavior without the DeepSpeed dependency. HuggingFace Trainer + Accelerate support it natively.

---

## SECTION 8: NLP & DEEP LEARNING

---

### Q45. What is the vanishing gradient problem and how do Transformers solve it?

> In deep networks, gradients become exponentially small as they propagate backward through many layers. Early layers receive near-zero gradients → learn very slowly or not at all.
>
> **Transformers solve this with residual connections**:
> ```
> output = LayerNorm(x + sublayer(x))
> ```
> The `+x` creates a "highway" for gradients to flow directly through — gradient of the loss reaches early layers without passing through all the multiplications.
>
> This is why Transformers can be trained with 96+ layers while RNNs struggled beyond 4-8.

---

### Q46. Why is AdamW the default optimizer for LLMs?

> **Adam**: adaptive learning rates per parameter (uses first and second moment estimates). Converges faster than SGD for transformer training.
>
> **AdamW**: Adam + **decoupled weight decay**. Regular Adam applies weight decay to the gradient update (incorrect — it mixes with the adaptive moment). AdamW applies weight decay directly to weights (correct L2 regularization).
>
> Result: better generalization, less overfitting. Every LLM (GPT, LLaMA, BERT) uses AdamW or a variant.

---

### Q47. What is transfer learning and why does it work?

> Pre-train a model on massive data → it learns general language representations. Fine-tune on a small task-specific dataset → adapt those representations to the task.
>
> Why it works: lower layers learn general syntax and semantics (universally useful), higher layers learn task-specific patterns (need to fine-tune). The pre-trained foundation is already excellent — you just adapt the top.
>
> In practice: a 1000-example fine-tuned LLaMA outperforms a 1M-example model trained from scratch on that task. The pre-training does the heavy lifting.

---

### Q48. What is the difference between BPE and WordPiece tokenization?

> Both are subword tokenization algorithms — they split rare words into subwords.
>
> **BPE (Byte Pair Encoding)**: iteratively merges the most frequent byte/character pair. Used by GPT-2, LLaMA.
>
> **WordPiece**: similar but uses maximum likelihood instead of frequency for merging. Produces slightly different vocabularies. Used by BERT. Marks non-first subwords with `##`:
> ```
> "tokenization" → ["token", "##ization"]
> ```
>
> Both handle out-of-vocabulary words gracefully by falling back to character-level tokens. In practice, the difference in quality is minimal — the vocabulary size (32K vs 128K) matters more.

---

## SECTION 9: SYSTEM DESIGN

---

### Q49. Design a production RAG system for a 10,000-page company knowledge base.

> **Indexing Pipeline:**
> - Parse PDFs/Confluence/Notion with LlamaParse or Apache Tika
> - Chunk into 512-token segments with 50-token overlap (parent-child indexing)
> - Embed with `BAAI/bge-large-en-v1.5` (best open-source English embeddings)
> - Store in Qdrant (self-hosted) or Pinecone (managed)
> - Also index with BM25 for hybrid retrieval
>
> **Query Pipeline:**
> - Rewrite query (LLM-based query expansion for better recall)
> - Hybrid search: dense + BM25 → top-50 results
> - Re-rank with `cross-encoder/ms-marco-MiniLM-L-6-v2` → top-5
> - Feed into LLM (GPT-4 or fine-tuned LLaMA) with system prompt enforcing grounded answers
>
> **Infra:**
> - FastAPI + async for API layer
> - Redis cache for repeated queries
> - Langfuse for tracing (log prompts, retrieved chunks, answers)
> - RAGAS eval pipeline for weekly quality checks
> - Kubernetes for deployment, auto-scale on request queue depth

---

### Q50. How do you decide between using GPT-4 API vs self-hosting an open-source LLM?

> **Choose GPT-4 API when:**
> - You need maximum quality with zero infra work
> - Traffic is unpredictable (pay-per-token scales naturally)
> - Data privacy is manageable (data goes to OpenAI)
> - Time to market is critical
>
> **Choose self-hosted (LLaMA, Mistral) when:**
> - Data privacy/compliance (healthcare, finance, legal)
> - High volume (API costs become significant > $10K/month)
> - Custom fine-tuning on proprietary data
> - Latency requirements (no network round-trip to OpenAI)
>
> **In practice at most companies**: GPT-4 API for prototyping and low-volume, self-hosted fine-tuned models for high-volume production workloads.

---

## BONUS: 5 Production Stories to Tell

These make you sound like you've done this in production:

### Story 1: Optimizing Inference Latency
> "We had a RAG endpoint with 3-second latency. Profiled it — 80% was LLM inference. Switched to vLLM from Transformers pipeline, got 4× throughput improvement. Then added INT8 quantization which cut GPU memory in half, allowing us to fit 2 model replicas on the same server."

### Story 2: RAG Quality Issue
> "Our RAG system had good recall but users complained answers were wrong. Used RAGAS to diagnose — faithfulness was 0.6 (LLM was ignoring retrieved context). Fixed by modifying the system prompt to explicitly say 'base your answer ONLY on the provided context' and lowering temperature to 0.2."

### Story 3: Fine-tuning with LoRA
> "We needed the model to output a specific JSON schema consistently. Collected 500 examples of correct JSON outputs. Fine-tuned with QLoRA (r=16, alpha=32) for 3 epochs on a single A100. Went from 60% valid JSON to 98% without touching the base model."

### Story 4: Cost Reduction
> "Production LLM costs were $50K/month on GPT-4. Analyzed query logs — 70% were simple retrieval questions that didn't need GPT-4 quality. Built a router: fast queries go to Mistral 7B (self-hosted), complex ones to GPT-4. Cut costs to $15K/month."

### Story 5: Distributed Training
> "Fine-tuning a 13B model for our domain. Single A100 ran OOM. Used DeepSpeed ZeRO-2 across 4× A100s with gradient checkpointing enabled. Training memory dropped from 104GB per GPU to 26GB per GPU. Run completed in 6 hours."
