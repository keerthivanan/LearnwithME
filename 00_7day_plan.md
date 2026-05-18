# 7-Day Production-Level Interview Prep Plan

> 1 week. Every topic from the JD. Production-level answers. No fluff.

---

## Daily Schedule (2–3 hours/day)

| Day | Focus | Files to Study | Goal |
|-----|-------|---------------|------|
| **Day 1** | Transformers + LLMs | 04, 05 | Nail architecture questions |
| **Day 2** | Fine-Tuning + LoRA + RLHF | 07 | Explain training end-to-end |
| **Day 3** | RAG (full system) | 08 | Design a RAG system live |
| **Day 4** | Optimization + Distributed Training | 10, 11 | Talk about production bottlenecks |
| **Day 5** | Deployment + Cloud | 12, 13 | Walk through a prod architecture |
| **Day 6** | Python + Deep Learning + NLP + Frameworks | 01, 02, 03, 06, 09 | Fill knowledge gaps |
| **Day 7** | INTERVIEW_MASTERSHEET | INTERVIEW_MASTERSHEET.md | Full mock drill |

---

## Day 1 — Transformers + LLMs (The Most Asked Topic)

### Must Know Cold (no thinking, instant answer)
- [ ] What is self-attention? (formula + intuition)
- [ ] What is causal masking and why GPT needs it?
- [ ] BERT vs GPT vs T5 — when to use each?
- [ ] What is a KV cache?
- [ ] What are RoPE, GQA, SwiGLU? (LLaMA improvements)
- [ ] What is tokenization? BPE vs WordPiece?

### 1-Hour Study Plan
```
30 min → Read practical/04_transformers_attention/ files (attention, core concepts, modern improvements, mamba)
15 min → Read practical/05_llm_models/ files (architectures, open-source models, reasoning models)
15 min → Close everything, say all 6 answers out loud
```

---

## Day 2 — Fine-Tuning + LoRA + RLHF

### Must Know Cold
- [ ] What is LoRA? Explain the math (W' = W + BA)
- [ ] What is QLoRA? How does it differ from LoRA?
- [ ] Full fine-tuning vs LoRA — when to choose which?
- [ ] What is RLHF? 3 stages?
- [ ] What is DPO and why is it better than RLHF?
- [ ] What is instruction tuning?

### 1-Hour Study Plan
```
40 min → Read practical/07_fine_tuning/ files (sft_lora_qlora_rlhf, advanced PEFT)
20 min → Practice explaining LoRA out loud with the formula
```

---

## Day 3 — RAG (Full System Design)

### Must Know Cold
- [ ] RAG pipeline from scratch (5 steps)
- [ ] Dense vs sparse vs hybrid retrieval
- [ ] What is a vector database? Give 3 examples.
- [ ] How do you evaluate a RAG system? (RAGAS metrics)
- [ ] RAG vs fine-tuning — when to use each?
- [ ] What is re-ranking? When to use cross-encoder?

### 1-Hour Study Plan
```
40 min → Read practical/08_rag/ files (rag_fundamentals, advanced_rag, vector_db_internals)
20 min → Draw the full RAG architecture from memory on paper
```

---

## Day 4 — Optimization + Distributed Training

### Must Know Cold
- [ ] What is quantization? FP32 → INT4 memory savings?
- [ ] GPTQ vs AWQ vs bitsandbytes?
- [ ] What is PagedAttention (vLLM)?
- [ ] What is speculative decoding?
- [ ] ZeRO stages 1, 2, 3 — what each shards?
- [ ] Data parallel vs tensor parallel vs pipeline parallel?

### 1-Hour Study Plan
```
30 min → Read practical/10_model_optimization/quantization_pruning_distillation.md
30 min → Read practical/11_distributed_training/distributed_training_zero_fsdp_parallelism.md
```

---

## Day 5 — Deployment + Cloud

### Must Know Cold
- [ ] How to serve a 7B LLM to 1000 concurrent users?
- [ ] What is continuous batching?
- [ ] TTFT, TPS, throughput — what are these?
- [ ] vLLM vs TGI — what's the difference?
- [ ] SageMaker vs Bedrock — when to use each?
- [ ] How do you scale down cost in cloud for LLMs?

### 1-Hour Study Plan
```
30 min → Read practical/12_deployment/production_deployment_vllm_serving.md
30 min → Read practical/13_cloud/aws_gcp_azure_for_llms.md
```

---

## Day 6 — Fill the Gaps

### Must Know Cold
- [ ] What is backpropagation? (1 sentence)
- [ ] Why GELU over ReLU in Transformers?
- [ ] What is AdamW and why is it used for LLMs?
- [ ] What is Chain-of-Thought prompting?
- [ ] What is hallucination and how do you fix it?
- [ ] What are BLEU, ROUGE, BERTScore?

### 1-Hour Study Plan
```
15 min → Read practical/02_deep_learning/deep_learning_fundamentals.md
15 min → Read practical/09_generative_ai/text_generation_sampling_hallucination.md & agents_function_calling_prompting.md
15 min → Read practical/06_nlp_techniques/tokenization_bpe_wordpiece_sentencepiece.md
15 min → Review Python & PyTorch libs in practical/01_python_libraries/ and practical/03_ml_frameworks/
```

---

## Day 7 — Full Drill

### Morning (90 min)
Read INTERVIEW_MASTERSHEET.md completely — all 50 questions + answers.

### Afternoon (60 min)
- Pick 10 random questions
- Answer out loud without looking
- Check your answer
- Repeat for the ones you missed

### Evening (30 min)
- Review your weak areas
- Confidence check — you know this

---

## The 5 Questions They ALWAYS Ask

1. **"Explain how transformers work"** — Spend Day 1 on this
2. **"What is RAG and how would you build one?"** — Spend Day 3 on this
3. **"Explain LoRA"** — Spend Day 2 on this
4. **"How would you deploy an LLM in production?"** — Spend Day 5 on this
5. **"What is RLHF?"** — Covered in Day 2

---

## Production-Level Answer Formula

When answering any question in the interview:

```
1. One-liner definition (5 seconds)
2. The problem it solves (why it exists)
3. How it works (the key mechanism)
4. Real-world trade-off or consideration
5. Example from production
```

Example for "What is LoRA?":
> "LoRA is a parameter-efficient fine-tuning technique. [1]
> Full fine-tuning a 70B model requires hundreds of GBs of GPU — not feasible. [2]
> LoRA adds trainable low-rank matrices B×A to frozen weights, so only 0.1% of params train. W' = W + BA. [3]
> Trade-off: slightly lower quality than full fine-tuning, but 100× cheaper. [4]
> In production we use QLoRA — LoRA on 4-bit quantized base model — fits a 7B model on a single 8GB GPU. [5]"

That's a 10/10 answer.
