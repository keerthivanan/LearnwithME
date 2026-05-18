# Production GenAI Engineer — Complete Learning Map
> Every file. Every topic. Perfectly organized. Nothing missing.

---

## FOLDER STRUCTURE (Follow This Order)

---

### PHASE 1 — FOUNDATION & FRAMEWORKS
> Start here. Python, Deep Learning, and fundamental frameworks.

#### 📁 01_python_libraries/
| File | What You Learn |
|------|---------------|
| `numpy_pandas_sklearn_production.md` | NumPy (vectorized ops, broadcasting, softmax), Pandas (dataset processing), Scikit-learn (metrics, baselines), GPU memory management |
| `lesson.py` | Runnable code: cosine similarity, softmax, attention shapes, dataset stats, and production patterns |

#### 📁 02_deep_learning/
| File | What You Learn |
|------|---------------|
| `deep_learning_fundamentals.md` | Neural networks, backprop, optimizers (AdamW), loss functions, regularization, embeddings |
| `lesson.py` | Implementation scripts and hands-on neural network exercises |

#### 📁 03_ml_frameworks/
| File | What You Learn |
|------|---------------|
| `pytorch_tensorflow_huggingface.md` | PyTorch training loop, autograd, HuggingFace Transformers (all Auto classes, Trainer API), Accelerate, vLLM, LangChain |
| `lesson.py` | Practical PyTorch & HuggingFace integration scripts, Tokenization, pipelines, and Trainer usage |

---

### PHASE 2 — TRANSFORMERS & LLM MODELS
> Master the architecture of modern language models.

#### 📁 04_transformers_attention/
| File | What You Learn |
|------|---------------|
| `attention_is_all_you_need_paper.md` | The full 2017 paper explained — Q/K/V, multi-head attention, positional encoding, encoder vs decoder |
| `transformers_core_concepts.md` | KV cache, Flash Attention, attention complexity, residuals, layer norm |
| `modern_transformer_improvements.md` | RoPE, FlashAttention, GQA, SwiGLU, RMSNorm, ALiBi, MoE, Speculative Decoding |
| `mamba_state_space_models.md` | Mamba, SSMs, selective scan, RWKV, Hybrid models — the Transformer alternative |
| `lesson.py` | Attention mechanism models and SSM/Transformer architectural scripting |

#### 📁 05_llm_models/
| File | What You Learn |
|------|---------------|
| `gpt_bert_t5_architecture.md` | GPT vs BERT vs T5 — architecture, training objectives, when to use each |
| `open_source_llms_llama_mistral_falcon.md` | LLaMA 1/2/3/3.1, Mistral, Mixtral, Falcon, Phi, Gemma, Qwen, DeepSeek — full comparison |
| `reasoning_models_multimodal_scaling_laws.md` | o1/o3, DeepSeek-R1, GRPO, Vision LLMs (CLIP/LLaVA), Chinchilla laws, Emergent abilities, all Benchmarks (MMLU/HumanEval/GPQA) |
| `lesson.py` | Hands-on model loader scripts and comparison tools |

---

### PHASE 3 — NLP & FINE-TUNING
> Learn tokenization, NLP basics, SFT, PEFT, and RLHF.

#### 📁 06_nlp_techniques/
| File | What You Learn |
|------|---------------|
| `tokenization_bpe_wordpiece_sentencepiece.md` | BPE algorithm step-by-step, byte-level BPE, WordPiece, SentencePiece, vocabulary size trade-offs, multilingual fertility problem, chat templates |
| `transfer_learning_fewshot_embeddings.md` | Transfer learning paradigm, zero/one/few-shot, in-context learning, word embeddings, sentence embeddings, bi-encoder vs cross-encoder |
| `lesson.py` | Tokenizer algorithm implementations and embedding similarity experiments |

#### 📁 07_fine_tuning/
| File | What You Learn |
|------|---------------|
| `sft_lora_qlora_rlhf.md` | Pre-training, SFT, LoRA math (W'=W+BA), QLoRA, RLHF pipeline, DPO, instruction tuning |
| `advanced_peft_dpo_grpo_safety.md` | ORPO, SimPO, online DPO, GRPO, IA³, DoRA, LoftQ, GaLore, Constitutional AI, RLAIF, data dedup, synthetic data |
| `lesson.py` | Fine-tuning pipelines, LoRA target config scripts, and training loops |

---

### PHASE 4 — RAG & GENERATIVE AI
> Build production-grade retrieval systems and advanced AI agents.

#### 📁 08_rag/
| File | What You Learn |
|------|---------------|
| `rag_fundamentals.md` | Full RAG pipeline, chunking, embedding models, vector DBs, dense/sparse/hybrid retrieval, reranking, LangChain RAG |
| `advanced_rag_raptor_graphrag_selfrag.md` | RAPTOR (tree indexing), GraphRAG (entity graphs, Leiden), Self-RAG (reflection tokens), CRAG (confidence + web fallback), Agentic RAG (multi-hop), Contextual Retrieval |
| `vector_db_internals_and_evaluation.md` | HNSW (graph layers, M/ef params), IVF (nlist/nprobe), Product Quantization (IVFPQ), MRR, NDCG, MAP, Hit Rate, RAGAS, full eval pipeline |
| `lesson.py` | Complete RAG pipelines, Vector DB indexing scripts, and RAGAS evaluations |

#### 📁 09_generative_ai/
| File | What You Learn |
|------|---------------|
| `text_generation_sampling_hallucination.md` | Autoregressive generation, temperature, top-k, top-p, beam search, NLP tasks, hallucination mitigation, perplexity, BLEU/ROUGE |
| `agents_function_calling_prompting.md` | Function calling (OpenAI/Anthropic format), ReAct loop, LangGraph, Tree of Thought, Self-Consistency CoT, LLM-as-judge biases, Chatbot Arena |
| `lesson.py` | Logit sampling scripts, function-calling formats, and ReAct loops |

---

### PHASE 5 — OPTIMIZATION, PARALLELISM & DEPLOYMENT
> Optimize models for extreme latency, deploy at scale, and manage cloud workloads.

#### 📁 10_model_optimization/
| File | What You Learn |
|------|---------------|
| `quantization_pruning_distillation.md` | FP32→INT4 memory savings, GPTQ, AWQ, bitsandbytes NF4, pruning (structured/unstructured), knowledge distillation, speculative decoding, ONNX, TensorRT |
| `lesson.py` | Speculative decoding simulation, basic post-training quantization pipelines |

#### 📁 11_distributed_training/
| File | What You Learn |
|------|---------------|
| `distributed_training_zero_fsdp_parallelism.md` | DDP, Tensor Parallelism, Pipeline Parallelism, ZeRO stages 1/2/3, FSDP, gradient checkpointing, gradient accumulation |
| `lesson.py` | Model sharding templates and DeepSpeed configuration examples |

#### 📁 12_deployment/
| File | What You Learn |
|------|---------------|
| `production_deployment_vllm_serving.md` | vLLM (PagedAttention, continuous batching), TGI, FastAPI + streaming, TTFT/TPS, Kubernetes GPU deployment, guardrails, monitoring |
| `lesson.py` | FastAPI streaming server endpoint, benchmarking client, guardrail filters |

#### 📁 13_cloud/
| File | What You Learn |
|------|---------------|
| `aws_gcp_azure_for_llms.md` | SageMaker, Bedrock, Vertex AI, TPUs, Azure ML, Azure OpenAI, spot instances, cost optimization |
| `lesson.py` | SageMaker deploy client scripts, Bedrock/Vertex API callers |

---

## MASTER INTERVIEW FILES (Root Level)
| File | What It Is |
|------|-----------|
| `../INTERVIEW_MASTERSHEET.md` | 50 production-level Q&A — read this on Day 7 |
| `../00_7day_plan.md` | Daily study schedule |
| `../learning.md` | Full topic checklist |

---

## 7-DAY STUDY ORDER

| Day | Folders | Focus |
|-----|---------|-------|
| **Day 1** | `01` → `03` | Python, Deep Learning foundations, PyTorch/HF frameworks |
| **Day 2** | `04` (all 4 files) | Transformers & Attention mechanisms |
| **Day 3** | `05` (all 3 files) | All LLM model families & scaling laws |
| **Day 4** | `06` + `07` | Tokenization, NLP, Fine-tuning, PEFT & RLHF/DPO |
| **Day 5** | `08` (all 3 files) | RAG systems & vector database internals |
| **Day 6** | `09` + `10` + `11` + `12` + `13` | Sampling, Agents, Optimization, Scaling, Serving & Cloud |
| **Day 7** | `INTERVIEW_MASTERSHEET.md` | Full mock drill — answer out loud |

---

## TOPIC QUICK-FIND

| Topic | File |
|-------|------|
| Attention mechanism | `04_transformers_attention/attention_is_all_you_need_paper.md` |
| LoRA explained | `07_fine_tuning/sft_lora_qlora_rlhf.md` |
| RAG pipeline | `08_rag/rag_fundamentals.md` |
| RAPTOR / GraphRAG | `08_rag/advanced_rag_raptor_graphrag_selfrag.md` |
| HNSW / IVF / PQ | `08_rag/vector_db_internals_and_evaluation.md` |
| GPT vs BERT vs T5 | `05_llm_models/gpt_bert_t5_architecture.md` |
| LLaMA / Mistral / Mixtral | `05_llm_models/open_source_llms_llama_mistral_falcon.md` |
| o1 / DeepSeek-R1 / GRPO | `05_llm_models/reasoning_models_multimodal_scaling_laws.md` |
| Mamba / SSM | `04_transformers_attention/mamba_state_space_models.md` |
| RoPE / GQA / FlashAttention | `04_transformers_attention/modern_transformer_improvements.md` |
| RLHF / DPO / ORPO | `07_fine_tuning/sft_lora_qlora_rlhf.md` + `advanced_peft_dpo_grpo_safety.md` |
| Constitutional AI | `07_fine_tuning/advanced_peft_dpo_grpo_safety.md` |
| Function calling / Agents | `09_generative_ai/agents_function_calling_prompting.md` |
| BPE / tokenization | `06_nlp_techniques/tokenization_bpe_wordpiece_sentencepiece.md` |
| Quantization | `10_model_optimization/quantization_pruning_distillation.md` |
| ZeRO / FSDP | `11_distributed_training/distributed_training_zero_fsdp_parallelism.md` |
| vLLM / production serving | `12_deployment/production_deployment_vllm_serving.md` |
| Scaling laws / benchmarks | `05_llm_models/reasoning_models_multimodal_scaling_laws.md` |
| Temperature / top-p | `09_generative_ai/text_generation_sampling_hallucination.md` |
| AWS / GCP / Azure | `13_cloud/aws_gcp_azure_for_llms.md` |
