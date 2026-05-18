# Generative AI Engineer — Interview Preparation Roadmap

## Job Target
- Role: Generative AI Engineer (LLMs)
- Experience: 2–4 Years
- Key Skills: Generative AI, LLM, RAG, Fine-tuning, Deep Learning, Transformers

---

## Learning Files Index

| # | Topic | Primary Files | Status |
|---|-------|---------------|--------|
| 01 | Python & Core ML Libraries | [practical/01_python_libraries/numpy_pandas_sklearn_production.md](practical/01_python_libraries/numpy_pandas_sklearn_production.md) | ⬜ |
| 02 | Deep Learning Fundamentals | [practical/02_deep_learning/deep_learning_fundamentals.md](practical/02_deep_learning/deep_learning_fundamentals.md) | ⬜ |
| 03 | ML Frameworks (PyTorch, HF) | [practical/03_ml_frameworks/pytorch_tensorflow_huggingface.md](practical/03_ml_frameworks/pytorch_tensorflow_huggingface.md) | ⬜ |
| 04 | Transformers & Attention Mechanisms | [practical/04_transformers_attention/attention_is_all_you_need_paper.md](practical/04_transformers_attention/attention_is_all_you_need_paper.md) <br> [practical/04_transformers_attention/transformers_core_concepts.md](practical/04_transformers_attention/transformers_core_concepts.md) <br> [practical/04_transformers_attention/modern_transformer_improvements.md](practical/04_transformers_attention/modern_transformer_improvements.md) <br> [practical/04_transformers_attention/mamba_state_space_models.md](practical/04_transformers_attention/mamba_state_space_models.md) | ⬜ |
| 05 | LLM Models (GPT, BERT, o1/R1) | [practical/05_llm_models/gpt_bert_t5_architecture.md](practical/05_llm_models/gpt_bert_t5_architecture.md) <br> [practical/05_llm_models/open_source_llms_llama_mistral_falcon.md](practical/05_llm_models/open_source_llms_llama_mistral_falcon.md) <br> [practical/05_llm_models/reasoning_models_multimodal_scaling_laws.md](practical/05_llm_models/reasoning_models_multimodal_scaling_laws.md) | ⬜ |
| 06 | NLP Techniques & Tokenization | [practical/06_nlp_techniques/tokenization_bpe_wordpiece_sentencepiece.md](practical/06_nlp_techniques/tokenization_bpe_wordpiece_sentencepiece.md) <br> [practical/06_nlp_techniques/transfer_learning_fewshot_embeddings.md](practical/06_nlp_techniques/transfer_learning_fewshot_embeddings.md) | ⬜ |
| 07 | SFT, Fine-Tuning & RLHF | [practical/07_fine_tuning/sft_lora_qlora_rlhf.md](practical/07_fine_tuning/sft_lora_qlora_rlhf.md) <br> [practical/07_fine_tuning/advanced_peft_dpo_grpo_safety.md](practical/07_fine_tuning/advanced_peft_dpo_grpo_safety.md) | ⬜ |
| 08 | Retrieval Augmented Generation (RAG) | [practical/08_rag/rag_fundamentals.md](practical/08_rag/rag_fundamentals.md) <br> [practical/08_rag/advanced_rag_raptor_graphrag_selfrag.md](practical/08_rag/advanced_rag_raptor_graphrag_selfrag.md) <br> [practical/08_rag/vector_db_internals_and_evaluation.md](practical/08_rag/vector_db_internals_and_evaluation.md) | ⬜ |
| 09 | Generative AI (Agents, Sampling) | [practical/09_generative_ai/text_generation_sampling_hallucination.md](practical/09_generative_ai/text_generation_sampling_hallucination.md) <br> [practical/09_generative_ai/agents_function_calling_prompting.md](practical/09_generative_ai/agents_function_calling_prompting.md) | ⬜ |
| 10 | Model Optimization (Quantization) | [practical/10_model_optimization/quantization_pruning_distillation.md](practical/10_model_optimization/quantization_pruning_distillation.md) | ⬜ |
| 11 | Distributed Training & ZeRO | [practical/11_distributed_training/distributed_training_zero_fsdp_parallelism.md](practical/11_distributed_training/distributed_training_zero_fsdp_parallelism.md) | ⬜ |
| 12 | Production Deployment & Serving | [practical/12_deployment/production_deployment_vllm_serving.md](practical/12_deployment/production_deployment_vllm_serving.md) | ⬜ |
| 13 | Cloud Platforms & LLM Scaling | [practical/13_cloud/aws_gcp_azure_for_llms.md](practical/13_cloud/aws_gcp_azure_for_llms.md) | ⬜ |

---

## Study Order (Recommended)

### Phase 1 — Foundation & Frameworks (Week 1)
1. Python & Core ML Libraries (`01_python_libraries`)
2. Deep Learning Fundamentals (`02_deep_learning`)
3. ML Frameworks (PyTorch, TensorFlow, HuggingFace) (`03_ml_frameworks`)

### Phase 2 — Transformers & LLM Architectures (Week 2)
4. Transformers & Attention Mechanisms (`04_transformers_attention`)
5. LLM Models — GPT, BERT, o1/R1, scaling laws (`05_llm_models`)
6. NLP Techniques & Tokenization (`06_nlp_techniques`)

### Phase 3 — Training, Fine-Tuning & RAG (Week 3)
7. Training & Fine-Tuning (SFT, LoRA, RLHF, DPO) (`07_fine_tuning`)
8. Retrieval Augmented Generation (RAG) (`08_rag`)
9. Generative AI (Autoregressive sampling, Agents, CoT) (`09_generative_ai`)

### Phase 4 — Optimization, Scaling & Deployment (Week 4)
10. Model Optimization (Quantization, Pruning) (`10_model_optimization`)
11. Distributed Training & Parallelism (`11_distributed_training`)
12. Model Deployment at Scale (vLLM, TGI, serving) (`12_deployment`)
13. Cloud Platforms & Cost Optimization (`13_cloud`)

---

## Interview Topics Checklist

### Generative AI
- [ ] What is Generative AI? How does it differ from discriminative AI?
- [ ] Explain the GPT architecture
- [ ] What is RLHF?
- [ ] How does temperature/top-k/top-p sampling work?

### LLMs
- [ ] Explain GPT vs BERT vs T5 differences
- [ ] What is a context window?
- [ ] What are embeddings?
- [ ] What is tokenization?

### RAG
- [ ] What is RAG and why is it used?
- [ ] Explain vector databases
- [ ] What is semantic search?
- [ ] RAG vs Fine-tuning — when to use which?

### Fine-Tuning
- [ ] What is fine-tuning?
- [ ] Explain LoRA and PEFT
- [ ] What is instruction tuning?
- [ ] What datasets are used for fine-tuning?

### Deep Learning
- [ ] Backpropagation explained
- [ ] Optimizers — Adam, SGD, AdaFactor
- [ ] Loss functions in NLP
- [ ] Regularization techniques

### Deployment
- [ ] How do you serve an LLM in production?
- [ ] What is model quantization?
- [ ] What is vLLM / TGI?
- [ ] How do you handle latency and throughput?
