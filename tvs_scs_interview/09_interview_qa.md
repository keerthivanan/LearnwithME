# Interview Q&A — Senior GenAI & Agentic AI (70+)

Practice out loud. Grouped by pillar. Answers are concise — expand with your projects.

---

## 🤖 AGENTIC AI (18)

**1. What is an AI agent?** LLM + tools + a reason-act loop; it decides which tools to call, observes results, and iterates to a goal.
**2. Explain ReAct.** Reason + Act: the model alternates reasoning and tool actions, using each observation to inform the next step.
**3. CoT vs ReAct?** CoT reasons internally before answering; ReAct adds tool actions between reasoning steps.
**4. What is function/tool calling?** The LLM outputs a structured request (name + JSON args) to call a function; the framework runs it and returns the result. I enforce a schema so args are valid.
**5. Types of agent memory?** Short-term (conversation window), long-term (vector store, cross-session), episodic (past events), working (in-task scratchpad).
**6. Why multi-agent over one agent?** Separation of concerns — focused prompts, scoped tools, easier testing, safer, cheaper (route to smaller models).
**7. Multi-agent patterns?** Router→specialists, sequential pipeline, hierarchical/supervisor, debate.
**8. What is HITL and when?** Human-in-the-loop — pause for human approval at high-stakes points (refunds, sending). The agent resumes after.
**9. LangChain vs LangGraph?** LangChain = components (chains, tools, memory). LangGraph = graph/state-machine for stateful, cyclic, branching agent workflows with persistence and HITL — better for reliable multi-agent.
**10. When LlamaIndex?** When RAG/data-indexing is the core — strong ingestion, indexing, query engines.
**11. AutoGen vs CrewAI?** AutoGen = conversational multi-agent with code execution; CrewAI = role/goal/task-based agent teams, very readable.
**12. How do you stop runaway agents?** Max iterations, loop detection, timeouts, cost caps.
**13. How to handle a tool failing?** Timeout + retry with backoff, fallback, and return a graceful message; trace it.
**14. Long-horizon tasks?** Planning + persistent state/memory + checkpointing (LangGraph); break into steps.
**15. Self-correction?** The agent (or a validator) checks its output and retries with feedback (reflexion).
**16. Structured output enforcement?** Use function-calling/JSON schema + Pydantic validation; retry on invalid.
**17. A hard agent bug you fixed?** (Your story) Tool returned "not found" for valid IDs; traces showed the agent passed the arg under a different key; fixed input reading and applied it to all tools.
**18. How do you test agents?** Eval sets of tasks, assert tool calls + final outcome, trace review, regression evals in CI.

---

## 📚 RAG (16)

**19. What is RAG and why?** Retrieve relevant docs into the prompt so the LLM answers from your data — current, private, grounded, no retraining.
**20. RAG vs fine-tuning?** RAG for knowledge/facts (easy to update, cite); fine-tuning for behavior/style/format.
**21. The pipeline?** Chunk→embed→store; query→(rewrite/HyDE)→retrieve (hybrid)→re-rank→grounded prompt→answer.
**22. Chunking strategies?** Fixed-size with overlap, recursive, semantic, contextual (prepend section context), structure-aware.
**23. Chunk size trade-off?** Too big dilutes embeddings/wastes context; too small loses meaning. Overlap preserves continuity.
**24. What are embeddings?** Vectors capturing meaning; similar text → nearby vectors (cosine). Query and docs must use the same model.
**25. Vector DBs you know?** FAISS, Chroma, Pinecone, Weaviate, Qdrant, Milvus, pgvector, Elasticsearch.
**26. What's HNSW?** A graph-based ANN index for fast approximate nearest-neighbor search; recall/speed trade-off.
**27. Hybrid search?** Combine dense (semantic) + sparse (BM25 keyword); fuse with RRF — catches meaning AND exact terms.
**28. Re-ranking?** Retrieve top-50 with a bi-encoder, then a cross-encoder scores query+chunk jointly → keep top-5. Big precision gain.
**29. Bi- vs cross-encoder?** Bi embeds separately (fast, retrieval); cross scores jointly (accurate, re-ranking).
**30. What is HyDE?** LLM generates a hypothetical answer, embed that, retrieve with it — often matches docs better than the short query.
**31. Multi-LLM routing?** Route by cost/quality (mini for simple, frontier for hard) with fallback if a provider fails.
**32. How do you evaluate RAG?** Separately: retrieval (context precision/recall, hit rate) and generation (faithfulness, answer relevance) — RAGAS/LangSmith.
**33. RAG hallucinates despite context — fix?** Strong grounding instruction, re-ranking for better context, cite sources, refuse if unsupported.
**34. Vector index lifecycle?** Re-embed/re-ingest on data or embedding-model changes; handle updates/deletes; version the index.

---

## ⚙️ MLOps / Deployment / Fine-tuning (14)

**35. Docker vs Kubernetes?** Docker packages one app into a container; K8s runs/scales/heals many containers with rolling updates + rollback.
**36. CI/CD for ML?** Automate build→eval→deploy on commit (GitHub Actions/ArgoCD); run evals as gates; keep rollback to last-good model.
**37. MLflow / W&B / DVC?** Experiment tracking + model registry/versioning (MLflow, W&B) and data versioning (DVC) for reproducibility.
**38. LoRA?** Add small trainable low-rank adapters, freeze the base → cheap, fast, swappable fine-tuning (PEFT).
**39. QLoRA?** LoRA on a 4-bit quantized base → fine-tune large models on a single GPU.
**40. SFT vs RLHF vs DPO?** SFT = train on input→output pairs. RLHF = reward model + RL to align to human preferences. DPO = align to preference pairs directly, no reward model (simpler).
**41. When fine-tune vs RAG vs prompt?** Prompt first, RAG for knowledge, fine-tune for behavior/format the model can't do via prompting.
**42. What is vLLM?** High-throughput LLM server using PagedAttention + continuous batching for efficient serving.
**43. Quantization?** Lower-precision weights (INT8/INT4) → smaller, faster, less memory, slight accuracy cost.
**44. What's Triton / TensorRT-LLM / ONNX?** NVIDIA model server / NVIDIA optimized inference engine / open export format for optimized inference.
**45. Continuous batching?** Dynamically add/remove requests in a batch → high GPU utilization (vLLM).
**46. KV cache?** Cache attention keys/values so past tokens aren't recomputed each step → faster generation.
**47. Rollback strategy?** Version models/images; if a deploy degrades metrics, revert to the previous version instantly.
**48. Cloud mapping?** SageMaker/Bedrock (AWS), ML Studio/Azure OpenAI (Azure), Vertex AI (GCP).

---

## 📊 Observability / Reliability (10)

**49. Why observability for GenAI?** LLMs fail silently and cost per token; you must trace every call and track quality/cost/latency.
**50. Metrics you track?** Token cost/request, p50/p95/p99 latency, TTFT, hallucination/faithfulness rate, tool-call traces, error rate, task success.
**51. What is p95 latency?** 95% of requests are faster than this; you optimize the slow tail (p95/p99), not just the average.
**52. LangSmith / LangFuse?** LLM tracing + evals + cost/latency; LangFuse is open-source/self-hostable.
**53. Distributed tracing?** Tie all spans of one request across services to a trace ID (OpenTelemetry) — essential for multi-agent debugging.
**54. Circuit breaker?** If a dependency keeps failing, trip and stop calling it briefly → fail fast, avoid pile-ups.
**55. Graceful degradation?** On AI failure, fall back to cache/simpler model/human handoff instead of a hard error.
**56. Retry strategy?** Exponential backoff + jitter on transient errors; timeouts; idempotency for actions.
**57. SLI/SLO/SLA?** Indicator (measured), Objective (target), Agreement (customer promise).
**58. Walk through an incident RCA.** (Your tool-bug story) symptom→trace→root cause→fix→prevent the class.

---

## 🔌 Backend (8)

**59. Why FastAPI for AI?** Async-native (great for I/O-bound LLM calls), Pydantic validation, auto docs, fast.
**60. async vs sync — when?** Async for I/O-bound (LLM/DB/API); sync/processes for CPU-bound. Async serves many concurrent waiters efficiently.
**61. FastAPI vs Flask vs Django?** FastAPI async/validation/docs for APIs; Flask lightweight; Django full-stack with ORM/admin.
**62. Streaming tokens?** WebSockets or SSE to stream tokens as generated → perceived speed.
**63. OAuth2 vs JWT?** OAuth2 = authorization framework; JWT = signed stateless token carrying claims.
**64. Rate limiting — why?** Protect the service and control LLM cost (token bucket per key/user).
**65. Celery?** Offload long jobs (batch embedding, reports) to background workers so the API stays responsive.
**66. Caching for cost?** Cache embeddings and repeated/semantically-similar prompts in Redis to skip LLM calls.

---

## 🛡️ LLM fundamentals & Safety (10)

**67. What is a context window?** Max tokens per call (in+out); manage by trimming/summarizing/retrieving only relevant.
**68. Temperature?** Randomness; low for factual/consistent, high for creative.
**69. How reduce hallucination?** Grounding, RAG, tool use, structured output, low temp, cite sources / "I don't know".
**70. What is prompt injection? Defenses?** Malicious instructions in input/data hijack the model; defend with delimiters, input filtering, least-privilege tools, output validation.
**71. Responsible AI pillars?** Fairness/bias, transparency, privacy, accountability, safety.
**72. SOC 2 / GDPR / PII?** Security controls audit / EU data-protection law / personal data to mask before sending to LLMs.
**73. How optimize LLM cost?** Right-size model, cache, trim context, cap output, route+fallback, monitor cost/request.
**74. How evaluate an LLM app?** Eval sets + LLM-as-judge + RAGAS for RAG + regression evals in CI + human sampling.
**75. Guardrails — input and output?** Input: injection/PII/topic. Output: schema, safety, groundedness, refuse-if-unsupported.
**76. Grounding?** Instruct the model to answer only from provided context and say "I don't know" if absent.

---

## 🎤 The mandatory application answers (rehearse — file 01)
- **Production system #1:** DHL multi-agent (router→5 agents→12 tools, omnichannel, voice, deployed, traced)
- **Production system #2:** Langflow multi-agent + MCP tool servers
- **Incident:** tool-argument mismatch found via traces, fixed + generalized

---

## When you don't know
> "I haven't run that at production scale, but conceptually it's [explain] and it fits [where]. I'd ramp on it quickly." Honest + reasoned beats bluffing — especially for a senior role that values judgment.
