# MLOps, Deployment, Fine-Tuning & Inference (⭐ Pillar)

Covers deployment, CI/CD, ML lifecycle, fine-tuning, and inference optimization. Know concepts cold; hands-on depth optional for now.

---

## PART 1 — Containerization & Orchestration

### Docker (know this hands-on — it's easy + high value)
Packages your app + dependencies into a portable **image** that runs anywhere as a **container**.
- **Dockerfile** → build → **image** → run → **container**
- Key commands: `docker build -t app .`, `docker run -p 8000:8000 app`, `docker ps`, `docker logs`
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```
> Dockerize a FastAPI app once — it makes every deployment answer credible.

### Kubernetes (K8s) — orchestrates many containers
Runs, scales, heals containers across machines.
- **Pod** (smallest unit, 1+ containers), **Deployment** (manages pod replicas + rollouts), **Service** (stable network endpoint), **Ingress** (external routing)
- **Autoscaling** (HPA), **rolling updates + rollback**, **self-healing** (restarts failed pods)
- **Helm** = package manager for K8s (templated charts)
> One-liner: "Docker packages one app; Kubernetes runs and scales many containers reliably with self-healing and rolling updates."

---

## PART 2 — CI/CD for ML
Automate build → test → deploy on every change.
- **GitHub Actions / GitLab CI / Jenkins** — pipelines triggered by commits
- **ArgoCD** — GitOps: deploys to K8s from git state
- **For ML/LLM:** test prompts/evals in the pipeline, build the image, deploy, and keep a **rollback strategy** (previous image/model version) for fast revert
- **Model rollback:** version models in a registry; if a new one degrades metrics, revert
> Senior signal: "Every change runs evals in CI; deploys are rolling with instant rollback to the last-good model/image version."

---

## PART 3 — ML Lifecycle & Experiment Tracking
- **MLflow** — experiment tracking, model registry, versioning, packaging
- **Weights & Biases (W&B)** — experiment tracking, dashboards, sweeps
- **DVC (Data Version Control)** — git-like versioning for data + models
- **Model Registry** — central store of model versions with stages (staging/production)
- **Data/Model Versioning** — reproducibility: know exactly which data + code produced a model
> "MLflow for tracking runs and the model registry, DVC for data versioning — so any result is reproducible and any model is rollback-able."

---

## PART 4 — Fine-Tuning (understand each in one line; hands-on optional)

**When to fine-tune vs RAG vs prompt:** prompt first (cheapest), RAG for knowledge, fine-tune for *behavior/style/format* or domain tone the model can't do via prompting.

| Technique | What it is |
|-----------|-----------|
| **SFT (Supervised Fine-Tuning)** | Train on labeled input→output pairs to teach a task/format |
| **Instruction tuning** | SFT on many instruction/response pairs → follows instructions better |
| **PEFT (Parameter-Efficient FT)** | Fine-tune only a small % of params (cheap) — umbrella term |
| **LoRA** | Add small trainable low-rank adapter matrices; freeze the base model. Cheap, fast, swappable |
| **QLoRA** | LoRA on a **4-bit quantized** base model → fine-tune huge models on one GPU |
| **RLHF** | Reinforcement Learning from Human Feedback: train a reward model from human preferences, optimize with PPO → align to preferences |
| **DPO (Direct Preference Optimization)** | Aligns to preference pairs directly, **no separate reward model/RL** — simpler, popular RLHF alternative |
| **Distillation** | Train a small "student" model to mimic a big "teacher" → cheaper/faster |

> **Interview line:** "For most needs I reach for prompting or RAG first. When behavior tuning is required, LoRA/QLoRA give efficient fine-tuning — QLoRA quantizes the base to 4-bit so a large model fits on a single GPU. For alignment, DPO is a simpler alternative to full RLHF because it skips the separate reward model."

---

## PART 5 — Inference Optimization (know the names + what they do)

Serving LLMs fast and cheap at scale.

| Tool/technique | What it does |
|----------------|--------------|
| **vLLM** | High-throughput serving; **PagedAttention** + **continuous batching** → serve many requests efficiently. The go-to open-source LLM server |
| **TensorRT-LLM** | NVIDIA's optimized inference engine for max GPU performance |
| **Triton Inference Server** | NVIDIA's model server — hosts models, batching, multi-framework |
| **ONNX** | Open format to export models for optimized cross-platform inference |
| **Quantization (INT8/INT4)** | Lower precision weights → smaller, faster, less memory (small accuracy trade-off) |
| **Continuous batching** | Add/remove requests from a batch dynamically → high GPU utilization (vLLM) |
| **KV cache** | Cache attention keys/values so you don't recompute past tokens each step |
| **Speculative decoding** | A small draft model proposes tokens, the big model verifies → faster |

**Latency terms:** **TTFT** (time to first token), **inter-token latency**, **throughput** (tokens/sec), **p50/p95/p99** latency.

> "For self-hosted serving I'd use vLLM for continuous batching and PagedAttention, quantize to INT8/INT4 to fit memory, and watch TTFT and p95 latency and tokens/sec."

---

## PART 6 — Cloud deployment (know the mapping)
| Task | AWS | Azure | GCP |
|------|-----|-------|-----|
| Managed ML | SageMaker | ML Studio | Vertex AI |
| Containers | ECS/EKS | AKS | Cloud Run/GKE |
| Serverless fn | Lambda | Functions | Cloud Functions |
| Storage | S3 | Blob | GCS |
| Managed LLMs | Bedrock | Azure OpenAI | Vertex AI |

---

## PART 7 — The deployment story (senior framing)
> "I own deployment end-to-end: containerize with Docker, deploy to Kubernetes with rolling updates and rollback, CI/CD via GitHub Actions running evals before deploy, models tracked and versioned in MLflow's registry. For serving I optimize with vLLM and quantization, and I design circuit breakers and graceful degradation for when the model API fails."

---

## Honesty note
You likely haven't run vLLM/QLoRA/K8s at production scale — that's fine. Say: "I understand the architecture and trade-offs; I've deployed via containers and would ramp on vLLM/K8s at scale quickly." Understanding + honesty is credible.
