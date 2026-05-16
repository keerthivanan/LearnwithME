# 11 — Model Deployment at Scale

> JD: "Deploy AI models at scale in production environments, ensuring scalability, robustness, and efficiency."

---

## 1. The Deployment Challenge

Deploying LLMs in production is fundamentally different from deploying small models:
- **Latency**: User wants a response in < 2 seconds
- **Throughput**: Handle 1000s of concurrent users
- **Memory**: 7B model = 14GB VRAM minimum
- **Cost**: GPU is expensive, must utilize it well
- **Reliability**: Can't have downtime

---

## 2. Inference Serving Frameworks

### vLLM (Most Important)
High-throughput LLM serving. De facto production standard.

**Key innovations:**
- PagedAttention: memory-efficient KV cache management
- Continuous batching: no wasted GPU time between requests
- Tensor parallelism support
- OpenAI-compatible API

```python
# Start vLLM server
# vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 2

from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=2,      # use 2 GPUs
    gpu_memory_utilization=0.90,
    max_model_len=8192,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
)

outputs = llm.generate(["What is the capital of France?"], sampling_params)
print(outputs[0].outputs[0].text)
```

### Hugging Face Text Generation Inference (TGI)
Production LLM serving from Hugging Face.
- Similar to vLLM
- Integrated with HF Hub
- Docker-based deployment

```bash
docker run --gpus all -p 8080:80 \
  -e MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \
  ghcr.io/huggingface/text-generation-inference:latest
```

```python
import requests
response = requests.post("http://localhost:8080/generate", json={
    "inputs": "What is AI?",
    "parameters": {"max_new_tokens": 200}
})
```

### Ollama (Local / Dev)
Easy local model serving.
```bash
ollama serve
ollama pull llama3.1
ollama run llama3.1 "Hello!"
```

### TorchServe
PyTorch's official model serving framework.
```python
torch-model-archiver --model-name llm --version 1.0 \
  --serialized-file model.pt --handler custom_handler.py
torchserve --start --model-store model_store --models llm=llm.mar
```

---

## 3. REST API Design for LLMs

### OpenAI-Compatible API (Standard)
Most frameworks implement this. Clients work with any backend.

```python
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

app = FastAPI()
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

class ChatRequest(BaseModel):
    messages: list
    max_tokens: int = 512
    temperature: float = 0.7

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    prompt = format_messages(request.messages)
    outputs = llm.generate([prompt], SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens
    ))
    return {
        "choices": [{"message": {"content": outputs[0].outputs[0].text}}]
    }
```

### Streaming Responses
LLMs generate token by token — stream them for better UX.

```python
from fastapi.responses import StreamingResponse

@app.post("/v1/chat/completions/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        for token in llm.stream_generate(prompt):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## 4. Scalability Patterns

### Horizontal Scaling (Multiple Instances)
Run multiple instances behind a load balancer.
```
Client → Load Balancer → [LLM Instance 1]
                      → [LLM Instance 2]
                      → [LLM Instance 3]
```

### Request Batching
Batch multiple requests together for GPU efficiency.
- vLLM's continuous batching does this automatically

### Async Processing with Queue
For high traffic: use a job queue (Celery, Redis Queue, Kafka).
```
Client → API → Job Queue → Worker (LLM) → Result Store → Client polls
```

### Caching
Cache common prompt responses (semantic cache):
```python
# If similar query was seen before, return cached response
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()
```

GPTCache, Redis, or in-memory caches.

---

## 5. Key Deployment Metrics

| Metric | Description | Target |
|--------|------------|--------|
| TTFT (Time to First Token) | Latency until first token arrives | < 500ms |
| TPS (Tokens per Second) | Generation speed per user | > 30 TPS |
| Throughput | Total tokens/second across all users | Maximize |
| GPU Utilization | % time GPU is doing useful work | > 80% |
| P99 Latency | 99th percentile response time | < 5s |

---

## 6. Model as API (Production Architecture)

### Full Architecture
```
                    ┌─────────────────────────────┐
User ──────────────→│       API Gateway           │
                    │  (Rate limiting, Auth)       │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      Inference Service       │
                    │   (FastAPI + vLLM/TGI)       │
                    └──────┬──────────────┬────────┘
                           │              │
               ┌───────────▼──┐    ┌──────▼──────────┐
               │  GPU Server  │    │   Vector DB      │
               │  (8×A100)    │    │   (for RAG)      │
               └──────────────┘    └─────────────────-┘
```

### Components
- **API Gateway**: Authentication, rate limiting, logging
- **Inference Service**: FastAPI/Flask + vLLM
- **GPU Cluster**: A100/H100 servers
- **Vector DB**: For RAG (Pinecone, Qdrant)
- **Cache**: Redis for prompt caching
- **Monitoring**: Prometheus + Grafana
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana)

---

## 7. Containerization & Kubernetes

### Docker for LLMs
```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

RUN pip install vllm fastapi uvicorn

COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t llm-service .
docker run --gpus all -p 8000:8000 llm-service
```

### Kubernetes GPU Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-deployment
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: llm
        image: llm-service:latest
        resources:
          limits:
            nvidia.com/gpu: "2"        # Request 2 GPUs
        env:
        - name: MODEL_ID
          value: "meta-llama/Llama-3.1-8B-Instruct"
```

---

## 8. Model Versioning & MLOps

### MLflow
```python
import mlflow
import mlflow.pytorch

with mlflow.start_run():
    mlflow.log_param("model", "llama-3.1-8b")
    mlflow.log_param("lora_rank", 16)
    mlflow.log_metric("eval_loss", 0.23)
    mlflow.pytorch.log_model(model, "model")
```

### Hugging Face Hub
```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo("my-finetuned-llama", private=True)
model.push_to_hub("my-finetuned-llama")
tokenizer.push_to_hub("my-finetuned-llama")
```

---

## 9. Guardrails & Safety

### Output Filtering
```python
from transformers import pipeline

# Detect harmful content
classifier = pipeline("text-classification", model="unitary/toxic-bert")
def is_safe(text):
    result = classifier(text)[0]
    return result["label"] != "toxic" or result["score"] < 0.8
```

### LLM Guardrails Libraries
- **Guardrails AI**: Schema validation, constraint satisfaction
- **NVIDIA NeMo Guardrails**: Conversation safety rails
- **LlamaGuard**: Meta's safety model for input/output filtering

---

## 10. Monitoring in Production

### Key Things to Monitor
```python
# Track with Prometheus + custom metrics
from prometheus_client import Counter, Histogram

request_count = Counter("llm_requests_total", "Total requests")
latency = Histogram("llm_latency_seconds", "Request latency", buckets=[0.1, 0.5, 1, 2, 5])
token_count = Counter("llm_tokens_generated_total", "Tokens generated")
```

### Drift Detection
Monitor if input distribution changes over time — could indicate the model needs retraining or RAG knowledge needs updating.

---

## 11. Interview Questions — Deployment

**Q: How would you serve a 7B LLM in production for 1000 concurrent users?**
> Use vLLM with continuous batching and PagedAttention. Deploy with tensor parallelism across 2 A100 40GB GPUs. Put multiple instances behind a load balancer. Add a Redis cache for repeated prompts. Use async/streaming responses to improve perceived latency. Monitor TTFT and throughput with Prometheus.

**Q: What is continuous batching in LLM serving?**
> Traditional batching waits for all sequences in a batch to finish before accepting new requests — idle GPU time. Continuous batching (vLLM) inserts new requests into the batch as soon as a slot opens, maximizing GPU utilization. This dramatically improves throughput for mixed-length requests.

**Q: What is PagedAttention?**
> A memory management technique in vLLM inspired by OS virtual memory paging. Instead of pre-allocating a contiguous KV cache for each sequence (wasteful and fragmenting), it stores the KV cache in non-contiguous pages and manages them dynamically. Eliminates internal fragmentation and enables more concurrent requests.

**Q: What is TTFT (Time to First Token) and why does it matter?**
> TTFT is the latency from when a user sends a request to when they see the first generated token. It's the main driver of perceived responsiveness. Streaming responses are used so users see output immediately rather than waiting for the full response.

**Q: How do you handle model versioning in production?**
> Use MLflow or Hugging Face Hub for version tracking. Use blue-green deployments or canary releases to gradually shift traffic to new model versions. Store adapter weights (LoRA) separately from base model for efficient updates.

---

## Quick Reference Cheat Sheet

```
vLLM:              Best production serving (PagedAttention + continuous batching)
TGI:               Hugging Face's serving framework
Continuous batching: Insert new requests as slots open, max GPU utilization
PagedAttention:    Non-contiguous KV cache pages, reduces fragmentation
Streaming:         Return tokens as generated, better UX
TTFT:              Time to first token — key latency metric
Scaling:           Horizontal (more instances) + batching + caching
Docker/K8s:        Containerize for reproducible deployment
Guardrails:        LlamaGuard, NeMo, Guardrails AI for safety
```
