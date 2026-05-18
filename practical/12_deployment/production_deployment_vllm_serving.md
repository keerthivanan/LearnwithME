# 11 — Model Deployment at Scale

> JD: "Deploy AI models at scale in production environments, ensuring scalability, robustness, and efficiency."

---

## 1. The Deployment Challenge

**What it is:** Getting a trained LLM to actually work in production is a completely different challenge from training. You need to serve thousands of users with low latency, high throughput, and zero downtime — all while controlling GPU costs.

Deploying LLMs in production is fundamentally different from deploying small models:
- **Latency**: User wants a response in < 2 seconds (or they leave)
- **Throughput**: Handle 1000s of concurrent users (not just one at a time)
- **Memory**: 7B model = 14GB VRAM minimum (expensive hardware required)
- **Cost**: GPU time is expensive — you must keep it utilized well to get value
- **Reliability**: Can't have downtime (users lose trust immediately)

**Analogy:** Deploying an LLM is like opening a restaurant. Your food (the model) might be excellent, but you also need to handle many customers at once (throughput), serve them quickly (latency), keep the kitchen running non-stop (reliability), and manage food costs carefully (GPU cost).

---

## 2. Inference Serving Frameworks

**What it is:** Specialized software frameworks purpose-built to serve LLMs efficiently in production. They implement optimizations like batching and memory management that naive code does not.

### vLLM (Most Important)

**What it is:** The de-facto standard for production LLM serving. Built at Berkeley, vLLM implements PagedAttention and continuous batching, making it 10–24× more throughput than naive HuggingFace serving.

High-throughput LLM serving. De facto production standard.

**Key innovations:**
- PagedAttention: memory-efficient KV cache management (see Section 4)
- Continuous batching: no wasted GPU time between requests (see Section 4)
- Tensor parallelism support: use multiple GPUs for one model
- OpenAI-compatible API: drop-in replacement for any OpenAI client code

```python
# Start vLLM server (run this in terminal, then call it like OpenAI API)
# vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 2

from vllm import LLM, SamplingParams  # vLLM library

# Load the model — vLLM automatically applies its optimizations
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",  # HuggingFace model ID
    tensor_parallel_size=2,                      # split model across 2 GPUs (needed if model > 1 GPU memory)
    gpu_memory_utilization=0.90,                 # use 90% of GPU memory for the model and KV cache (leave 10% buffer)
    max_model_len=8192,                          # maximum sequence length to support
)

# Configure how text is generated
sampling_params = SamplingParams(
    temperature=0.7,     # some creativity (not too random)
    top_p=0.9,           # nucleus sampling
    max_tokens=512,      # stop generating after 512 tokens
)

# Generate — vLLM batches this with other concurrent requests automatically
outputs = llm.generate(["What is the capital of France?"], sampling_params)
print(outputs[0].outputs[0].text)  # access the generated text from the output
```

**WHY vLLM is the standard:** Before vLLM, serving LLMs meant one request at a time or wasteful static batching. vLLM's PagedAttention and continuous batching make GPU utilization 5-10× better than naive implementations.

### Hugging Face Text Generation Inference (TGI)

**What it is:** Hugging Face's production serving solution. Similar capabilities to vLLM but with tighter integration into the HuggingFace Hub ecosystem and Docker-first deployment.

Production LLM serving from Hugging Face.
- Similar performance to vLLM
- Integrated with HF Hub (pull any HF model directly)
- Docker-based deployment (easy to get started)

```bash
docker run --gpus all -p 8080:80 \   # use all GPUs, expose port 8080
  -e MODEL_ID=meta-llama/Llama-3.1-8B-Instruct \  # set model via environment variable
  ghcr.io/huggingface/text-generation-inference:latest  # run TGI Docker image
```

```python
import requests                                   # standard HTTP library
response = requests.post("http://localhost:8080/generate", json={
    "inputs": "What is AI?",                      # the prompt to generate from
    "parameters": {"max_new_tokens": 200}         # generation parameters
})
```

### Ollama (Local / Dev)

**What it is:** A simple tool for running LLMs locally on your laptop. Not for production — for local development, testing, and prototyping.

Easy local model serving — ideal for development and testing.
```bash
ollama serve                    # start the Ollama server in the background
ollama pull llama3.1            # download the LLaMA 3.1 model from Ollama's registry
ollama run llama3.1 "Hello!"    # run inference with a prompt directly from command line
```

**WHY Ollama for dev:** You can iterate on prompts locally without cloud costs or API rate limits. Then switch to vLLM for production.

### TorchServe

**What it is:** PyTorch's official production serving framework. More general-purpose (not LLM-specific) — you package your model into a `.mar` archive and deploy via a management API.

PyTorch's official model serving framework.
```python
torch-model-archiver --model-name llm --version 1.0 \   # create the model archive
  --serialized-file model.pt --handler custom_handler.py  # bundle model weights + inference code
torchserve --start --model-store model_store --models llm=llm.mar  # start server with model
```

---

## 3. REST API Design for LLMs

### OpenAI-Compatible API (Standard)

**What it is:** The OpenAI API format has become the industry standard. By implementing the same API format, your server works with all existing OpenAI client libraries — a huge ecosystem advantage.

**Analogy:** Using OpenAI's API format is like using standard USB connectors. Any device with a USB plug works with any USB socket. Any code using the OpenAI client works with your server.

Most frameworks implement this. Clients work with any backend.

```python
from fastapi import FastAPI               # Python web framework for building APIs
from pydantic import BaseModel            # data validation library
from vllm import LLM, SamplingParams     # vLLM for model serving

app = FastAPI()                           # create the FastAPI web application
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")  # load model once at startup (not per request)

# Define the request schema — this validates incoming JSON automatically
class ChatRequest(BaseModel):
    messages: list       # list of message dicts: [{"role": "user", "content": "..."}]
    max_tokens: int = 512   # default 512 tokens if not specified
    temperature: float = 0.7  # default temperature if not specified

# Route that matches OpenAI's API format exactly
@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    prompt = format_messages(request.messages)    # convert messages list to single prompt string
    outputs = llm.generate([prompt], SamplingParams(
        temperature=request.temperature,          # use the requested temperature
        max_tokens=request.max_tokens             # use the requested max tokens
    ))
    # Return response in OpenAI's exact JSON format
    return {
        "choices": [{"message": {"content": outputs[0].outputs[0].text}}]  # match OpenAI's response structure
    }
```

**WHY follow OpenAI's format:** All LLM application code (LangChain, LlamaIndex, etc.) is written to call OpenAI's format. By matching it, you can swap your backend without changing application code.

### Streaming Responses

**What it is:** Instead of waiting for the full response to be generated before sending anything, stream tokens back to the user as they are generated. This makes the app feel much faster because the user sees output immediately.

**Analogy:** Streaming is like reading a book while it is being printed page by page versus waiting for the whole book to be printed before you can start reading. Same total time — but feels much faster.

LLMs generate token by token — stream them for better UX.

```python
from fastapi.responses import StreamingResponse  # special response type for streaming

@app.post("/v1/chat/completions/stream")
async def chat_stream(request: ChatRequest):
    # Generator function that yields tokens as they are produced
    async def generate():
        for token in llm.stream_generate(prompt):        # get tokens one at a time from the model
            yield f"data: {json.dumps({'token': token})}\n\n"  # send each token as a Server-Sent Event
        yield "data: [DONE]\n\n"                          # signal that generation is complete

    # Return a streaming response — data flows to client as tokens are generated
    return StreamingResponse(generate(), media_type="text/event-stream")
```

**WHY Server-Sent Events format:** This is the standard format browsers understand for streaming. `data: ...\n\n` is the SSE protocol — any browser can consume this without special libraries.

---

## 4. Scalability Patterns

**What it is:** Techniques to serve more users, handle more requests, and stay responsive under high load.

### Horizontal Scaling (Multiple Instances)

**What it is:** Run multiple copies of your LLM service. A load balancer distributes incoming requests across all copies. This is the most straightforward way to increase throughput.

**Analogy:** Horizontal scaling is like opening more checkout lanes in a supermarket. Each lane is one LLM instance. More lanes = more customers served simultaneously.

Run multiple instances behind a load balancer.
```
Client → Load Balancer → [LLM Instance 1]
                      → [LLM Instance 2]
                      → [LLM Instance 3]
```

**WHY horizontal scaling is preferred over vertical:** Vertical scaling (bigger GPU) has limits and single points of failure. Horizontal scaling is unlimited in principle and provides redundancy — one instance failing doesn't take down the service.

### Request Batching

**What it is:** Group multiple user requests into a single batch and process them together. GPUs are much more efficient processing batches than single requests.

Batch multiple requests together for GPU efficiency.
- vLLM's continuous batching does this automatically (you don't have to manage it)

### Async Processing with Queue

**What it is:** For very high traffic or long-running requests, use a job queue. Requests go into the queue immediately (fast acknowledgment to user), workers process them, user polls or is notified when done.

**Analogy:** Like ordering food at a restaurant via a ticket system. You place your order (go into queue), get a ticket number, and pick up when ready. The kitchen processes orders in parallel without customers blocking each other.

For high traffic: use a job queue (Celery, Redis Queue, Kafka).
```
Client → API → Job Queue → Worker (LLM) → Result Store → Client polls for result
```

**WHY queues:** Without a queue, a burst of 1000 simultaneous requests might crash your server or cause timeouts. With a queue, all requests are accepted immediately, processed at the server's sustainable rate, and results retrieved when ready.

### Caching

**What it is:** If users frequently ask the same or very similar questions, you can save the answer and return it instantly without running inference again. Semantic caching finds similar (not just exact) matches.

Cache common prompt responses (semantic cache):
```python
# If similar query was seen before, return cached response
from langchain.cache import InMemoryCache    # LangChain's built-in cache
langchain.llm_cache = InMemoryCache()        # enable LLM response caching globally
```

GPTCache, Redis, or in-memory caches.

**WHY semantic caching:** Exact string matching misses cases like "What is RAG?" vs "Can you explain RAG?" A semantic cache uses embeddings to find similar past queries — much higher cache hit rate.

---

## 5. Key Deployment Metrics

**What it is:** The numbers you need to monitor in production to know if your LLM service is healthy and meeting user expectations.

| Metric | Description | Target |
|--------|------------|--------|
| TTFT (Time to First Token) | Latency until first token arrives at user's screen | < 500ms |
| TPS (Tokens per Second) | Generation speed per user | > 30 TPS |
| Throughput | Total tokens per second across all users | Maximize |
| GPU Utilization | % time GPU is doing useful compute | > 80% |
| P99 Latency | 99th percentile response time (worst case for 99% of users) | < 5s |

**WHY TTFT is the most important UX metric:** Users do not perceive "total generation time" — they perceive how long they wait before *something* appears. A model that takes 3 seconds to start but then streams fast feels much better than one that waits 4 seconds and dumps everything at once.

**WHY P99 not average:** Average latency hides the "long tail." If 1% of requests take 30 seconds, that is 1 in 100 users having a terrible experience. P99 reveals this.

---

## 6. Model as API (Production Architecture)

**What it is:** The full reference architecture for a production LLM deployment. Each layer has a specific responsibility.

### Full Architecture
```
                    ┌─────────────────────────────┐
User ──────────────→│       API Gateway           │  ← controls who can access and how much
                    │  (Rate limiting, Auth)       │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      Inference Service       │  ← your application logic + LLM
                    │   (FastAPI + vLLM/TGI)       │
                    └──────┬──────────────┬────────┘
                           │              │
               ┌───────────▼──┐    ┌──────▼──────────┐
               │  GPU Server  │    │   Vector DB      │  ← retrieved context for RAG
               │  (8×A100)    │    │   (for RAG)      │
               └──────────────┘    └─────────────────-┘
```

### Components

**What each layer does:**
- **API Gateway**: Authentication (who are you?), rate limiting (how many requests/second?), logging (what was asked?)
- **Inference Service**: FastAPI receives HTTP requests, builds prompts, calls vLLM
- **GPU Cluster**: A100/H100 servers running the actual model
- **Vector DB**: For RAG — stores document embeddings so the model can retrieve relevant context
- **Cache**: Redis for prompt caching — if same question asked again, skip inference entirely
- **Monitoring**: Prometheus collects metrics, Grafana visualizes them
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana) stores and searches logs

---

## 7. Containerization & Kubernetes

### Docker for LLMs

**What it is:** Package your LLM service into a Docker container — a self-contained unit with all dependencies included. This makes deployment reproducible (works the same everywhere) and portable (runs on any machine with Docker).

**Analogy:** Docker is like a shipping container. You pack everything your application needs into a standardized container. The container can be shipped to any port (server) and it will work exactly the same way.

```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04   # start from NVIDIA's base image with CUDA pre-installed

RUN pip install vllm fastapi uvicorn        # install Python dependencies inside the container

COPY app.py .                               # copy your application code into the container

EXPOSE 8000                                 # document that the container listens on port 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]  # command to start the server
```

```bash
docker build -t llm-service .              # build the Docker image (named "llm-service")
docker run --gpus all -p 8000:8000 llm-service  # run container with GPU access, expose port 8000
```

**WHY the NVIDIA CUDA base image:** Your LLM needs GPU drivers. Starting from NVIDIA's image means CUDA is pre-installed correctly — you don't have to manage GPU driver installation.

### Kubernetes GPU Deployment

**What it is:** Kubernetes (K8s) is a container orchestration system that manages deploying, scaling, and restarting containers automatically. For LLM serving, it manages your Docker containers across a cluster of GPU machines.

```yaml
apiVersion: apps/v1
kind: Deployment              # tells K8s this is a Deployment (manages container replicas)
metadata:
  name: llm-deployment        # name of this deployment
spec:
  replicas: 3                 # run 3 identical copies of this service for redundancy
  template:
    spec:
      containers:
      - name: llm
        image: llm-service:latest    # use our Docker image
        resources:
          limits:
            nvidia.com/gpu: "2"      # each container gets 2 GPUs
        env:
        - name: MODEL_ID
          value: "meta-llama/Llama-3.1-8B-Instruct"  # pass model ID as environment variable
```

**WHY Kubernetes:** K8s automatically restarts crashed containers, distributes load across nodes, and scales up/down based on traffic. This gives you production-grade reliability with much less manual management.

---

## 8. Model Versioning & MLOps

### MLflow

**What it is:** An experiment tracking and model registry tool. It logs all your training runs (hyperparameters, metrics, model artifacts) so you can reproduce results, compare experiments, and version your models.

**Analogy:** MLflow is like a lab notebook for machine learning. Every experiment is recorded with all its settings and results. You can look back at any previous experiment and reproduce it exactly.

```python
import mlflow                  # experiment tracking library
import mlflow.pytorch          # PyTorch integration

with mlflow.start_run():       # start tracking an experiment run
    mlflow.log_param("model", "llama-3.1-8b")  # record which model you used
    mlflow.log_param("lora_rank", 16)           # record LoRA rank hyperparameter
    mlflow.log_metric("eval_loss", 0.23)        # record the evaluation loss result
    mlflow.pytorch.log_model(model, "model")    # save the model itself, linked to this run
```

### Hugging Face Hub

**What it is:** A model registry (like GitHub for models). You push your fine-tuned model to the Hub and it is stored, versioned, and accessible from anywhere with a single model ID.

```python
from huggingface_hub import HfApi  # HuggingFace Hub API client

api = HfApi()
api.create_repo("my-finetuned-llama", private=True)   # create a private repo for your model
model.push_to_hub("my-finetuned-llama")               # upload model weights to the Hub
tokenizer.push_to_hub("my-finetuned-llama")           # upload tokenizer alongside the model
```

**WHY HF Hub for model versioning:** Your model weights are tied to your training code in git. The Hub gives every model upload a unique version, lets you tag releases, and makes it easy to roll back to any previous version.

---

## 9. Guardrails & Safety

**What it is:** Mechanisms that filter unsafe or harmful content from LLM inputs and outputs in production. Essential for any consumer-facing application.

### Output Filtering

**What it is:** Run a classifier on model outputs before sending them to users to detect and block harmful content.

```python
from transformers import pipeline       # HuggingFace pipeline for classification

# Load a toxicity detection model
classifier = pipeline("text-classification", model="unitary/toxic-bert")  # BERT fine-tuned for toxicity
def is_safe(text):
    result = classifier(text)[0]        # classify the text
    return result["label"] != "toxic" or result["score"] < 0.8  # safe if not toxic OR low confidence
```

**WHY 0.8 threshold:** Setting the threshold too low blocks too many harmless messages (false positives). 0.8 means you only block content the classifier is highly confident is toxic.

### LLM Guardrails Libraries

**What it is:** Purpose-built libraries that handle safety, schema validation, and constraint satisfaction for LLM outputs in production.

- **Guardrails AI**: Validates output structure (ensures JSON schema compliance, formats numbers correctly) and satisfies content constraints
- **NVIDIA NeMo Guardrails**: Defines conversation safety rails (topics the model will refuse to discuss) using a declarative language
- **LlamaGuard**: Meta's LLM specifically trained to classify whether inputs and outputs are safe or unsafe according to a customizable safety taxonomy

---

## 10. Monitoring in Production

**What it is:** Continuously track your LLM service's health, performance, and quality in production. Without monitoring you will not know when things break or degrade.

### Key Things to Monitor

```python
# Track with Prometheus — a time-series metrics database
from prometheus_client import Counter, Histogram   # Prometheus metric types

# Counter: monotonically increasing count (never goes down)
request_count = Counter("llm_requests_total", "Total requests")  # track how many requests served

# Histogram: tracks distribution of values (good for latencies)
latency = Histogram("llm_latency_seconds", "Request latency", buckets=[0.1, 0.5, 1, 2, 5])  # measure response times

# Another counter for output size
token_count = Counter("llm_tokens_generated_total", "Tokens generated")  # track total tokens (cost monitoring)
```

**WHY Prometheus:** It is the industry standard for metrics collection. Grafana (visualization tool) reads from Prometheus to create dashboards. Alertmanager (alerting tool) reads from Prometheus to send alerts when metrics cross thresholds.

### Drift Detection

**What it is:** Monitor if the distribution of incoming requests changes over time. If user questions suddenly become very different from what the model was trained or tested on, quality may have silently degraded.

Monitor if input distribution changes over time — could indicate:
- The model needs retraining (world has changed)
- RAG knowledge base needs updating (documents are outdated)
- A new use case is being discovered by users
- A prompt injection attack is in progress

---

## Blue-Green and Canary Deployment for LLM Models

**What it is:** Safe deployment strategies for releasing new model versions without causing downtime or exposing all users to a potentially broken new version at once.

**Analogy:** Blue-Green is like having an understudy in theatre — the main actor can be swapped out instantly with no show interruption. Canary is like a gradual product rollout — launch in 5 cities first, check reviews, then expand.

Blue-Green Deployment:
- Run TWO identical environments: Blue (current live version) and Green (new version being tested)
- Deploy new model to Green environment, run full automated tests
- Switch traffic: Blue → Green (one DNS or load balancer configuration change)
- Blue stays live as instant rollback if Green has issues within the first hours
- Zero downtime deployment (no gap between old and new version)

Kubernetes blue-green:
```bash
# Deploy new model version as a separate deployment in Kubernetes
kubectl apply -f llm-service-green.yaml      # create the Green deployment
# Run smoke tests against Green before switching traffic
pytest tests/smoke_test.py --endpoint=green-service  # verify Green is healthy
# Switch traffic (update which pods the service selector points to)
kubectl patch service llm-service -p '{"spec":{"selector":{"version":"green"}}}'  # flip traffic to Green
# After confirming Green is stable, clean up Blue
kubectl delete deployment llm-service-blue   # remove the old Blue deployment
```

Canary Release (safer for LLMs because you monitor quality before full rollout):
- Send 5% of real traffic to new model, 95% to old model
- Monitor: latency, error rate, user satisfaction scores, LLM-as-judge quality scores
- Gradually increase: 5% → 20% → 50% → 100% over hours or days
- If metrics degrade at any point: roll back to 0% immediately

Kubernetes canary with weighted routing (Istio service mesh):
```yaml
# Route 5% to new model, 95% to old model
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService              # Istio virtual service for traffic splitting
spec:
  http:
  - route:
    - destination: {host: llm-v1}  # old model
      weight: 95                    # 95% of traffic goes here
    - destination: {host: llm-v2}  # new model
      weight: 5                     # 5% of traffic goes here (the "canary")
```

**Interview: "How do you deploy a new LLM version without downtime?"** → "Blue-green for full version switches — deploy to green environment, run tests, flip traffic. Canary for gradual rollout — start at 5% traffic, monitor quality metrics, incrementally increase. Always keep previous version ready for instant rollback."

---

## LLM Serving Cost Estimation

**What it is:** How to calculate and compare the cost of different serving options. This comes up in system design interviews when asked about build vs buy decisions.

Key formula:
```
Monthly cost = (requests/day × tokens/request × cost_per_1M_tokens × 30) / 1M
```

Cloud API costs (2024):
| Model | Input $/1M tokens | Output $/1M tokens |
|-------|-----------------|-------------------|
| GPT-4o | $2.50 | $10.00 |
| Claude 3.5 Sonnet | $3.00 | $15.00 |
| GPT-4o-mini | $0.15 | $0.60 |
| LLaMA-3.1-8B (self-hosted) | ~$0.05 | ~$0.05 |

Self-hosted GPU cost calculation:
```
A100 80GB on AWS (p4d.24xlarge): ~$32/hour
Serves LLaMA-3.1-70B at ~1000 tokens/sec
1000 tokens/sec × 3600 sec × 24hr × 30days = 2.6B tokens/month
Cost per 1M tokens = $32 × 720hr / 2600 = ~$8.9/1M tokens
```

**WHY self-hosting can be cheaper at scale:**
- High volume: >10M tokens/day (API per-token costs quickly exceed fixed GPU costs)
- Latency: need <100ms TTFT (cloud APIs have additional network latency)
- Privacy: cannot send user data to third-party APIs (HIPAA, GDPR)
- Customization: need a fine-tuned model not available via any API

**Interview: "How do you decide between cloud API vs self-hosted LLM?"** → "Calculate monthly token volume × cost per token for both options. Self-hosting wins above ~10M tokens/day. Also consider: latency requirements (self-hosted has no network overhead), data privacy constraints, and whether you need a fine-tuned model. Start with API for validation, switch to self-host when cost justifies it."

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
TGI:               Hugging Face's serving framework, Docker-first
Continuous batching: Insert new requests as slots open, maximizes GPU utilization
PagedAttention:    Non-contiguous KV cache pages, reduces fragmentation
Streaming:         Return tokens as generated, better perceived UX
TTFT:              Time to first token — the key latency metric users feel
P99 latency:       99th percentile — reveals worst-case experience
Scaling:           Horizontal (more instances) + batching + caching
Docker/K8s:        Containerize for reproducible, auto-scaling deployment
Guardrails:        LlamaGuard, NeMo, Guardrails AI for safety filtering
Blue-Green:        Zero downtime swap between old and new model
Canary:            Gradual traffic shift with rollback capability
```
