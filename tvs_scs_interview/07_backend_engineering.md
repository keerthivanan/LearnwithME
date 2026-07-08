# Backend & API Engineering (Pillar)

The JD wants real backend skills for AI microservices. This is very learnable — and testable.

---

## 1. FastAPI (the AI-serving default — know it well)
Modern, async, fast Python web framework with automatic docs + validation.
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):          # Pydantic model = automatic validation
    question: str
    top_k: int = 5

@app.post("/ask")
async def ask(q: Query):         # async endpoint
    answer = await run_rag(q.question, q.top_k)
    return {"answer": answer}
```
- **Pydantic** validates request/response → structured, typed, auto error messages
- Auto **Swagger UI** at `/docs`
- `async def` endpoints for concurrency
- Dependency injection (`Depends`) for auth, DB sessions

**FastAPI vs Flask vs Django:**
- **FastAPI:** async-native, fast, auto-validation/docs — best for AI microservices/APIs
- **Flask:** simple, lightweight, sync (async possible) — quick apps
- **Django:** batteries-included, ORM, admin — full web apps, heavier

---

## 2. Async programming (know this cold — you asked about it)
Async lets the server do other work **while waiting** on I/O (LLM API, DB, network) instead of blocking. Critical for AI apps that spend most time waiting on model calls.
```python
import asyncio
# run two LLM calls concurrently instead of sequentially
a, b = await asyncio.gather(call_llm(p1), call_llm(p2))
```
- `async def` / `await` — pause on I/O, let other requests run
- **Use for I/O-bound** (LLM/DB/API calls), **not CPU-bound** (heavy math → use processes)
- One async server handles many concurrent users because it never sits idle waiting

### Celery (background tasks)
For long/heavy jobs (batch embedding, fine-tuning trigger, report generation) → offload to **Celery** workers with a broker (Redis/RabbitMQ) so the API responds instantly and the job runs in the background.
> "Fast requests are async in FastAPI; long-running jobs go to Celery workers so the API stays responsive."

---

## 3. API design
- **REST** — resources + HTTP methods (GET/POST/PUT/DELETE), JSON
- **WebSockets** — persistent two-way connection → **streaming LLM tokens** to the UI in real time (chat)
- **gRPC** — fast binary RPC (service-to-service, low latency)
- **GraphQL** — client asks for exactly the fields it wants
- **Streaming responses** — stream tokens as they generate (Server-Sent Events / WebSocket) for perceived speed

> For AI chat: REST for standard calls, WebSockets/SSE for streaming tokens.

---

## 4. Auth & security
- **OAuth2** — delegated authorization standard (login with provider, scoped access tokens)
- **JWT (JSON Web Token)** — signed token carrying claims; stateless auth (server verifies signature, no session lookup)
- **API keys** — simple service auth
- **Rate limiting** — cap requests per user/key → protect system + control LLM cost (token bucket)
- **Input validation** — Pydantic/schema; sanitize to prevent injection
- **Structured error handling** — consistent error responses, never leak stack traces/secrets
- **Secrets** — env vars / secret managers, never in code (you practiced this — .env + .gitignore)

---

## 5. Databases & caching (know the mapping)
| Store | Use |
|-------|-----|
| **PostgreSQL** | Relational; **pgvector** adds vector search in the same DB |
| **MySQL** | Relational |
| **Redis** | In-memory cache, sessions, rate-limit counters, Celery broker, pub/sub |
| **MongoDB** | Document/NoSQL — flexible JSON docs |
| **Elasticsearch** | Full-text + vector search, logs |
| **Object storage (S3/Blob)** | Files, model artifacts, documents |

**Caching for LLM apps (cost saver):**
- Cache embeddings and identical prompts/answers (semantic cache) in Redis → skip repeat LLM calls
- **Schema design, indexing, query optimization** — know: an index speeds reads, slows writes; normalize vs denormalize; avoid N+1 queries

---

## 6. Microservices
Split the system into small, independently deployable services (e.g., ingestion service, retrieval service, agent service, gateway). Pros: scale/deploy independently, fault isolation. Cons: complexity, network calls, need tracing (→ observability file). An **API Gateway** fronts them (routing, auth, rate limiting).

---

## 7. A production AI service checklist (senior answer)
> "My AI microservice is FastAPI with Pydantic validation, async endpoints for model calls, WebSocket streaming for chat, JWT/OAuth2 auth, per-key rate limiting to control cost, Redis caching for embeddings and repeated prompts, Postgres (pgvector) for data + vectors, Celery for heavy background jobs, structured error handling and logging, and it's containerized with health checks for Kubernetes."

---

## The sentence that proves you can build the backend
> "I build AI services in FastAPI — async for the I/O-bound LLM calls, Pydantic for typed validation, WebSockets for token streaming, JWT auth with rate limiting for cost control, Redis caching to cut repeat LLM spend, and Celery for long jobs. Everything's containerized with proper error handling and structured logs."
