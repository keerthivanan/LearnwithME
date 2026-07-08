# RAG & Knowledge Retrieval — Deep (⭐ Top Pillar)

RAG is the #1 production GenAI pattern. This file covers everything the JD lists.

---

## 1. What RAG is and why
**Retrieval-Augmented Generation:** fetch relevant documents and put them in the prompt so the LLM answers from YOUR data — current, private, grounded. Reduces hallucination; no retraining needed.

```
INGESTION (once):  docs → chunk → embed → store in vector DB
QUERY (per request): question → embed → retrieve top-k chunks → prompt with context → LLM answers
```

**Why RAG over fine-tuning for knowledge:** RAG adds *facts* (easy to update, cite sources); fine-tuning changes *behavior/style*. For "answer from our docs," RAG wins.

---

## 2. Chunking (where RAG quality is won or lost)

Split documents into pieces small enough to embed and fit in context, big enough to be meaningful.

- **Fixed-size** (e.g., 500-1000 tokens) with **overlap** (10-20%) so context isn't cut mid-idea
- **Recursive character splitting** — split on paragraphs → sentences → words (LangChain default)
- **Semantic chunking** — split where meaning shifts (embedding similarity)
- **Contextual chunking** (JD lists this) — prepend document/section context to each chunk before embedding so an isolated chunk still makes sense (Anthropic's "Contextual Retrieval")
- **Structure-aware** — split by markdown headers, tables, code blocks

> **Interview line:** "Chunk size is a trade-off — too big dilutes the embedding and wastes context; too small loses meaning. I use recursive splitting with overlap, and contextual chunking for docs where a chunk is ambiguous without its section context."

---

## 3. Embeddings

Turn text into vectors capturing meaning; similar meaning → nearby vectors (cosine similarity).

**Models the JD names:**
- **OpenAI** `text-embedding-3-small/large` — easy, strong, API
- **BGE** (BAAI) — top open-source, self-hostable
- **E5** (Microsoft) — strong open-source
- **sentence-transformers** — the library; many models (all-MiniLM is a fast default)

**Know:** dimension (e.g., 384, 768, 1536), cosine similarity, that query and docs must use the SAME embedding model, and re-embedding cost when you change models (vector index lifecycle).

---

## 4. Vector databases (know several)

Store vectors + metadata, do fast approximate nearest-neighbor (ANN) search.

| DB | Note |
|----|------|
| **FAISS** | Library (Meta), in-memory, fast, self-managed — great for local/POC |
| **Chroma** | Simple, local/dev-friendly, open-source |
| **Pinecone** | Managed, serverless, production-popular |
| **Weaviate** | Open-source, hybrid search built in |
| **Qdrant** | Open-source, fast, good filtering |
| **Milvus** | Open-source, built for massive scale |
| **pgvector** | Postgres extension — vectors in your existing DB (great when you already run Postgres) |
| **Elasticsearch** | Full-text + vector, strong hybrid |

**ANN algorithm to name:** **HNSW** (Hierarchical Navigable Small World) — the common graph index for fast vector search. Trade recall vs speed.

**Metadata filtering:** filter by fields (date, source, tenant) alongside vector search — essential for multi-tenant/production.

---

## 5. Advanced retrieval (the JD lists these — know each)

### Hybrid search (dense + sparse)
Combine **dense** (embedding/semantic) + **sparse** (keyword — BM25/TF-IDF) retrieval. Dense catches meaning; sparse catches exact terms (IDs, names, codes). Merge with **RRF (Reciprocal Rank Fusion)**. Best of both.

### Re-ranking (cross-encoder)
First retrieve top-50 fast (bi-encoder), then **re-rank** them with a **cross-encoder** (e.g., Cohere Rerank, BGE-reranker) that reads query+chunk together for a precise relevance score → keep top-5. Big quality boost.
- **Bi-encoder:** embeds query and doc separately (fast, for retrieval)
- **Cross-encoder:** scores query+doc jointly (accurate, for re-ranking, slow → only on the shortlist)

### HyDE (Hypothetical Document Embeddings)
Have the LLM **generate a hypothetical answer** to the question first, embed THAT, and retrieve with it — often matches real docs better than the short question. JD lists it.

### Query transformation
- **Query expansion / rewriting** — reformulate the question for better recall
- **Multi-query** — generate several query variants, retrieve for each, merge
- **Step-back prompting** — ask a more general question first

### Other advanced RAG (bonus knowledge)
- **Parent-document / small-to-big:** retrieve small chunks but pass their larger parent for context
- **RAPTOR:** hierarchical summary tree of chunks
- **GraphRAG:** build a knowledge graph, retrieve over entities/relationships
- **Self-RAG / CRAG:** the model decides when to retrieve and grades retrieval quality

---

## 6. The full production RAG pipeline
```
1. Ingest & chunk (with metadata) 
2. Embed (batch) → upsert to vector DB
3. Query: (optional) rewrite/HyDE → embed
4. Hybrid retrieve (dense + BM25) top-k
5. Re-rank (cross-encoder) → top-n
6. Build prompt: system + retrieved context + question (with citations)
7. LLM answers, grounded, with sources
8. Evaluate + trace
```

---

## 7. Multi-LLM routing (JD lists this)
Route requests across providers (OpenAI, Claude, Azure OpenAI, Bedrock, Vertex) for:
- **Cost control:** cheap model for simple/classification, strong model for hard reasoning
- **Fallback:** if the primary is down/rate-limited, retry on another
- **Latency/quality:** pick per task
> "I implement a router with a fallback chain and per-model cost caps, sending classification to a mini model and complex synthesis to a frontier model." (You literally did model selection reasoning in your DHL system.)

---

## 8. RAG evaluation (senior signal — they'll ask)
- **Retrieval metrics:** context precision, context recall, hit rate, MRR
- **Generation metrics:** faithfulness/groundedness (is the answer supported by context?), answer relevance
- **Tools:** **RAGAS**, LangSmith/LangFuse evals, TruLens
- **The key idea:** measure retrieval and generation separately — a wrong answer is either "didn't retrieve the right chunk" or "retrieved it but answered wrong."

---

## 9. Common RAG failures & fixes
| Problem | Cause | Fix |
|---------|-------|-----|
| Right doc not retrieved | Bad chunking/embedding, semantic-only | Hybrid search, better chunking, HyDE |
| Retrieved but wrong answer | Weak prompt/grounding | Grounding instruction, re-ranking, cite sources |
| Hallucinated despite context | Model ignores context | "Answer ONLY from context; say 'not found' if absent" |
| Slow | Re-ranking everything, huge k | Retrieve 50 → re-rank → 5; cache; smaller k |
| Stale answers | Index not updated | Vector index lifecycle: re-ingest on change |
| Exact IDs/codes missed | Pure semantic search | Add sparse/keyword (hybrid) |

---

## The sentence that proves you get RAG
> "Production RAG for me is: contextual chunking with overlap, hybrid dense+sparse retrieval fused with RRF, cross-encoder re-ranking down to the top few, and a grounded prompt that cites sources and refuses when the answer isn't in context. I evaluate retrieval and generation separately with RAGAS-style metrics, manage the vector index lifecycle on updates, and route across models for cost and fallback."
