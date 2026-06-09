"""
Vector Databases — Definitions + Code + Outputs
=================================================
pip install chromadb pinecone-client qdrant-client sentence-transformers supabase anthropic

WHAT IS A VECTOR DATABASE?
  → A database optimized for storing and searching EMBEDDINGS (vectors)
  → Regular DB: find row WHERE id = 5  (exact match)
  → Vector DB : find the 3 rows most SIMILAR to this vector (approximate match)
  → Used in: semantic search, RAG, recommendation systems, image search

WHAT IS AN EMBEDDING?
  → A list of numbers (vector) that represents the MEANING of text/image/audio
  → Similar meaning → similar vectors → close in vector space
  → "cat" and "dog" → close vectors | "cat" and "democracy" → far vectors
  → all-MiniLM-L6-v2: text → 384-dimensional vector (fast, good quality)

HOW SIMILARITY SEARCH WORKS:
  → Store: embed all your documents → store vectors in DB
  → Query: embed the question → find top-k closest document vectors
  → "Closest" = cosine similarity or euclidean distance

VECTOR DB COMPARISON:
  ChromaDB   → local, free, easiest to start, no server needed
  Pinecone   → cloud-only, production-scale, managed, fastest query
  Qdrant     → self-host or cloud, advanced filtering, production-grade
  Supabase   → PostgreSQL + pgvector, full SQL + vectors together
"""

from sentence_transformers import SentenceTransformer
import numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")

docs = [
    "Python is a popular programming language for data science",
    "Machine learning uses algorithms to find patterns in data",
    "Deep learning is a subset of machine learning using neural networks",
    "FastAPI is a modern Python web framework for building APIs",
    "Docker helps containerize applications for easy deployment",
    "Transformers revolutionized NLP with self-attention mechanism",
    "RAG combines retrieval with generation for better answers",
    "Vector databases store embeddings for semantic search",
]
doc_embeddings = embedder.encode(docs, convert_to_numpy=True)
print(f"Embeddings shape: {doc_embeddings.shape}")   # (8, 384)


# ══════════════════════════════════════════════════════
# 1. CHROMADB — Local, Free, Easiest
# ══════════════════════════════════════════════════════
# WHAT IS CHROMADB?
#   → Open-source vector database — runs locally, no cloud needed
#   → In-memory: data lost when script ends (testing)
#   → Persistent: saved to disk (production local use)
#
# KEY CONCEPTS:
#   → Collection : like a table — groups related documents together
#   → Documents  : the original text (optional, but helpful for retrieval)
#   → Embeddings : the vectors (you can provide them or let Chroma embed)
#   → IDs        : unique string identifier for each document
#   → Metadata   : extra key-value info (source, date, category) for filtering
#
# OPERATIONS:
#   → collection.add()    : insert documents + embeddings
#   → collection.query()  : find similar documents
#   → collection.update() : update an existing document
#   → collection.delete() : remove documents by ID
#   → collection.count()  : how many documents stored
#
# DISTANCE METRICS:
#   → cosine   : angle between vectors (best for text — insensitive to length)
#   → l2       : euclidean distance
#   → ip       : inner product

print("\n" + "=" * 55)
print("1. CHROMADB — LOCAL VECTOR DB")
print("=" * 55)

import chromadb

client     = chromadb.Client()              # in-memory (for testing)
# Persistent (saves to disk):
# client = chromadb.PersistentClient(path="./chroma_db")

collection = client.create_collection(
    name     = "my_docs",
    metadata = {"hnsw:space": "cosine"}    # cosine similarity metric
)

collection.add(
    documents  = docs,
    embeddings = doc_embeddings.tolist(),
    ids        = [f"doc_{i}" for i in range(len(docs))],
    metadatas  = [{"source": "manual", "topic": "tech"} for _ in docs]
)
print(f"Added {collection.count()} documents")

# Semantic search — find most similar documents
query = "how to deploy machine learning models"
q_emb = embedder.encode([query]).tolist()

results = collection.query(
    query_embeddings = q_emb,
    n_results        = 3,
    include          = ["documents", "distances", "metadatas"]
)
print(f"\nQuery: '{query}'")
for doc, dist in zip(results["documents"][0], results["distances"][0]):
    similarity = 1 - dist   # cosine distance → cosine similarity
    print(f"  [{similarity:.3f}] {doc}")
# [0.612] Vector databases store embeddings for semantic search
# [0.578] Docker helps containerize applications for easy deployment
# [0.541] RAG combines retrieval with generation for better answers

# Filter by metadata — only return documents with source="manual"
filtered = collection.query(
    query_embeddings = q_emb,
    n_results        = 2,
    where            = {"source": "manual"},   # metadata filter
    include          = ["documents"]
)

# Update a document — re-embed and replace
collection.update(
    ids        = ["doc_0"],
    documents  = ["Python is the most popular language for AI and ML"],
    embeddings = embedder.encode(["Python is the most popular language for AI and ML"]).tolist()
)

collection.delete(ids=["doc_0"])   # remove by ID
print(f"After delete: {collection.count()} documents")


# ══════════════════════════════════════════════════════
# 2. PINECONE — Cloud, Production-Scale
# ══════════════════════════════════════════════════════
# WHAT IS PINECONE?
#   → Managed cloud vector database — no infrastructure to manage
#   → Scales to billions of vectors with millisecond query latency
#   → Organized into: Indexes (like tables) → Namespaces (like partitions)
#
# KEY CONCEPTS:
#   → Index     : stores vectors of a fixed dimension (must match your embedder)
#   → Namespace : partition within an index (e.g., "production", "staging")
#   → Upsert    : insert OR update — if ID exists, update; if not, insert
#   → Metadata  : key-value pairs stored with each vector for filtering
#
# PINECONE METADATA FILTERS:
#   → $eq  : equal to       → {"source": {"$eq": "manual"}}
#   → $ne  : not equal to
#   → $in  : in list        → {"category": {"$in": ["tech", "science"]}}
#   → $gt  : greater than

print("\n" + "=" * 55)
print("2. PINECONE — CLOUD VECTOR DB")
print("=" * 55)

import pinecone

PINECONE_API_KEY = "your-api-key-here"
PINECONE_ENV     = "gcp-starter"
INDEX_NAME       = "ml-docs"
DIMENSION        = 384   # all-MiniLM-L6-v2 produces 384-dim vectors

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        name      = INDEX_NAME,
        dimension = DIMENSION,
        metric    = "cosine",
        pod_type  = "p1.x1"
    )

index = pinecone.Index(INDEX_NAME)

# Upsert: (id, vector, metadata) tuples
vectors = [(f"doc_{i}", doc_embeddings[i].tolist(), {"text": doc, "source": "manual"})
           for i, doc in enumerate(docs)]
index.upsert(vectors=vectors, namespace="production")
print(f"Upserted {len(vectors)} vectors")

# Query — find top 3 similar vectors
q_emb   = embedder.encode(["deploying ML models"]).tolist()[0]
results = index.query(vector=q_emb, top_k=3, include_metadata=True,
                       namespace="production")
for match in results["matches"]:
    print(f"  [{match['score']:.3f}] {match['metadata']['text']}")

# Filtered query — only vectors with source="manual"
results_filtered = index.query(
    vector           = q_emb,
    top_k            = 3,
    filter           = {"source": {"$eq": "manual"}},
    include_metadata = True
)

index.delete(ids=["doc_0"], namespace="production")

stats = index.describe_index_stats()
print(f"Total vectors: {stats['total_vector_count']}")


# ══════════════════════════════════════════════════════
# 3. QDRANT — Self-Hosted or Cloud, Advanced Filtering
# ══════════════════════════════════════════════════════
# WHAT IS QDRANT?
#   → Open-source vector database — can self-host (Docker) or use cloud
#   → Excellent for complex filtering (AND/OR/NOT conditions)
#   → HNSW index for fast approximate nearest neighbor search
#
# KEY CONCEPTS:
#   → Collection  : like a table — name + vector config
#   → Points      : records = id + vector + payload (metadata)
#   → Payload     : key-value metadata stored with each point
#   → Filter      : FieldCondition for filtering by payload values
#
# DISTANCE METRICS:
#   → Distance.COSINE    : for text embeddings (most common)
#   → Distance.EUCLID    : for image/geometric data
#   → Distance.DOT       : for normalized vectors (similar to cosine)
#
# RUNNING QDRANT LOCALLY:
#   → docker run -p 6333:6333 qdrant/qdrant
#   → client = QdrantClient("localhost", port=6333)

print("\n" + "=" * 55)
print("3. QDRANT — SELF-HOSTED VECTOR DB")
print("=" * 55)

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)

client_qdrant = QdrantClient(":memory:")   # in-memory for demo
# Local server: QdrantClient("localhost", port=6333)
# Cloud:        QdrantClient(url="https://xxx.cloud.qdrant.io", api_key="xxx")

COLLECTION  = "tech_docs"
VECTOR_SIZE = 384

client_qdrant.create_collection(
    collection_name = COLLECTION,
    vectors_config  = VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
)

points = [
    PointStruct(
        id      = i,
        vector  = doc_embeddings[i].tolist(),
        payload = {"text": doc, "source": "manual", "topic": "tech"}
    )
    for i, doc in enumerate(docs)
]
client_qdrant.upsert(collection_name=COLLECTION, points=points)
print(f"Added {client_qdrant.get_collection(COLLECTION).points_count} points")

# Search
q_emb   = embedder.encode(["machine learning deployment"]).tolist()[0]
results = client_qdrant.search(
    collection_name = COLLECTION,
    query_vector    = q_emb,
    limit           = 3,
    with_payload    = True
)
for r in results:
    print(f"  [{r.score:.3f}] {r.payload['text']}")

# Filtered search — only points with source="manual"
results_filtered = client_qdrant.search(
    collection_name = COLLECTION,
    query_vector    = q_emb,
    query_filter    = Filter(
        must = [FieldCondition(key="source", match=MatchValue(value="manual"))]
    ),
    limit = 3
)

client_qdrant.delete(collection_name=COLLECTION, points_selector=[0])
print(f"After delete: {client_qdrant.get_collection(COLLECTION).points_count} points")


# ══════════════════════════════════════════════════════
# 4. SUPABASE — PostgreSQL + pgvector
# ══════════════════════════════════════════════════════
# WHAT IS SUPABASE?
#   → Open-source Firebase alternative — PostgreSQL database + auth + storage + realtime
#   → pgvector: PostgreSQL extension that adds vector similarity search to SQL
#   → Advantage: use SQL for everything — regular queries AND vector search together
#
# WHEN TO USE SUPABASE OVER DEDICATED VECTOR DBs:
#   → You already use PostgreSQL
#   → You need SQL joins (e.g., vector search + filter by user_id + join products table)
#   → You want one database for everything (no separate vector DB to maintain)
#
# SQL SETUP NEEDED (run in Supabase SQL editor):
#   CREATE EXTENSION IF NOT EXISTS vector;
#   CREATE TABLE documents (
#       id BIGSERIAL PRIMARY KEY, content TEXT,
#       embedding VECTOR(384), metadata JSONB
#   );
#   CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);
#
# OPERATORS:
#   → <=>  : cosine distance (most common for text)
#   → <->  : L2 (euclidean) distance
#   → <#>  : inner product distance

print("\n" + "=" * 55)
print("4. SUPABASE + pgvector")
print("=" * 55)

from supabase import create_client

SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-key"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# SQL setup (run once in Supabase dashboard):
SQL_SETUP = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id         BIGSERIAL PRIMARY KEY,
    content    TEXT,
    embedding  VECTOR(384),   -- matches all-MiniLM-L6-v2 output dim
    metadata   JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- IVFFlat index: approximate search, fast
-- lists=100: good for ~1M vectors
CREATE INDEX ON documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Similarity search function
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(384),
    match_threshold float,
    match_count int
)
RETURNS TABLE (id bigint, content text, similarity float)
LANGUAGE sql STABLE AS $$
    SELECT id, content,
           1 - (embedding <=> query_embedding) AS similarity
    FROM documents
    WHERE 1 - (embedding <=> query_embedding) > match_threshold
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;
"""

def insert_doc(content: str, metadata: dict = {}):
    embedding = embedder.encode([content])[0].tolist()
    return supabase.table("documents").insert({
        "content":   content,
        "embedding": embedding,
        "metadata":  metadata
    }).execute()

def semantic_search(query: str, threshold: float = 0.5, k: int = 3):
    q_emb = embedder.encode([query])[0].tolist()
    result = supabase.rpc("match_documents", {
        "query_embedding": q_emb,
        "match_threshold": threshold,
        "match_count":     k,
    }).execute()
    return result.data

# insert_doc("Python is great for ML", {"source": "manual"})
# results = semantic_search("machine learning frameworks")

print("  Supabase requires live credentials to run — see SQL_SETUP above")


# ══════════════════════════════════════════════════════
# 5. COMPARISON TABLE
# ══════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("5. VECTOR DB COMPARISON")
print("=" * 55)

print("""
┌─────────────┬──────────────┬─────────────┬────────────────┬──────────────┐
│             │ ChromaDB     │ Pinecone    │ Qdrant         │ Supabase     │
├─────────────┼──────────────┼─────────────┼────────────────┼──────────────┤
│ Hosting     │ Local/Cloud  │ Cloud only  │ Self/Cloud     │ Cloud        │
│ Price       │ Free         │ Free tier   │ Free tier      │ Free tier    │
│ Setup       │ Easiest      │ Easy        │ Easy           │ Moderate     │
│ Scale       │ Medium       │ Very Large  │ Large          │ Medium       │
│ SQL support │ No           │ No          │ No             │ YES (Postgres)│
│ Filtering   │ Basic        │ Advanced    │ Very Advanced  │ Full SQL     │
│ Best for    │ Prototyping  │ Production  │ Production     │ Full-stack   │
└─────────────┴──────────────┴─────────────┴────────────────┴──────────────┘
""")


# ══════════════════════════════════════════════════════
# 6. COMPLETE RAG PIPELINE WITH CHROMADB
# ══════════════════════════════════════════════════════
# WHAT IS RAG (Retrieval Augmented Generation)?
#   → Give an LLM access to YOUR documents — without retraining
#   → Step 1 (offline): embed all docs → store in vector DB
#   → Step 2 (online) : embed question → retrieve top-k similar docs
#   → Step 3 (online) : LLM reads retrieved docs + answers the question
#
# WHY RAG?
#   → LLM's training data has a cutoff date — doesn't know recent events
#   → LLM doesn't know your private documents
#   → Fine-tuning is expensive — RAG is cheaper and more flexible
#   → When to use fine-tuning instead: when you need the model to BEHAVE
#     differently (different style/tone/format), not just know more facts

print("\n" + "=" * 55)
print("6. COMPLETE RAG PIPELINE")
print("=" * 55)

import anthropic

def build_rag_pipeline(documents: list, emb_model):
    client_local = chromadb.Client()
    coll = client_local.create_collection("rag_docs", metadata={"hnsw:space": "cosine"})
    embeddings = emb_model.encode(documents).tolist()
    coll.add(
        documents  = documents,
        embeddings = embeddings,
        ids        = [f"id_{i}" for i in range(len(documents))]
    )
    return coll

def rag_query(question: str, collection, emb_model, top_k: int = 3) -> str:
    q_emb   = emb_model.encode([question]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=top_k,
                               include=["documents"])
    context = "\n".join(results["documents"][0])

    client   = anthropic.Anthropic()
    response = client.messages.create(
        model      = "claude-sonnet-4-6",
        max_tokens = 500,
        system     = "Answer using ONLY the context provided. Say 'I don't know' if not in context.",
        messages   = [{"role": "user",
                       "content": f"Context:\n{context}\n\nQuestion: {question}"}]
    )
    return response.content[0].text

# Build vector store from our docs
collection = build_rag_pipeline(docs, embedder)

# Query — answer will be grounded in our documents
answer = rag_query("What is RAG?", collection, embedder)
print(f"Question: What is RAG?")
print(f"Answer: {answer}")

print("\nAll done! ✓")
