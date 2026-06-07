"""
Vector Databases — ChromaDB, Pinecone, Qdrant, Supabase
=========================================================
pip install chromadb pinecone-client qdrant-client sentence-transformers supabase
"""

from sentence_transformers import SentenceTransformer
import numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Sample documents
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

# ════════════════════════════════════════════
# 1. CHROMADB (Local, Free, Easy)
# ════════════════════════════════════════════
print("1. CHROMADB")

import chromadb
from chromadb.config import Settings

# In-memory (for testing)
client = chromadb.Client()

# Persistent (saves to disk)
# client = chromadb.PersistentClient(path="./chroma_db")

# Create collection
collection = client.create_collection(
    name="my_docs",
    metadata={"hnsw:space": "cosine"}  # distance metric
)

# Add documents
collection.add(
    documents=docs,
    embeddings=doc_embeddings.tolist(),
    ids=[f"doc_{i}" for i in range(len(docs))],
    metadatas=[{"source": "manual", "topic": "tech"} for _ in docs]
)
print(f"  Added {collection.count()} documents")

# Query
query = "how to deploy machine learning models"
q_emb = embedder.encode([query]).tolist()

results = collection.query(
    query_embeddings=q_emb,
    n_results=3,
    include=["documents", "distances", "metadatas"]
)

print(f"\n  Query: '{query}'")
for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"  [{1-dist:.3f}] {doc}")

# Filter by metadata
filtered = collection.query(
    query_embeddings=q_emb,
    n_results=2,
    where={"source": "manual"},
    include=["documents"]
)

# Update document
collection.update(
    ids=["doc_0"],
    documents=["Python is the most popular language for AI and ML"],
    embeddings=embedder.encode(["Python is the most popular language for AI and ML"]).tolist()
)

# Delete document
collection.delete(ids=["doc_0"])
print(f"  After delete: {collection.count()} documents")

# ════════════════════════════════════════════
# 2. PINECONE (Cloud, Production)
# ════════════════════════════════════════════
print("\n2. PINECONE")

import pinecone

PINECONE_API_KEY = "your-api-key-here"
PINECONE_ENV     = "gcp-starter"

# Initialize
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

INDEX_NAME = "ml-docs"
DIMENSION  = 384   # all-MiniLM-L6-v2 output dim

# Create index (run once)
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        pod_type="p1.x1"
    )

index = pinecone.Index(INDEX_NAME)

# Upsert vectors
vectors = [(f"doc_{i}", doc_embeddings[i].tolist(), {"text": doc, "source": "manual"})
           for i, doc in enumerate(docs)]
index.upsert(vectors=vectors, namespace="production")
print(f"  Upserted {len(vectors)} vectors")

# Query
q_emb = embedder.encode(["deploying ML models"]).tolist()[0]
results = index.query(
    vector=q_emb,
    top_k=3,
    include_metadata=True,
    namespace="production"
)
for match in results["matches"]:
    print(f"  [{match['score']:.3f}] {match['metadata']['text']}")

# Filtered query
results_filtered = index.query(
    vector=q_emb,
    top_k=3,
    filter={"source": {"$eq": "manual"}},
    include_metadata=True
)

# Delete
index.delete(ids=["doc_0"], namespace="production")

# Stats
stats = index.describe_index_stats()
print(f"  Total vectors: {stats['total_vector_count']}")

# ════════════════════════════════════════════
# 3. QDRANT (Self-hosted or Cloud)
# ════════════════════════════════════════════
print("\n3. QDRANT")

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)

# In-memory
client_qdrant = QdrantClient(":memory:")

# Local server: QdrantClient("localhost", port=6333)
# Cloud:        QdrantClient(url="https://xxx.cloud.qdrant.io", api_key="xxx")

COLLECTION = "tech_docs"
VECTOR_SIZE = 384

# Create collection
client_qdrant.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
)

# Upsert points
points = [
    PointStruct(
        id=i,
        vector=doc_embeddings[i].tolist(),
        payload={"text": doc, "source": "manual", "topic": "tech"}
    )
    for i, doc in enumerate(docs)
]
client_qdrant.upsert(collection_name=COLLECTION, points=points)
print(f"  Added {client_qdrant.get_collection(COLLECTION).points_count} points")

# Search
q_emb = embedder.encode(["machine learning deployment"]).tolist()[0]
results = client_qdrant.search(
    collection_name=COLLECTION,
    query_vector=q_emb,
    limit=3,
    with_payload=True
)
for r in results:
    print(f"  [{r.score:.3f}] {r.payload['text']}")

# Filtered search
results_filtered = client_qdrant.search(
    collection_name=COLLECTION,
    query_vector=q_emb,
    query_filter=Filter(
        must=[FieldCondition(key="source", match=MatchValue(value="manual"))]
    ),
    limit=3
)

# Delete point
client_qdrant.delete(
    collection_name=COLLECTION,
    points_selector=[0]
)

# ════════════════════════════════════════════
# 4. SUPABASE (PostgreSQL + pgvector)
# ════════════════════════════════════════════
print("\n4. SUPABASE + pgvector")

from supabase import create_client

SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-key"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# First, create table in Supabase SQL editor:
"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id          BIGSERIAL PRIMARY KEY,
    content     TEXT,
    embedding   VECTOR(384),
    metadata    JSONB,
    created_at  TIMESTAMP DEFAULT NOW()
);

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
LANGUAGE sql STABLE
AS $$
    SELECT id, content,
           1 - (embedding <=> query_embedding) AS similarity
    FROM documents
    WHERE 1 - (embedding <=> query_embedding) > match_threshold
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;
"""

# Insert documents
def insert_doc(content: str, metadata: dict = {}):
    embedding = embedder.encode([content])[0].tolist()
    return supabase.table("documents").insert({
        "content":   content,
        "embedding": embedding,
        "metadata":  metadata
    }).execute()

# insert_doc("Python is great for ML", {"source": "manual"})

# Search using RPC function
def semantic_search(query: str, threshold: float = 0.5, k: int = 3):
    q_emb = embedder.encode([query])[0].tolist()
    result = supabase.rpc("match_documents", {
        "query_embedding": q_emb,
        "match_threshold":  threshold,
        "match_count":      k,
    }).execute()
    return result.data

# results = semantic_search("machine learning frameworks")
# for r in results:
#     print(f"  [{r['similarity']:.3f}] {r['content']}")

print("  (Supabase requires live credentials to run)")

# ════════════════════════════════════════════
# 5. VECTOR DB COMPARISON
# ════════════════════════════════════════════
print("\n5. COMPARISON TABLE")

comparison = """
┌─────────────┬──────────────┬─────────────┬────────────────┬──────────────┐
│             │ ChromaDB     │ Pinecone    │ Qdrant         │ Supabase     │
├─────────────┼──────────────┼─────────────┼────────────────┼──────────────┤
│ Hosting     │ Local/Cloud  │ Cloud only  │ Self/Cloud     │ Cloud        │
│ Price       │ Free         │ Free tier   │ Free tier      │ Free tier    │
│ Setup       │ Easiest      │ Easy        │ Easy           │ Moderate     │
│ Scale       │ Medium       │ Large       │ Large          │ Medium       │
│ SQL support │ No           │ No          │ No             │ YES (Postgres)│
│ Best for    │ Prototyping  │ Production  │ Production     │ Full-stack   │
│ Filter      │ Basic        │ Advanced    │ Advanced       │ SQL filters  │
└─────────────┴──────────────┴─────────────┴────────────────┴──────────────┘
"""
print(comparison)

# ════════════════════════════════════════════
# 6. RAG WITH CHROMADB (Production Pattern)
# ════════════════════════════════════════════
print("6. COMPLETE RAG PIPELINE")

import anthropic

def build_rag_pipeline(docs: list, embedder):
    client_local = chromadb.Client()
    coll = client_local.create_collection("rag_docs", metadata={"hnsw:space": "cosine"})
    embeddings = embedder.encode(docs).tolist()
    coll.add(documents=docs, embeddings=embeddings,
             ids=[f"id_{i}" for i in range(len(docs))])
    return coll

def rag_query(question: str, collection, embedder, top_k: int = 3) -> str:
    q_emb   = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=top_k,
                               include=["documents"])
    context = "\n".join(results["documents"][0])

    client  = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system="Answer using ONLY the context. Say 'I don't know' if not in context.",
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }]
    )
    return response.content[0].text

# Build and query
collection = build_rag_pipeline(docs, embedder)
answer = rag_query("What is RAG?", collection, embedder)
print(f"  Answer: {answer}")

print("\nAll done! ✓")
