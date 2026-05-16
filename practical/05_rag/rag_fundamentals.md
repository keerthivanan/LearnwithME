# 05 — Retrieval Augmented Generation (RAG)

> RAG is listed as a key skill in your JD. Know it end-to-end — architecture, components, evaluation, and trade-offs.

---

## 1. What is RAG and Why Does It Exist?

### The Problem with LLMs Alone
LLMs have **static knowledge** — they only know what was in their training data, up to their cutoff date.

Problems:
- Outdated information (no real-time data)
- Hallucinations (confidently wrong answers)
- No access to private/internal documents
- Context window limits (can't fit entire knowledge base)

### What RAG Does
RAG augments an LLM with a **retrieval system** that fetches relevant documents at inference time.

```
User Query
    ↓
[Retriever] → Search knowledge base → Retrieve relevant docs
    ↓
[Augment] → Combine query + retrieved docs into prompt
    ↓
[Generator (LLM)] → Generate answer grounded in retrieved docs
    ↓
Final Answer
```

---

## 2. RAG Architecture — Full Pipeline

```
INDEXING (Done once / periodically):
Documents → Chunking → Embedding → Vector Store

RETRIEVAL (Done per query):
Query → Embedding → Vector Search → Top-K Chunks

GENERATION:
Query + Top-K Chunks → LLM → Answer
```

---

## 3. Step-by-Step RAG Pipeline

### Step 1: Document Ingestion & Chunking
Split documents into smaller chunks (LLMs have context limits).

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_documents(documents)
```

**Chunking Strategies:**
| Strategy | Description | When to Use |
|----------|------------|------------|
| Fixed size | Split every N chars/tokens | Simple, fast |
| Recursive | Split by paragraph → sentence → word | General purpose |
| Semantic | Group semantically similar sentences | Higher quality retrieval |
| Document-based | Respect document structure (headers, sections) | Structured docs |

**Chunk Size Trade-off:**
- Too small → lacks context, incomplete answers
- Too large → noisy retrieval, exceeds context window
- Typical: 256–1024 tokens with 10–20% overlap

### Step 2: Embedding
Convert chunks to dense vectors that capture semantic meaning.

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5"
)
chunk_vectors = embeddings.embed_documents(chunks)
```

**Popular Embedding Models:**
| Model | Dims | Notes |
|-------|------|-------|
| text-embedding-ada-002 | 1536 | OpenAI, strong baseline |
| BAAI/bge-large-en-v1.5 | 1024 | Best open-source for English |
| sentence-transformers/all-MiniLM | 384 | Fast, small |
| nomic-embed-text | 768 | Long context open-source |
| E5-large | 1024 | Strong multilingual |

### Step 3: Vector Store (Index)
Store embeddings for fast similarity search.

```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
```

**Popular Vector Databases:**
| Database | Type | Notes |
|----------|------|-------|
| FAISS | In-memory / Local | Facebook, fast, no server |
| ChromaDB | Local / Hosted | Easy to use, Python-native |
| Pinecone | Managed cloud | Production-ready, serverless |
| Weaviate | Open-source | GraphQL API, hybrid search |
| Qdrant | Open-source | Fast, Rust-based |
| pgvector | PostgreSQL extension | Stay in existing DB |

### Step 4: Retrieval (Query Time)
```python
query = "What is the refund policy?"
query_vector = embeddings.embed_query(query)
results = vectorstore.similarity_search(query, k=5)
```

**Similarity Metrics:**
| Metric | Formula | Notes |
|--------|---------|-------|
| Cosine Similarity | cos(θ) = A·B / (|A||B|) | Direction-based, most common |
| Dot Product | A·B | Fast, needs normalized vectors |
| L2 (Euclidean) | √Σ(A-B)² | Distance-based |

### Step 5: Generation
```python
context = "\n\n".join([doc.page_content for doc in results])

prompt = f"""Use the following context to answer the question.
Context: {context}
Question: {query}
Answer:"""

response = llm.generate(prompt)
```

---

## 4. Vector Search — How it Works

### The Problem: Exact Search is Slow
For 1M vectors of 1024 dimensions: exact search = O(n×d) = too slow.

### ANN (Approximate Nearest Neighbor) Search
Trade tiny accuracy loss for huge speed gains.

**HNSW (Hierarchical Navigable Small World)**
- Graph-based ANN
- Very fast search, high recall
- Used in FAISS, Qdrant, Weaviate

**IVF (Inverted File Index)**
- Cluster vectors, search only nearby clusters
- Used in FAISS

---

## 5. Retrieval Strategies

### Dense Retrieval (Semantic Search)
Use embedding vectors for semantic similarity.
- Good: captures meaning, handles paraphrases
- Bad: misses exact keyword matches

### Sparse Retrieval (BM25 / TF-IDF)
Traditional keyword-based search (BM25 is best).
- Good: exact keyword matching
- Bad: misses synonyms, semantics

### Hybrid Retrieval
Combine dense + sparse results (best of both worlds).
```
final_score = α * dense_score + (1-α) * sparse_score
```

### Re-ranking
Use a cross-encoder to re-rank the top-K retrieved chunks:
1. Retrieve top-50 with fast bi-encoder (vector search)
2. Re-rank top-50 with slower but more accurate cross-encoder
3. Pass top-5 to LLM

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = reranker.predict([(query, doc) for doc in retrieved_docs])
reranked = [doc for _, doc in sorted(zip(scores, retrieved_docs), reverse=True)]
```

---

## 6. Advanced RAG Techniques

### Query Rewriting
Rewrite the user query before retrieval to improve results:
- Expand abbreviations
- Break down complex questions
- Generate multiple query variants (Multi-Query RAG)

```python
# Multi-Query: generate 3 query variants, retrieve for each
queries = llm.generate_queries(original_query, n=3)
all_results = [retriever.retrieve(q) for q in queries]
merged = deduplicate(all_results)
```

### HyDE — Hypothetical Document Embeddings
1. Ask LLM to generate a *hypothetical* document that would answer the query
2. Embed the hypothetical document
3. Use that embedding for retrieval (often more similar to real answers)

### Parent Document Retrieval
- Index small chunks for precise retrieval
- But retrieve the full parent document for context
- Best of both: precision + context

### Self-RAG
The LLM itself decides when to retrieve (not always retrieval) and critiques retrieved documents.

---

## 7. RAG Evaluation

### Key Metrics
| Metric | Measures | Tool |
|--------|---------|------|
| Context Precision | Are retrieved docs relevant? | RAGAS |
| Context Recall | Were all relevant docs retrieved? | RAGAS |
| Faithfulness | Does answer stay grounded in context? | RAGAS |
| Answer Relevance | Does answer address the question? | RAGAS |

### RAGAS Framework
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)
```

---

## 8. RAG vs Fine-Tuning

| Dimension | RAG | Fine-Tuning |
|-----------|-----|------------|
| Knowledge update | Easy (update vector DB) | Need to retrain |
| Private data | Yes, naturally | Yes, but baked in |
| Cost | Low | Higher (training) |
| Hallucination | Reduced (grounded) | Can still hallucinate |
| Response style | Unchanged | Can change behavior |
| Latency | +retrieval overhead | Same as base model |
| When to use | Dynamic/external knowledge | Behavior/style change |

**Use both when possible** — fine-tune for style/reasoning, RAG for knowledge.

---

## 9. RAG Implementation Example (LangChain)

```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# 1. Embeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# 2. Vector store
vectorstore = FAISS.load_local("faiss_index", embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. LLM
llm = Ollama(model="llama3")

# 4. RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

result = qa_chain({"query": "What is the return policy?"})
print(result["result"])
```

---

## 10. Interview Questions — RAG

**Q: What is RAG and why is it used?**
> RAG (Retrieval Augmented Generation) augments an LLM with an external retrieval system. When a query comes in, relevant documents are retrieved from a knowledge base and injected into the LLM's context before generation. It's used to ground responses in specific documents, handle knowledge that wasn't in training data, and reduce hallucinations.

**Q: What is a vector database?**
> A database optimized for storing and searching high-dimensional vectors (embeddings). It uses approximate nearest neighbor algorithms (like HNSW) to find semantically similar vectors quickly. Examples: Pinecone, Qdrant, Weaviate, ChromaDB, FAISS.

**Q: What is the difference between dense and sparse retrieval?**
> Dense retrieval uses neural embeddings to find semantically similar documents — it captures meaning but may miss exact keywords. Sparse retrieval (BM25) uses term frequency to find keyword matches — exact but misses semantic similarity. Hybrid search combines both.

**Q: How do you evaluate a RAG system?**
> Using metrics like: Faithfulness (does the answer stay grounded in retrieved context?), Context Precision (are retrieved docs relevant?), Context Recall (were all relevant docs retrieved?), and Answer Relevance (does it answer the question?). RAGAS is a popular framework for this.

**Q: What is the trade-off in chunk size?**
> Smaller chunks give more precise retrieval but may lack context for answering. Larger chunks contain more context but may retrieve noisy, off-topic content and hit context window limits. Typical sweet spot: 256–512 tokens with ~10% overlap.

**Q: When would you choose RAG over fine-tuning?**
> Choose RAG when: you need access to frequently updated or external data, data is too large to fit in context but too dynamic to bake into weights, or you want traceable answers with source citations. Choose fine-tuning when: you need to change model behavior/tone, teach domain-specific reasoning patterns, or improve performance on a narrow task type.

---

## Quick Reference Cheat Sheet

```
RAG Pipeline:    Ingest → Chunk → Embed → Index → Retrieve → Generate
Vector DB:       FAISS (local), Pinecone (cloud), ChromaDB (easy), Qdrant (fast)
Embeddings:      BAAI/bge (open), text-embedding-ada-002 (OpenAI)
Retrieval:       Dense (semantic) + Sparse (BM25) + Hybrid + Reranking
Evaluation:      RAGAS: Faithfulness, Precision, Recall, Relevance
RAG vs FT:       RAG for dynamic knowledge, FT for behavior change
```
