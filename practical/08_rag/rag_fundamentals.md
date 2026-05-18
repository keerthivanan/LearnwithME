# 05 — Retrieval Augmented Generation (RAG)

> RAG is listed as a key skill in your JD. Know it end-to-end — architecture, components, evaluation, and trade-offs.

---

## 1. What is RAG and Why Does It Exist?

### The Problem with LLMs Alone

**What it is:** LLMs have a fundamental limitation — their knowledge is frozen at training time. They know everything in their training data and nothing that happened afterward. RAG is the standard production fix for this.

**Analogy:** Imagine a brilliant consultant who studied everything published before 2023, but then went into a sealed room with no internet. They are incredibly knowledgeable about the past, but cannot answer questions about recent events, your company's internal data, or anything private. RAG is the system that lets you slip relevant documents under the door before they answer.

Problems with LLMs alone:
- Outdated information (no real-time data — the training cutoff is months in the past)
- Hallucinations (confidently wrong answers when the LLM fills knowledge gaps)
- No access to private/internal documents (confidential data was never in training)
- Context window limits (you can't fit an entire knowledge base into one prompt)

### What RAG Does
RAG augments an LLM with a **retrieval system** that fetches relevant documents at inference time and injects them into the prompt. The LLM then generates its answer grounded in those specific documents.

```
User Query
    ↓
[Retriever] → Search knowledge base → Retrieve relevant docs
    ↓
[Augment] → Combine query + retrieved docs into a prompt
    ↓
[Generator (LLM)] → Generate answer grounded in retrieved docs
    ↓
Final Answer (with citations)
```

---

## 2. RAG Architecture — Full Pipeline

**What it is:** RAG has two distinct phases. Indexing happens once (or periodically when documents change). Retrieval happens for every user query.

```
INDEXING (Done once / periodically):
Documents → Chunking → Embedding → Vector Store

RETRIEVAL (Done per query):
Query → Embedding → Vector Search → Top-K Chunks

GENERATION:
Query + Top-K Chunks → LLM → Answer
```

**WHY:** Separating indexing from retrieval is the key efficiency insight. You do the expensive embedding work once during indexing. At query time, you only need to embed the (short) query and do a fast ANN search — no re-embedding documents.

---

## 3. Step-by-Step RAG Pipeline

### Step 1: Document Ingestion & Chunking

**What it is:** Split large documents into smaller pieces (chunks) that fit within the embedding model's context limit and are granular enough for precise retrieval. This is one of the most important (and most underestimated) decisions in a RAG system.

**Analogy:** You cannot search a library by scanning every book front to back. You split each book into chapters, then sections, then paragraphs — each with a clear label. When searching, you look for the right paragraph, not the right book. Chunking is creating those searchable paragraphs.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter  # LangChain's smart text splitter

# Create a splitter that tries to split on paragraph boundaries, then sentence, then word
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,                            # target maximum 500 characters per chunk
    chunk_overlap=50,                          # include 50 characters from previous chunk (for context continuity)
    separators=["\n\n", "\n", ".", " "]        # try splitting on paragraph → line → sentence → word (in order)
)
chunks = splitter.split_documents(documents)  # returns list of Document objects, each one chunk
```

**WHY:** `chunk_overlap=50` is important. When you split a document, the sentence at the boundary between two chunks loses half its context. Overlap ensures the boundary sentences appear in both chunks, so retrieval always finds the full context.

**Chunking Strategies:**
| Strategy | Description | When to Use |
|----------|------------|------------|
| Fixed size | Split every N chars/tokens | Simple, fast, no structure needed |
| Recursive | Split by paragraph → sentence → word | General purpose (most common) |
| Semantic | Group semantically similar sentences | Higher quality retrieval (more compute) |
| Document-based | Respect document structure (headers, sections) | Structured docs like PDFs, HTML |

**Chunk Size Trade-off:**
- Too small → chunk lacks context, retrieved answer is incomplete ("The answer is 42." — but 42 what?)
- Too large → retrieval is noisy (chunk about 10 topics, only 1 is relevant)
- Sweet spot: 256–1,024 tokens with 10–20% overlap

### Step 2: Embedding

**What it is:** Convert each text chunk into a dense numerical vector that captures the semantic meaning of that text. Semantically similar texts produce similar vectors, enabling similarity-based search.

**Analogy:** Think of each chunk being pinned to a giant map. Chunks about similar topics are pinned near each other. When you search, you find your "location" on the map and look at what is pinned nearby.

```python
from langchain_huggingface import HuggingFaceEmbeddings  # HuggingFace embedding wrapper for LangChain

# Load BAAI/bge-large — best open-source English embedding model as of 2024
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5"         # 1024-dimensional embeddings, strong retrieval performance
)

# embed_documents() converts a list of text strings to a list of vectors
chunk_vectors = embeddings.embed_documents(chunks)  # returns list of 1024-dim float vectors
```

**Popular Embedding Models:**
| Model | Dims | Notes |
|-------|------|-------|
| text-embedding-ada-002 | 1536 | OpenAI, strong baseline, API-only |
| BAAI/bge-large-en-v1.5 | 1024 | Best open-source for English |
| sentence-transformers/all-MiniLM | 384 | Fast and small — use when speed matters |
| nomic-embed-text | 768 | Long context, fully open-source |
| E5-large | 1024 | Strong multilingual performance |

### Step 3: Vector Store (Index)

**What it is:** A database specifically designed to store and search high-dimensional embedding vectors efficiently. Unlike a regular database that searches by exact match, a vector store searches by *similarity* — finding the vectors most similar to your query vector.

```python
from langchain_community.vectorstores import FAISS  # Facebook AI Similarity Search

# Create vector store from document chunks — this embeds all chunks and builds the HNSW index
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save the index to disk so you don't have to rebuild it every time
vectorstore.save_local("faiss_index")  # saves to local folder called "faiss_index"
```

**Popular Vector Databases:**
| Database | Type | Notes |
|----------|------|-------|
| FAISS | In-memory / Local | Facebook, fast, no server needed — great for development |
| ChromaDB | Local / Hosted | Easy to use, Python-native, good for prototyping |
| Pinecone | Managed cloud | Production-ready, serverless, fully managed |
| Weaviate | Open-source | GraphQL API, built-in hybrid search |
| Qdrant | Open-source | Fast, Rust-based, excellent filtering |
| pgvector | PostgreSQL extension | Stay in your existing Postgres database |

**WHY choosing the right vector DB matters:** FAISS is fine for prototypes but has no built-in metadata filtering and no server. Pinecone handles 100M+ vectors with SLA guarantees but costs money. For production with complex filtering requirements, Qdrant or Weaviate are better choices.

### Step 4: Retrieval (Query Time)

**What it is:** At query time, embed the user's question the same way you embedded your documents, then find the K most similar document vectors using ANN (Approximate Nearest Neighbor) search.

```python
query = "What is the refund policy?"               # the user's question
query_vector = embeddings.embed_query(query)        # embed the query — same model as documents
results = vectorstore.similarity_search(query, k=5) # find 5 most similar chunks by vector similarity
```

**Similarity Metrics:**
| Metric | Formula | Notes |
|--------|---------|-------|
| Cosine Similarity | cos(θ) = A·B / (\|A\|\|B\|) | Direction-based — most common, scale-invariant |
| Dot Product | A·B | Fast — but requires normalized vectors to equal cosine |
| L2 (Euclidean) | √Σ(A-B)² | Distance-based — less common for embeddings |

**WHY cosine similarity is standard:** Cosine similarity only cares about the *direction* of vectors, not their magnitude. This means a short tweet and a long article about the same topic can have high cosine similarity — which is what we want. The "aboutness" is the same even though the lengths differ.

### Step 5: Generation

**What it is:** Take the retrieved documents and the original question, format them into a prompt, and send to the LLM for answer generation. The LLM is now "grounded" — it has real evidence to work from instead of relying on memorized knowledge.

```python
# Join retrieved chunks into a single context string, separated by blank lines
context = "\n\n".join([doc.page_content for doc in results])

# Build the prompt — put the context BEFORE the question so the LLM reads evidence first
prompt = f"""Use the following context to answer the question.

Context: {context}

Question: {query}
Answer:"""

response = llm.generate(prompt)  # LLM generates the answer grounded in the provided context
```

**WHY putting context before question:** Research shows LLMs attend more strongly to information that appears earlier in the prompt. Putting context first ensures the LLM reads all the evidence before formulating its answer.

---

## 4. Vector Search — How it Works

### The Problem: Exact Search is Slow

**What it is:** If you have 1 million document vectors and need to find the 5 most similar to a query, the naive approach is to compute similarity with all 1 million vectors. This is too slow for production.

For 1 million vectors of 1,024 dimensions: exact search = O(n × d) = too slow.

Concrete math: 1M × 1024 × 2 operations = 2 billion operations per query → ~0.5 seconds on GPU.

### ANN (Approximate Nearest Neighbor) Search

**What it is:** ANN algorithms find the *approximate* nearest neighbors instead of the *exact* nearest neighbors. They miss a tiny percentage of results in exchange for 100–1,000× speedup.

**Analogy:** Instead of checking every item in a warehouse to find the closest one to the door, you first identify which aisle it is in (coarse search), then look through just that aisle (fine search). You might occasionally miss the closest item by a tiny margin, but you find a good answer 100× faster.

Trade tiny accuracy loss for huge speed gains.

**HNSW (Hierarchical Navigable Small World)**
- Graph-based ANN algorithm
- Very fast search, high recall (~98% at same speed as exact search at 10%)
- Used in FAISS, Qdrant, Weaviate
- O(log n) search time

**IVF (Inverted File Index)**
- Cluster vectors first (K-Means), then search only nearby clusters
- Used in FAISS
- Configurable speed vs accuracy trade-off

---

## 5. Retrieval Strategies

### Dense Retrieval (Semantic Search)

**What it is:** Use embedding vectors to find semantically similar documents. Good for capturing meaning, synonyms, and paraphrases.

Use embedding vectors for semantic similarity.
- **Good:** captures meaning, handles paraphrases, works across different word choices
- **Bad:** misses exact keyword matches (searching "GPT-4" won't find "GPT4" or "gpt 4")

### Sparse Retrieval (BM25 / TF-IDF)

**What it is:** Traditional keyword-based search. Scores documents by how often query terms appear (BM25 is the best sparse retrieval algorithm, used in Elasticsearch).

Traditional keyword-based search.
- **Good:** exact keyword matching — finds "GPT-4" only in documents containing "GPT-4"
- **Bad:** misses synonyms, paraphrases, semantically similar content with different words

### Hybrid Retrieval

**What it is:** Combine dense (semantic) and sparse (keyword) scores for the best of both worlds. The combination consistently outperforms either approach alone on most benchmarks.

Combine dense + sparse results:
```
final_score = α × dense_score + (1-α) × sparse_score   # weighted combination; α=0.5 is common default
```

**WHY:** Dense retrieval is great for "find me documents about X concept" but bad for "find the exact document mentioning error code ABC-123." Sparse retrieval is great for exact terms but bad for "find me documents about the same topic even with different words." Combining them covers both scenarios.

### Re-ranking

**What it is:** A two-stage retrieval strategy. First, use fast vector search to get the top 50 candidates. Then, use a slower but more accurate cross-encoder to reorder those 50 by true relevance. Only the top 5–10 re-ranked results are sent to the LLM.

Use a cross-encoder to re-rank the top-K retrieved chunks:
1. Retrieve top-50 with fast bi-encoder (vector search) — milliseconds
2. Re-rank top-50 with slower but more accurate cross-encoder — ~100ms
3. Pass top-5 to LLM — only the best results

```python
from sentence_transformers import CrossEncoder  # cross-encoder for pairwise relevance scoring

# Load a cross-encoder trained specifically for MS-MARCO passage ranking
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Score every (query, retrieved_doc) pair — cross-encoder reads both together
scores = reranker.predict([(query, doc) for doc in retrieved_docs])

# Sort documents by their re-ranking score (highest first)
reranked = [doc for _, doc in sorted(zip(scores, retrieved_docs), reverse=True)]
```

**WHY:** Cross-encoders are slow (they process query + document together, can't pre-compute) but they are much more accurate than bi-encoders at ranking. You can't afford to run a cross-encoder on 1 million documents, but you can afford to run it on 50 candidates. This two-stage approach gets you both speed and accuracy.

---

## 6. Advanced RAG Techniques

### Query Rewriting

**What it is:** Transform the user's raw query into a better form before retrieval. Users ask vague or ambiguous questions; query rewriting makes them more retrievable.

Rewrite the user query before retrieval to improve results:
- Expand abbreviations ("LLM" → "Large Language Model")
- Break down complex questions into simpler sub-questions
- Generate multiple query variants (Multi-Query RAG)

```python
# Multi-Query RAG: generate 3 different versions of the same question
# Each version may match different relevant documents in the index
queries = llm.generate_queries(original_query, n=3)     # ask LLM to rephrase the query 3 ways
all_results = [retriever.retrieve(q) for q in queries]  # retrieve for each version
merged = deduplicate(all_results)                        # merge and deduplicate the combined results
```

**WHY:** The user's query is written in their vocabulary. The documents were written by experts in domain vocabulary. These vocabularies may not overlap well. Multi-query expands the vocabulary coverage — if one rephrasing hits the document's exact terminology, retrieval succeeds.

### HyDE — Hypothetical Document Embeddings

**What it is:** Instead of embedding the query directly, ask the LLM to generate a "hypothetical document" that would answer the question, then embed THAT. Hypothetical documents tend to be more similar to real answer documents than a question is.

1. Ask LLM to generate a *hypothetical* document that would answer the query
2. Embed the hypothetical document (not the original query)
3. Use that embedding for vector search — it is often more similar to real answer documents

**WHY:** A question and its answer have different linguistic styles. "What is the capital of France?" and "Paris is the capital of France" do not have identical embeddings. But a hypothetical answer and the real answer are linguistically similar (both state facts), so they embed closer together. HyDE bridges the question-answer embedding gap.

### Parent Document Retrieval

**What it is:** Use small chunks for precise, targeted retrieval. But once you find a matching chunk, retrieve the full parent document (or parent section) for the LLM. You get precise matching AND rich context.

- Index small chunks (100–200 tokens) for precise, targeted retrieval
- When a match is found, retrieve the full parent document (500–2,000 tokens)
- Best of both: high precision retrieval + enough context for accurate answers

**WHY:** If you chunk too small, retrieved chunks lack context. If you chunk too large, you retrieve noisy, multi-topic content. Parent document retrieval lets you have both: retrieve precisely, then expand to the full context.

### Self-RAG

**What it is:** Train the LLM to decide when retrieval is needed, evaluate retrieved documents for relevance, and critique its own answers for faithfulness. See the advanced RAG file for full details.

---

## 7. RAG Evaluation

**What it is:** How do you measure whether your RAG system is actually good? You need to evaluate both the retrieval step and the generation step separately, because they can fail independently.

### Key Metrics
| Metric | Measures | Tool |
|--------|---------|------|
| Context Precision | Are retrieved docs relevant to the query? | RAGAS |
| Context Recall | Were ALL relevant docs retrieved? | RAGAS |
| Faithfulness | Does the answer stay grounded in the retrieved context? | RAGAS |
| Answer Relevance | Does the answer actually address the question? | RAGAS |

### RAGAS Framework
**What it is:** RAGAS (Retrieval Augmented Generation Assessment) is an open-source framework for automated RAG evaluation using LLM-as-judge metrics.

```python
from ragas import evaluate                                      # main evaluation function
from ragas.metrics import faithfulness, answer_relevancy, context_precision  # specific metrics

# Run evaluation on a test dataset
result = evaluate(
    dataset,                                                   # your test Q&A dataset
    metrics=[faithfulness, answer_relevancy, context_precision] # which metrics to compute
)
# Returns a dict with scores for each metric, e.g., {'faithfulness': 0.87, 'answer_relevancy': 0.92}
```

**WHY:** RAGAS uses an LLM (usually GPT-4) to judge whether the retrieved context supports the answer (faithfulness), whether the right context was retrieved (context metrics), and whether the answer is relevant. This automated evaluation is fast enough to run on every deployment.

---

## BM25 — The Keyword Search Behind Hybrid RAG

**What it is:** BM25 (Best Match 25) is the gold-standard keyword search algorithm, used internally by Elasticsearch and OpenSearch. It improves on simple word counting by handling document length and diminishing returns from repeated terms.

BM25 = Best Match 25. The algorithm behind Elasticsearch, OpenSearch, and the sparse component of hybrid RAG.

Formula:
```
score(D, Q) = Σ IDF(qi) × [f(qi,D) × (k1+1)] / [f(qi,D) + k1×(1 - b + b×|D|/avgdl)]
```

Breaking down each part:
- `f(qi, D)` = how many times query term qi appears in document D (raw term frequency)
- `IDF(qi)` = log((N - n(qi) + 0.5) / (n(qi) + 0.5)) — rare words get a much higher score than common words
- `|D|/avgdl` = document length normalization — longer documents are penalized to avoid length bias
- `k1=1.5` controls term frequency saturation (double occurrences do NOT double the score)
- `b=0.75` controls how strongly document length is penalized

**WHY IDF is the key insight:** A word that appears in every document ("the", "is", "a") tells you nothing about what makes a document relevant — it gets near-zero IDF. A word that appears in only 3 out of 1 million documents is highly discriminative — any document containing it is almost certainly specifically about that topic, so it gets a massive IDF boost.

Why BM25 is still used in production alongside neural embeddings:
- Exact keyword matching (neural embeddings miss "GPT-4o" vs "GPT-4" differences)
- No inference needed — pure statistical algorithm, extremely fast
- Works well for technical queries with specific terms (model names, error codes, IDs)
- Built into Elasticsearch/OpenSearch — already in your stack

Python implementation:
```python
from rank_bm25 import BM25Okapi  # pure Python BM25 implementation

# Training corpus — each document is a list of words
corpus = ["the cat sat on the mat", "the dog ran fast", "cats are great pets"]
tokenized = [doc.split() for doc in corpus]  # tokenize by splitting on spaces

# Build BM25 index from the tokenized corpus
bm25 = BM25Okapi(tokenized)

# Score all documents against a query — returns array of BM25 scores
scores = bm25.get_scores("cat mat".split())
# [0.93, 0.0, 0.44] — first doc scores highest (contains both "cat" and "mat")
```

Hybrid search combination:
```
final_score = alpha × dense_score + (1-alpha) × bm25_score  (alpha=0.5 is a good default)
```
Or use Reciprocal Rank Fusion (RRF): no need to tune alpha at all.

**WHY RRF is preferred over simple score fusion:** BM25 scores and dense similarity scores are on completely different scales (BM25 outputs can range from 0 to thousands; cosine similarity is always 0–1). RRF works purely on rank positions — "was this document ranked 1st, 5th, or 50th?" — so no calibration is needed.

---

## Metadata Filtering — Production RAG Must-Have

**What it is:** Storing structured attributes (date, author, document type, etc.) alongside your embeddings, and filtering on those attributes BEFORE or DURING vector search. This enforces hard constraints that semantic similarity cannot enforce.

**Analogy:** Semantic search is like searching by the *meaning* of a book. Metadata filtering is like adding library catalog constraints: "only books published after 2020, written in English, in the Science section." Both together give you the right books quickly.

Pure semantic search ignores document structure. Metadata filtering adds hard constraints.

Example: "What did our CEO say about revenue in Q4 2024?"
- **Without metadata filter:** retrieves docs from all years, all speakers, all quarters — semantic noise
- **With metadata filter:** filter where year=2024 AND quarter=Q4 AND speaker=CEO, THEN semantic search

**WHY:** Without metadata filters, your top-K results may be semantically similar to the query but from the wrong time period, wrong product, or wrong tenant. Semantic similarity cannot enforce "only show documents from 2024" — that is a hard constraint that metadata filtering provides.

Qdrant example:
```python
from qdrant_client.models import Filter, FieldCondition, MatchValue  # Qdrant filtering primitives

results = client.search(
    collection_name="documents",        # which vector collection to search
    query_vector=query_embedding,       # the embedded query to find similar vectors for
    query_filter=Filter(
        must=[                          # ALL conditions must be true (AND logic)
            FieldCondition(key="year",    match=MatchValue(value=2024)),    # only 2024 documents
            FieldCondition(key="quarter", match=MatchValue(value="Q4")),    # only Q4
        ]
    ),
    limit=5                             # return top 5 matches after filtering
)
```

Common metadata fields to store alongside every chunk:
- `document_id`, `source_url`, `date_created`, `author` — provenance
- `document_type` (contract, email, report) — content type filtering
- `department`, `product`, `region` — organizational filtering
- `chunk_index` (position of chunk in original document) — for ordering results

**Pre-filtering vs Post-filtering:**
- **Pre-filter:** Filter candidates FIRST using metadata, then vector search on filtered set (faster, always returns up to limit)
- **Post-filter:** Vector search first on full index, then filter results by metadata (more accurate but may return fewer than limit if many results are filtered out)

**Interview answer:** "How do you handle queries that need both semantic and exact matching?" → "Metadata filtering. Store structured attributes alongside embeddings. Use pre-filtering to narrow candidates before ANN search. Critical for multi-tenant systems where users must not see other tenants' data."

---

## Late Chunking — Jina AI 2024

**What it is:** A smarter way to create chunk embeddings that preserves the full document context. Standard chunking splits THEN embeds, so each chunk is embedded without context. Late chunking embeds THEN splits, using the full-document token representations.

**Problem with standard chunking:** When you split a document into chunks THEN embed each chunk independently, each chunk loses context about the surrounding document.

Example:
```
Document: "He was born in 1879. Einstein developed the theory of relativity."
Standard chunking:
  Chunk 1: "He was born in 1879."             ← who is "he"? The embedding model never sees Einstein's name
  Chunk 2: "Einstein developed..."            ← separate embedding, no connection to "He was born"
```

**WHY standard chunking fails here:** The embedding model processes each chunk in isolation. When it embeds "He was born in 1879," it does not know who "he" refers to because it never sees the second sentence. The resulting vector is ambiguous and retrieval is unreliable.

Late Chunking approach:
1. Embed the ENTIRE document first (get token-level embeddings for every token in the document)
2. THEN split into chunks at the designated boundaries
3. Pool (average) the token embeddings within each chunk boundary

**WHY late chunking works:** Transformer self-attention means every token's final embedding already reflects the full document — the token "He" in the first sentence has been cross-attended with "Einstein" in the second sentence. When you pool those post-attention token embeddings into a chunk vector, each chunk's vector naturally encodes the surrounding context.

Limitation: requires a model that supports long context (8K+ tokens). Cannot use for documents longer than the model's context window.

When to use: documents where pronoun resolution or entity references span chunk boundaries — legal documents, stories, research papers with forward references.

---

## 8. RAG vs Fine-Tuning

**What it is:** A decision framework for when to use RAG vs when to use fine-tuning. They solve different problems and are often used together.

| Dimension | RAG | Fine-Tuning |
|-----------|-----|------------|
| Knowledge update | Easy (update vector DB, no retraining) | Need to retrain (expensive) |
| Private data | Yes, naturally | Yes, but baked into weights |
| Cost | Low (just inference + retrieval) | Higher (training cost) |
| Hallucination | Reduced (grounded in retrieved context) | Can still hallucinate |
| Response style | Unchanged (base model behavior) | Can change behavior/tone |
| Latency | +retrieval overhead (~50–200ms) | Same as base model |
| When to use | Dynamic/external knowledge | Behavior/style change |

**Use both when possible** — fine-tune for style/reasoning/format, RAG for dynamic knowledge injection.

**WHY:** RAG cannot teach the model *how to reason* — it can only provide facts. Fine-tuning cannot provide up-to-date external facts — knowledge is baked in at training time. The combination gives you a model that reasons well AND has access to current information.

---

## 9. RAG Implementation Example (LangChain)

```python
from langchain_community.vectorstores import FAISS          # Facebook AI Similarity Search wrapper
from langchain_huggingface import HuggingFaceEmbeddings     # HuggingFace embedding model wrapper
from langchain.chains import RetrievalQA                    # pre-built RAG chain
from langchain_community.llms import Ollama                 # local LLM via Ollama

# Step 1: Set up the embedding model — same model used for both indexing and querying
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# Step 2: Load the pre-built vector store from disk (was created during indexing)
vectorstore = FAISS.load_local("faiss_index", embeddings)   # load index from disk
retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # wrap as retriever, returning top-5 chunks

# Step 3: Load a local LLM via Ollama (running llama3 locally — no API key needed)
llm = Ollama(model="llama3")

# Step 4: Combine retriever + LLM into a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,                        # the language model for generation
    chain_type="stuff",             # "stuff" = put all retrieved docs into one prompt (simplest approach)
    retriever=retriever,            # the retriever for finding relevant chunks
    return_source_documents=True    # return the source documents so you can show citations
)

# Run a query through the full RAG pipeline
result = qa_chain({"query": "What is the return policy?"})
print(result["result"])             # print the LLM's answer
# result["source_documents"] contains the retrieved chunks used for grounding
```

**WHY `chain_type="stuff"`:** "Stuff" simply concatenates all retrieved documents into the prompt. This is the simplest and most effective approach when the total number of tokens (query + retrieved docs) fits within the LLM's context window. For larger document sets, use "map_reduce" or "refine" chain types.

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
RAG Pipeline:     Ingest → Chunk → Embed → Index → Retrieve → Generate
Vector DB:        FAISS (local), Pinecone (cloud), ChromaDB (easy), Qdrant (fast+filter)
Embeddings:       BAAI/bge (open-source), text-embedding-ada-002 (OpenAI)
Retrieval:        Dense (semantic) + Sparse (BM25) + Hybrid + Reranking
Evaluation:       RAGAS: Faithfulness, Precision, Recall, Relevance
RAG vs FT:        RAG for dynamic knowledge, FT for behavior change
BM25:             Keyword search — use alongside dense for hybrid retrieval
Metadata filter:  Hard constraints on structured attributes — essential for multi-tenant
Late chunking:    Embed full doc first, split after — preserves cross-sentence context
```
