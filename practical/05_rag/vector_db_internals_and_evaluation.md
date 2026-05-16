# Vector DB Internals + RAG Evaluation Metrics

> How vector search actually works under the hood.
> Every evaluation metric you need to measure RAG quality.

---

## PART 1: VECTOR DATABASE INTERNALS

### Why You Can't Just Use Brute Force

You have 10 million document chunks, each a 1024-dim vector.
For each query, find the top-5 most similar chunks.

**Brute force:** compute cosine similarity with all 10M vectors.
10M × 1024 × multiply-add = 20 billion operations per query.
On a GPU: ~0.5 seconds per query. Unacceptable for production.

**Solution:** Approximate Nearest Neighbor (ANN) algorithms.
Trade a tiny amount of accuracy for massive speed gains.

---

## HNSW — Hierarchical Navigable Small World

**Used by:** Qdrant, Weaviate, FAISS, pgvector, Pinecone
**Speed:** O(log n) search
**Quality:** Best recall among ANN methods

### The Intuition: Skip Lists + Small World Graphs

Think of how a GPS navigates:
1. First, find the right country → continent-level map
2. Then, find the right city → country-level map
3. Then, find the right street → city-level map
4. Then, find the exact address → street-level map

HNSW does the same with vector space.

### The Structure

HNSW builds a multi-layer graph:

```
Layer 2 (sparse): only ~5% of vectors — long-range connections
  [A]-------[E]-------[I]

Layer 1 (medium): ~20% of vectors — medium-range connections
  [A]--[B]--[C]--[D]--[E]--[F]--[G]--[H]--[I]

Layer 0 (dense): ALL vectors — short-range connections
  [A][B][C][D][E][F][G][H][I][J][K][L]...
```

### Search Algorithm

```
Query vector q:

Step 1: Enter at top layer (Layer 2)
        Start at a random entry point
        Greedily move to the neighbor closest to q
        Until local minimum (no neighbor is closer than current)

Step 2: Drop to Layer 1
        Continue greedy search from where you stopped

Step 3: Drop to Layer 0
        Full search in local neighborhood
        Return top-K results
```

### Key Parameters

| Parameter | What it controls | Default | Trade-off |
|-----------|-----------------|---------|-----------|
| `M` | Number of connections per node | 16 | Higher M → better recall, more memory |
| `ef_construction` | Beam width during index build | 200 | Higher → better quality index, slower build |
| `ef_search` | Beam width during search | 50 | Higher → better recall, slower search |

```python
# Creating HNSW index in FAISS
import faiss

d = 1024         # embedding dimension
M = 16           # connections per node
ef_construction = 200

index = faiss.IndexHNSWFlat(d, M)
index.hnsw.efConstruction = ef_construction
index.hnsw.efSearch = 50

index.add(embeddings)     # add all vectors
D, I = index.search(query, k=5)   # search top-5
```

### Why HNSW is the Default Choice
- Best recall-speed trade-off
- No training required (unlike IVF)
- Dynamic: add/delete vectors without rebuilding
- Memory: ~1.5× the raw vector size

---

## IVF — Inverted File Index

**Used by:** FAISS (IVFFlat, IVFPQ), Milvus
**Speed:** O(nprobe × cluster_size) — configurable
**Quality:** Tunable recall vs speed

### The Intuition: Clustering First

Instead of searching all vectors, first cluster them.
At query time, only search the nearest clusters.

```
Build phase:
  Run K-Means on all vectors → nlist clusters (centroids)
  
  Cluster 1 (centroid: [0.2, 0.8, ...]): docs 45, 102, 891, ...
  Cluster 2 (centroid: [0.9, 0.1, ...]): docs 12, 677, 203, ...
  ...
  Cluster nlist: docs 3, 99, 1041, ...

Search phase:
  1. Find the nprobe closest cluster centroids to query
  2. Search only vectors in those nprobe clusters
  3. Return top-K from that subset
```

### Key Parameters

| Parameter | What it controls | Default |
|-----------|-----------------|---------|
| `nlist` | Number of clusters | sqrt(N) is a good rule |
| `nprobe` | Clusters to search | 1-10% of nlist |

```python
import faiss

d = 1024
nlist = 1000       # number of clusters

quantizer = faiss.IndexFlatL2(d)    # coarse quantizer
index = faiss.IndexIVFFlat(quantizer, d, nlist)

index.train(embeddings)             # MUST train IVF (learns centroids)
index.add(embeddings)

index.nprobe = 10                   # search 10 nearest clusters
D, I = index.search(query, k=5)
```

### IVF vs HNSW Trade-offs

| Dimension | HNSW | IVF |
|-----------|------|-----|
| Requires training | No | Yes (K-Means) |
| Memory | Higher | Lower |
| Dynamic updates | Easy | Rebuild needed |
| Recall at same speed | Better | Slightly lower |
| Production default | Yes | When memory is limited |

---

## PRODUCT QUANTIZATION (PQ) — Vector Compression

**Used in:** FAISS (IVFPQ), Qdrant, Vespa
**Purpose:** Compress vectors to save memory (10-100× compression)

### The Intuition

A 1024-dim float32 vector = 4096 bytes (4KB).
1 million vectors = 4 GB.
100 million vectors = 400 GB. Doesn't fit in RAM.

PQ compresses each vector from thousands of bytes to tens of bytes.

### How It Works

**Step 1: Split the vector into M subspaces**
```
Original: [d1, d2, d3, d4, ..., d1024]
Split into M=8 subspaces of 128 dims each:
  Sub1: [d1,   ..., d128]
  Sub2: [d129, ..., d256]
  ...
  Sub8: [d897, ..., d1024]
```

**Step 2: Train a codebook for each subspace**
- Run K-Means with k=256 on each subspace independently
- Gives 256 "codewords" per subspace

**Step 3: Encode each vector**
- For each subspace, find the nearest codeword (1 of 256)
- Store only the codeword index (1 byte = 8 bits for 256 options)
- Result: M bytes per vector instead of d × 4 bytes

```
Original: 1024 × 4 bytes = 4096 bytes per vector
PQ M=8:   8 × 1 byte    = 8 bytes per vector
Compression: 512×
```

**Step 4: Approximate distance computation**
```
For query q:
  Pre-compute distances from q to all 256 codewords in each subspace
  For each database vector (now encoded as M indices):
    distance ≈ sum of pre-computed distances for each subspace
```

### IVFPQ — The Production Combination

Combine IVF coarse quantization + PQ compression:

```python
import faiss

d = 1024
nlist = 1000   # IVF clusters
M = 8          # PQ subspaces (8 bytes per vector)
nbits = 8      # bits per subspace (256 codewords)

quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)

index.train(embeddings)
index.add(embeddings)

index.nprobe = 10
D, I = index.search(query, k=5)
```

### Memory Comparison (1 million vectors, d=1024)

| Index Type | Memory | Recall@10 |
|-----------|--------|-----------|
| Flat (exact) | 4 GB | 100% |
| HNSW | 6 GB | 98% |
| IVF (nlist=1000) | 4.1 GB | 95% |
| IVFPQ (M=8) | 8 MB | 85% |
| IVFPQ (M=32) | 32 MB | 92% |

---

## PART 2: RAG EVALUATION METRICS

### Why RAGAS Alone Isn't Enough

RAGAS measures model-level quality (faithfulness, relevance).
In production you also need **retrieval-level** metrics from Information Retrieval (IR).

---

## METRIC 1: Hit Rate (Recall@K)

**The simplest metric. Measures: "Did we retrieve the right document?"**

```
For each query:
  Did the correct document appear in the top-K retrieved results?
  Yes → 1    No → 0

Hit Rate@K = (number of queries with correct doc in top-K) / total queries
```

**Example:**
```
Query: "What is the refund policy?"
Correct doc: "returns_policy.pdf, page 3"
Top-5 retrieved: [returns.pdf p3, shipping.pdf p1, faq.pdf p5, ...]

Correct doc IS in top-5 → Hit = 1
```

**Typical targets:** Hit Rate@5 > 0.85 for production RAG

---

## METRIC 2: MRR — Mean Reciprocal Rank

**Measures: "How HIGH in the ranking is the correct document?"**

```
For each query:
  Find the rank of the first correct document (1, 2, 3, ...)
  Reciprocal Rank = 1 / rank

MRR = mean of Reciprocal Ranks across all queries
```

**Example:**
```
Query 1: correct doc at rank 1 → RR = 1/1 = 1.0
Query 2: correct doc at rank 3 → RR = 1/3 = 0.33
Query 3: correct doc at rank 2 → RR = 1/2 = 0.5

MRR = (1.0 + 0.33 + 0.5) / 3 = 0.61
```

**Why MRR matters:**
The correct document at rank 1 vs rank 5 makes a huge difference
when you only pass top-3 docs to the LLM. MRR captures this.

---

## METRIC 3: NDCG — Normalized Discounted Cumulative Gain

**Measures: "Are the MOST relevant docs ranked HIGHEST?"**

More sophisticated than MRR — handles **graded relevance** (not just correct/wrong).

**Relevance grades:**
```
3 = Perfectly relevant (direct answer to the question)
2 = Highly relevant (closely related)
1 = Somewhat relevant (loosely related)
0 = Not relevant
```

**DCG (Discounted Cumulative Gain):**
```
DCG@K = Σ (rel_i / log₂(i+1)) for i=1 to K
```
Documents ranked higher contribute more (log discounting).

**NDCG = DCG / IDCG**
Where IDCG = DCG of perfect ranking (most relevant first).
Normalizes to [0, 1]. NDCG=1 = perfect ranking.

**Example:**
```
Query: "What is the capital of France?"

Retrieval ranking:
  Rank 1: "Paris is the capital of France" → relevance = 3
  Rank 2: "France is a country in Europe"  → relevance = 1
  Rank 3: "French cuisine includes..."      → relevance = 0

DCG@3 = 3/log₂(2) + 1/log₂(3) + 0/log₂(4)
      = 3/1 + 1/1.585 + 0
      = 3 + 0.631
      = 3.631

Ideal ranking: [3, 1, 0] → IDCG = 3.631 (same here, already ideal)
NDCG@3 = 3.631 / 3.631 = 1.0 (perfect)
```

**When to use NDCG:** When you have human-labeled relevance grades. Best for
rigorous academic-style RAG evaluation.

---

## METRIC 4: MAP — Mean Average Precision

**Measures: "Are ALL relevant documents retrieved, and ranked high?"**

```
For each query with multiple relevant documents:
  Average Precision (AP) = average of Precision@k at each rank
                           where a relevant document is retrieved

MAP = mean of AP across all queries
```

**Example:**
```
Query has 3 relevant docs. Retrieval of top-5:
  Rank 1: relevant    → P@1 = 1/1 = 1.0
  Rank 2: not relevant
  Rank 3: relevant    → P@3 = 2/3 = 0.67
  Rank 4: not relevant
  Rank 5: relevant    → P@5 = 3/5 = 0.6

AP = (1.0 + 0.67 + 0.6) / 3 = 0.76
```

---

## METRIC 5: RAGAS Metrics (Model-Level)

These measure LLM output quality, not just retrieval:

| Metric | Formula (simplified) | What it catches |
|--------|---------------------|-----------------|
| **Context Precision** | Relevant retrieved chunks / Total retrieved | Noisy retrieval |
| **Context Recall** | Relevant chunks retrieved / All relevant chunks | Missing information |
| **Faithfulness** | Claims supported by context / Total claims | Hallucination |
| **Answer Relevance** | Does answer address the question? | Off-topic answers |

```python
from ragas import evaluate
from ragas.metrics import (
    context_precision, context_recall,
    faithfulness, answer_relevancy
)
from datasets import Dataset

data = {
    "question":  ["What is RAG?"],
    "answer":    ["RAG combines retrieval with generation..."],
    "contexts":  [["RAG (Retrieval Augmented Generation) is..."]],
    "ground_truth": ["RAG is a technique that..."]
}

dataset = Dataset.from_dict(data)
result = evaluate(dataset, metrics=[
    context_precision, context_recall,
    faithfulness, answer_relevancy
])
print(result)
```

---

## FULL EVALUATION PIPELINE FOR PRODUCTION RAG

```
STEP 1: Create evaluation dataset
  - 100-200 representative queries
  - Ground truth answers
  - Labeled relevant documents (for retrieval metrics)

STEP 2: Measure retrieval quality
  - Hit Rate@5: did we get the right chunk?
  - MRR: is the right chunk ranked high?
  - NDCG@5: is ranking quality good?

STEP 3: Measure generation quality
  - RAGAS Faithfulness: is the LLM hallucinating?
  - RAGAS Answer Relevance: is it answering the question?
  - RAGAS Context Precision/Recall: retrieval quality

STEP 4: End-to-end metrics
  - Exact Match: is the answer exactly right?
  - F1 Score: partial credit for overlapping words
  - Human evaluation: sample 50 answers, rate 1-5

STEP 5: Monitor in production
  - Log every query + retrieved chunks + answer
  - Weekly RAGAS evaluation on production traffic sample
  - Alert if faithfulness drops below threshold
```

---

## INTERVIEW BLAST — Vector DB & Evaluation

**"Explain how HNSW works."**
> "HNSW builds a multi-layer graph where the top layer is sparse with long-range
> connections and the bottom layer is dense. Search starts at the top layer,
> greedily moves toward the query, drops to the next layer, and repeats until
> reaching the dense bottom layer. This achieves O(log n) search time.
> Key parameters: M controls connections per node (more = better recall, more memory),
> ef_construction controls build quality, ef_search controls query recall vs speed."

**"What is Product Quantization and why use it?"**
> "PQ splits each embedding vector into M subspaces, trains a codebook of 256
> codewords per subspace, then encodes each vector as M byte-sized indices.
> A 1024-dim float32 vector (4KB) becomes 8 bytes with M=8 — 512× compression.
> Distance computation uses pre-computed lookup tables: O(M) per vector instead
> of O(d). Used in IVFPQ when you have 100M+ vectors and can't fit them in RAM."

**"How do you evaluate a RAG system in production?"**
> "Two levels: retrieval and generation. For retrieval: Hit Rate@K (did we get
> the right chunk?), MRR (how high is it ranked?), NDCG (is graded relevance
> in the right order?). For generation: RAGAS Faithfulness (no hallucinations),
> Context Precision (no noisy retrieval), Answer Relevance (actually answering
> the question). In production I log all queries and run weekly automated evaluation
> with alerts when faithfulness drops below 0.8."
