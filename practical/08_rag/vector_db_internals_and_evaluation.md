# Vector DB Internals + RAG Evaluation Metrics

> How vector search actually works under the hood.
> Every evaluation metric you need to measure RAG quality.

---

## PART 1: VECTOR DATABASE INTERNALS

### Why You Can't Just Use Brute Force

**What it is:** Brute force search means comparing your query vector against every single vector in the database, one by one. This is mathematically simple but computationally too slow for production systems with millions of documents.

You have 10 million document chunks, each a 1,024-dimensional vector.
For each user query, you need to find the top-5 most similar chunks.

**Brute force calculation:**
10 million vectors × 1,024 dimensions × 2 operations (multiply + add) = 20 billion operations per query.
On a GPU: ~0.5 seconds per query. Completely unacceptable for a production API.

**Solution:** Approximate Nearest Neighbor (ANN) algorithms.
Trade a tiny amount of recall accuracy for massive (100–1,000×) speed gains.

**Analogy:** Finding your friend in a city. Brute force = knock on every door in the city one by one. ANN = use GPS to narrow to the right neighborhood, then look at nearby streets. You might miss if your friend is visiting someone far away, but you find them 99% of the time in seconds instead of days.

---

## HNSW — Hierarchical Navigable Small World

**What it is:** HNSW is the most widely used ANN algorithm. It builds a multi-layer graph where the top layer has sparse, long-range connections for fast navigation, and the bottom layer has dense, short-range connections for precise local search.

**Used by:** Qdrant, Weaviate, FAISS, pgvector, Pinecone
**Speed:** O(log n) — logarithmic search time (10M vectors takes about as long as 1K vectors)
**Quality:** Best recall among ANN methods (typically 97–99%)

### The Intuition: Skip Lists + Small World Graphs

**Analogy:** Think of how a GPS navigates from London to a specific house in Paris:
1. First, plan the route at the highway level — Paris is in France, use the Channel Tunnel (long-range, coarse)
2. Then, navigate to the right district of Paris (medium-range)
3. Then, navigate to the right street (short-range)
4. Then, find the exact house number (precise, local)

HNSW does the exact same thing with vector space — navigate from coarse to fine.

### The Structure

HNSW builds a multi-layer graph where each upper layer is a progressively sparser subset of the lower layers:

```
Layer 2 (very sparse — only ~5% of all vectors — for long-range navigation):
  [A]-------[E]-------[I]     (only 3 nodes, but they are far apart — fast coarse search)

Layer 1 (medium density — ~20% of vectors):
  [A]--[B]--[C]--[D]--[E]--[F]--[G]--[H]--[I]   (more nodes, medium-range connections)

Layer 0 (dense — ALL vectors — for precise local search):
  [A][B][C][D][E][F][G][H][I][J][K][L][M]...    (every vector is here with local connections)
```

**WHY this hierarchy:** You want to avoid checking every vector (too slow) but also avoid missing the closest vector (too imprecise). The hierarchy gives you the best of both: fast coarse navigation at the top to get into the right neighborhood, then precise local search at the bottom to find the exact nearest neighbor.

### Search Algorithm

```
Query vector q:

Step 1: Enter at the TOP layer (Layer 2 — sparsest)
        Start at a random entry point (or a fixed global entry point)
        Greedily move to whichever neighbor is closest to q
        Stop when no neighbor is closer than the current node (local minimum)

Step 2: Drop down to Layer 1
        Continue greedy search from the node you stopped at
        Again move greedily until local minimum

Step 3: Drop down to Layer 0 (densest layer — all vectors)
        Full local search in the dense neighborhood
        Collect top-K results from this neighborhood
        Return top-K as the approximate nearest neighbors
```

### Key Parameters

| Parameter | What it controls | Default | Trade-off |
|-----------|-----------------|---------|-----------|
| `M` | Number of bidirectional connections per node | 16 | Higher M → better recall at query time, but more memory for the graph |
| `ef_construction` | Beam width during index building | 200 | Higher → better quality index (more accurate graph), but slower to build |
| `ef_search` | Beam width during search | 50 | Higher → better recall at query time, but slower search |

```python
import faiss  # Facebook AI Similarity Search — most widely used ANN library

d = 1024          # embedding dimension (must match your embedding model's output size)
M = 16            # connections per node — 16 is the standard default
ef_construction = 200  # build quality — 200 is a good default

# Create HNSW index with flat (exact) distance computation at Layer 0
index = faiss.IndexHNSWFlat(d, M)

# Set construction and search parameters
index.hnsw.efConstruction = ef_construction   # higher = better index quality, slower build
index.hnsw.efSearch = 50                       # higher = better recall at query, slower search

index.add(embeddings)                           # add all vectors to build the index
D, I = index.search(query, k=5)                # search: D=distances, I=indices of top-5 nearest
```

**WHY HNSW is the default choice:**
- Best recall-speed trade-off among ANN algorithms
- No training required (unlike IVF which needs K-Means pre-clustering)
- Dynamic: add/delete vectors without rebuilding the entire index
- Memory overhead: ~1.5× the raw vector size (manageable)

---

## IVF — Inverted File Index

**What it is:** IVF first clusters all vectors using K-Means, then at query time only searches the nearest clusters instead of all vectors. This is faster than HNSW when memory is constrained, but requires a training step.

**Used by:** FAISS (IVFFlat, IVFPQ), Milvus
**Speed:** O(nprobe × cluster_size) — configurable speed/recall trade-off
**Quality:** Tunable — more clusters searched = better recall, slower query

### The Intuition: Clustering First

**Analogy:** A library with 1 million books, organized into 1,000 subject areas. Instead of scanning all 1 million books, you first identify the 10 most relevant subject areas (nprobe=10), then search only those 10 areas (~10,000 books). 100× faster, and you rarely miss the best match.

Instead of searching all vectors, first cluster them.
At query time, only search the nearest clusters.

```
Build phase (done once before serving):
  Run K-Means on all vectors → nlist clusters (centroids stored)
  
  Cluster 1 (centroid near [0.2, 0.8, ...]): contains docs 45, 102, 891, ... (all similar vectors)
  Cluster 2 (centroid near [0.9, 0.1, ...]): contains docs 12, 677, 203, ...
  ...
  Cluster nlist: contains docs 3, 99, 1041, ...

Search phase (for each query):
  1. Find the nprobe cluster centroids that are closest to the query vector
  2. Search only the vectors in those nprobe clusters (ignoring all others)
  3. Return top-K from that subset
```

### Key Parameters

| Parameter | What it controls | Default |
|-----------|-----------------|---------|
| `nlist` | Number of clusters | sqrt(N) is a good rule of thumb |
| `nprobe` | Clusters to search at query time | 1–10% of nlist is typical |

```python
import faiss  # FAISS library

d = 1024             # embedding dimension
nlist = 1000         # number of clusters (sqrt of dataset size is a good heuristic)

quantizer = faiss.IndexFlatL2(d)              # the "coarse" quantizer — measures distance to centroids
index = faiss.IndexIVFFlat(quantizer, d, nlist) # create IVF index using L2 distance

index.train(embeddings)   # CRITICAL: IVF must be trained first to compute K-Means cluster centroids
                          # HNSW does not need this — IVF does
index.add(embeddings)     # add all vectors to the index after training

index.nprobe = 10         # at query time, search only the 10 nearest clusters
D, I = index.search(query, k=5)  # find top-5 nearest neighbors
```

**WHY `index.train()` is required:** IVF needs to know where to put the cluster centroids before it can assign vectors to clusters. This K-Means training step looks at all your vectors and finds the best nlist cluster centers. Without training, the index doesn't know how to organize the vectors.

### IVF vs HNSW Trade-offs

| Dimension | HNSW | IVF |
|-----------|------|-----|
| Requires training | No — just add vectors | Yes — K-Means first |
| Memory usage | Higher (~1.5× vectors) | Lower (~1.0× vectors) |
| Dynamic updates (add/remove) | Easy — just add | Harder — may need rebuild |
| Recall at same query speed | Better | Slightly lower |
| Best for | Default production choice | When memory is very tight |

---

## PRODUCT QUANTIZATION (PQ) — Vector Compression

**What it is:** PQ compresses each high-dimensional vector into a tiny byte-level code by splitting the vector into subspaces and replacing each subspace with a codebook index. Enables 100–500× memory reduction at the cost of some recall accuracy.

**Used in:** FAISS (IVFPQ), Qdrant, Vespa

**Why memory compression is critical:**
A 1,024-dim float32 vector = 4,096 bytes = 4 KB per vector.
1 million vectors = 4 GB — fits in RAM.
100 million vectors = 400 GB — does NOT fit in RAM. You need compression.

**Analogy:** Instead of storing every pixel of 100 million photos, you store a color palette (the codebook) and for each photo, just note which palette colors are used (the codes). You can reconstruct an approximate version of each photo from the codes + palette. PQ does this with vector "pixels."

### How It Works

**Step 1: Split the vector into M subspaces**
```
Original 1024-dim vector: [d1, d2, d3, d4, ..., d1024]

Split into M=8 equal subspaces of 128 dims each:
  Subspace 1: [d1,   ..., d128]   ← first 128 dimensions
  Subspace 2: [d129, ..., d256]   ← next 128 dimensions
  ...
  Subspace 8: [d897, ..., d1024]  ← last 128 dimensions
```

**Step 2: Train a codebook for each subspace**
- Run K-Means with k=256 on each subspace independently (using all document vectors)
- Gives 256 "codewords" (cluster centroids) per subspace
- Each codeword is a 128-dim vector representing a common pattern in that subspace

**Step 3: Encode each vector**
- For each subspace, find the nearest codeword (1 of 256 options)
- Store only the codeword index (an integer 0–255 = 1 byte)
- Result: M bytes per vector instead of d × 4 bytes

```
Original: 1024 × 4 bytes = 4,096 bytes per vector
PQ M=8:   8 × 1 byte     = 8 bytes per vector
Compression ratio: 4096 / 8 = 512× compression!
```

**Step 4: Approximate distance computation at query time**
```
For query vector q:
  Step A: Pre-compute the distance from q to all 256 codewords in each subspace
           (256 × 8 = 2,048 distance computations — done ONCE per query, not per document)
  
  Step B: For each database vector (stored as M=8 byte-sized indices):
           Look up the pre-computed distance for each subspace index
           Sum the 8 looked-up distances → approximate total distance to q
           
This is the ADC (Asymmetric Distance Computation) trick:
  Cost per vector = M lookups (M=8) instead of d multiplications (d=1024)
  128× faster distance computation per vector!
```

### IVFPQ — The Production Combination

**What it is:** Combine IVF (coarse search over clusters) with PQ (compressed vector storage) for the most memory-efficient ANN setup. Standard for 100M+ vector databases.

Combine IVF coarse quantization + PQ compression for the best memory-speed trade-off:

```python
import faiss  # FAISS library

d = 1024       # original embedding dimension
nlist = 1000   # IVF: number of clusters to organize vectors into
M = 8          # PQ: number of subspaces — 8 bytes per vector
nbits = 8      # PQ: bits per subspace index — 8 bits = 256 codewords per subspace

quantizer = faiss.IndexFlatL2(d)                     # coarse quantizer for IVF cluster centroid search
index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)  # combined IVF + PQ index

index.train(embeddings)   # train both the IVF cluster centroids AND the PQ codebooks
index.add(embeddings)     # add all vectors (now stored compressed as M-byte codes)

index.nprobe = 10         # search 10 nearest clusters at query time
D, I = index.search(query, k=5)  # find approximate top-5 nearest neighbors
```

### Memory Comparison (1 million vectors, d=1024)

| Index Type | Memory | Recall@10 | When to Use |
|-----------|--------|-----------|-------------|
| Flat (exact brute force) | 4 GB | 100% | Dev/testing only |
| HNSW (M=16) | 6 GB | 98% | Production default |
| IVF (nlist=1000) | 4.1 GB | 95% | When memory is tight |
| IVFPQ (M=8) | 8 MB | 85% | 100M+ vectors, RAM-constrained |
| IVFPQ (M=32) | 32 MB | 92% | Large scale, better quality |

**WHY IVFPQ's 85% recall is acceptable:** Missing 15% of results means occasionally retrieving 4 relevant documents instead of 5. In practice, the top 4 retrieved documents are usually sufficient for good generation quality. The 512× memory savings (4 GB → 8 MB) is worth the 15% recall drop for massive-scale deployments.

---

## PART 2: RAG EVALUATION METRICS

### Why RAGAS Alone Isn't Enough

**What it is:** RAGAS measures how good the LLM's answer is (faithfulness, relevance). But before the LLM generates an answer, the retrieval must work correctly. You need separate metrics for the retrieval step.

**Analogy:** Evaluating a restaurant is not just about the final dish. You also evaluate whether the chef got the right ingredients from the market (retrieval). If the chef asked for salmon and got tuna, even a masterful chef cannot make the right dish (generation fails because retrieval failed).

RAGAS measures model-level quality (faithfulness, relevance).
In production you also need **retrieval-level** metrics from Information Retrieval (IR).

---

## METRIC 1: Hit Rate (Recall@K)

**What it is:** The simplest and most important retrieval metric. For each test question, check: did the correct document appear in the top-K retrieved results? Hit Rate@K is the percentage of questions where this is true.

**The simplest metric. Measures: "Did we retrieve the right document?"**

```
For each query in your test set:
  Did the correct document appear in the top-K retrieved results?
  Yes → score = 1    No → score = 0

Hit Rate@K = (number of queries with correct doc in top-K) / total queries
```

**Example:**
```
Query: "What is the refund policy?"
Correct document: "returns_policy.pdf, page 3"
Top-5 retrieved documents: [returns_policy.pdf p3, shipping.pdf p1, faq.pdf p5, ...]

Correct doc IS in the top-5 → Hit = 1 (success)
```

**Typical production targets:** Hit Rate@5 > 0.85 means 85%+ of user questions find the right document in the top 5 results.

**WHY Hit Rate@K is the primary metric:** If the correct document is not in the top-K, the LLM never sees it and cannot answer correctly — this is a hard failure. No amount of good generation can compensate for retrieval misses. Fix retrieval first.

---

## METRIC 2: MRR — Mean Reciprocal Rank

**What it is:** MRR measures not just whether the correct document was retrieved, but HOW HIGH in the ranking it appeared. A correct document at rank 1 (very useful) is much better than one at rank 5 (often missed).

**Measures: "How HIGH in the ranking is the correct document?"**

```
For each query:
  Find the rank of the first correct document (rank 1 = top result, rank 5 = fifth result)
  Reciprocal Rank = 1 / rank   (rank 1 → 1.0, rank 2 → 0.5, rank 5 → 0.2)

MRR = average of Reciprocal Ranks across all queries
```

**Example:**
```
Query 1: correct doc at rank 1 → RR = 1/1 = 1.00  (perfect — top result)
Query 2: correct doc at rank 3 → RR = 1/3 = 0.33  (found but not at top)
Query 3: correct doc at rank 2 → RR = 1/2 = 0.50  (second result — good)

MRR = (1.00 + 0.33 + 0.50) / 3 = 0.61
```

**WHY MRR matters:**
If you only pass the top-3 documents to the LLM (to save tokens), a correct document at rank 1 vs rank 5 makes an enormous difference. MRR captures this distinction that Hit Rate@K misses — you want not just retrieval but high-ranked retrieval.

---

## METRIC 3: NDCG — Normalized Discounted Cumulative Gain

**What it is:** NDCG is the most sophisticated retrieval metric. Instead of binary (correct/wrong), it uses graded relevance scores (0–3). Instead of just checking if a document is present, it measures whether the MOST relevant documents appear FIRST.

**Measures: "Are the MOST relevant docs ranked HIGHEST?"**

More sophisticated than MRR — handles **graded relevance** (not just correct/wrong).

**Relevance grades:**
```
3 = Perfectly relevant — directly answers the question
2 = Highly relevant — closely related, provides useful context
1 = Somewhat relevant — loosely related, marginally useful
0 = Not relevant — irrelevant to the query
```

**DCG (Discounted Cumulative Gain):**
```
DCG@K = Σ (rel_i / log₂(i+1)) for i=1 to K
```
Documents ranked higher contribute more (the log₂(i+1) denominator discounts lower-ranked items).
A relevance=3 doc at rank 1 contributes 3/log₂(2) = 3.0.
A relevance=3 doc at rank 4 contributes 3/log₂(5) = 1.29. (much less — it's ranked lower)

**NDCG = DCG / IDCG**
Where IDCG = DCG of the *ideal* ranking (most relevant documents first).
NDCG is normalized to [0, 1]. NDCG=1 means perfect ranking. NDCG=0.5 means roughly half as good as ideal.

**Example:**
```
Query: "What is the capital of France?"

Your retrieval ranking:
  Rank 1: "Paris is the capital and largest city of France" → relevance = 3 (perfect)
  Rank 2: "France is a country in Western Europe"          → relevance = 1 (somewhat relevant)
  Rank 3: "French cuisine includes croissants and baguettes" → relevance = 0 (not relevant)

DCG@3 = 3/log₂(2) + 1/log₂(3) + 0/log₂(4)
      = 3/1 + 1/1.585 + 0
      = 3.0 + 0.631 + 0
      = 3.631

Ideal ranking: [3, 1, 0] → IDCG@3 = 3.631 (this IS the ideal ranking already)
NDCG@3 = 3.631 / 3.631 = 1.0 (perfect score — most relevant doc is at rank 1)
```

**When to use NDCG:** When you have human-labeled relevance grades (multi-level, not just binary). Best for rigorous academic-style RAG evaluation or when you care about ranking quality, not just presence.

---

## METRIC 4: MAP — Mean Average Precision

**What it is:** MAP measures whether ALL relevant documents are retrieved (not just one) and whether they are ranked high. It is the standard metric when a query has multiple correct answers.

**Measures: "Are ALL relevant documents retrieved, and ranked high?"**

```
For each query with multiple relevant documents:
  Average Precision (AP) = average of Precision@k values at each rank where a relevant doc appears

MAP = mean of AP values across all queries
```

**Example:**
```
Query has 3 relevant documents. Retrieval returns top-5:
  Rank 1: RELEVANT → Precision@1 = 1/1 = 1.0   (1 relevant in top 1)
  Rank 2: not relevant
  Rank 3: RELEVANT → Precision@3 = 2/3 = 0.67  (2 relevant in top 3)
  Rank 4: not relevant
  Rank 5: RELEVANT → Precision@5 = 3/5 = 0.60  (3 relevant in top 5)

AP = (1.0 + 0.67 + 0.60) / 3 = 0.76   (average over the 3 relevant ranks)
```

**WHY MAP is used:** When answering questions requires combining information from multiple documents (multi-hop questions, comparative questions), you need ALL relevant documents, not just one. MAP rewards systems that retrieve all relevant documents AND rank them high.

---

## METRIC 5: RAGAS Metrics (Model-Level)

**What it is:** RAGAS metrics evaluate the quality of the entire RAG pipeline output — not just retrieval, but whether the LLM's answer is grounded in the context and actually answers the question.

These measure LLM output quality, not just retrieval:

| Metric | Formula (simplified) | What it catches |
|--------|---------------------|-----------------|
| **Context Precision** | Relevant retrieved chunks / Total retrieved chunks | Noisy retrieval — you retrieved irrelevant documents |
| **Context Recall** | Relevant chunks retrieved / All relevant chunks that exist | Missing information — you failed to retrieve key documents |
| **Faithfulness** | Claims in answer supported by context / Total claims | Hallucination — LLM made up facts not in the retrieved docs |
| **Answer Relevance** | Does the answer actually address the question asked? | Off-topic answers — answer is grounded but doesn't help |

```python
from ragas import evaluate                                          # main RAGAS evaluation function
from ragas.metrics import (
    context_precision, context_recall,                            # retrieval quality metrics
    faithfulness, answer_relevancy                                # generation quality metrics
)
from datasets import Dataset  # HuggingFace datasets format

# Build the evaluation dataset — requires 4 fields
data = {
    "question":     ["What is RAG?"],                            # the user's question
    "answer":       ["RAG combines retrieval with generation..."], # the system's actual answer
    "contexts":     [["RAG (Retrieval Augmented Generation) is..."]],  # the retrieved chunks (list of lists)
    "ground_truth": ["RAG is a technique that..."]               # the correct reference answer
}

dataset = Dataset.from_dict(data)      # convert to HuggingFace Dataset format

# Run evaluation — RAGAS uses an LLM (default GPT-4) as the judge
result = evaluate(dataset, metrics=[
    context_precision,   # are the retrieved chunks relevant?
    context_recall,      # did we retrieve all the relevant chunks?
    faithfulness,        # does the answer stay grounded in the context?
    answer_relevancy     # does the answer actually answer the question?
])
print(result)  # returns dict like: {'context_precision': 0.87, 'faithfulness': 0.91, ...}
```

**WHY RAGAS uses an LLM as judge:** These metrics require understanding natural language to judge quality. "Is this answer grounded in this context?" is a reading comprehension task. RAGAS uses GPT-4 (or another powerful LLM) to make these judgments, rather than writing complex rule-based logic that would inevitably miss edge cases.

---

## FULL EVALUATION PIPELINE FOR PRODUCTION RAG

**What it is:** A systematic process for measuring, monitoring, and improving RAG quality in production. Not just a one-time evaluation, but ongoing monitoring.

```
STEP 1: Create evaluation dataset (do this BEFORE building the system)
  - 100–200 representative queries from your target user base
  - Ground truth answers for each query
  - Labeled relevant documents for each query (for retrieval metrics)

STEP 2: Measure RETRIEVAL quality
  - Hit Rate@5: did the correct chunk appear in the top 5? (most important)
  - MRR: is the correct chunk ranked near the top? (rank quality)
  - NDCG@5: is graded relevance ordering correct? (for multi-level relevance)

STEP 3: Measure GENERATION quality
  - RAGAS Faithfulness: is the LLM hallucinating, or staying grounded in context?
  - RAGAS Answer Relevance: is the LLM actually answering the question asked?
  - RAGAS Context Precision/Recall: comprehensive retrieval quality view

STEP 4: End-to-end metrics
  - Exact Match: is the answer exactly right? (for short factual answers)
  - F1 Score: partial credit — does the answer contain the correct words?
  - Human evaluation: sample 50 answers, rate each 1–5 for quality

STEP 5: Monitor in production (ongoing)
  - Log every query + retrieved chunks + final answer to a database
  - Run weekly automated RAGAS evaluation on a random sample of production traffic
  - Set alerts: if faithfulness drops below 0.8, investigate immediately
  - Track MRR over time — if it degrades, your vector DB may need refreshing
```

**WHY the evaluation dataset should be built BEFORE the system:** You need an objective test set that was not influenced by your implementation choices. If you build the system first, you might unconsciously make choices that favor your existing test queries. Treat evaluation like a held-out test set in traditional ML.

---

## INTERVIEW BLAST — Vector DB & Evaluation

**"Explain how HNSW works."**
> "HNSW builds a multi-layer graph where the top layer is sparse with long-range
> connections and the bottom layer is dense with local connections. Search starts
> at the top layer, greedily moves toward the query vector, drops to the next layer,
> and repeats until reaching the dense bottom layer. This achieves O(log n) search.
> Key parameters: M controls connections per node (more = better recall, more memory),
> ef_construction controls build quality, ef_search controls query recall vs speed."

**"What is Product Quantization and why use it?"**
> "PQ splits each embedding vector into M subspaces, trains a codebook of 256
> codewords per subspace, then encodes each vector as M byte-sized indices.
> A 1024-dim float32 vector (4KB) becomes 8 bytes with M=8 — 512× compression.
> Distance computation uses pre-computed lookup tables: O(M) per vector instead
> of O(d). Used in IVFPQ when you have 100M+ vectors and cannot fit them in RAM."

**"How do you evaluate a RAG system in production?"**
> "Two levels: retrieval and generation. For retrieval: Hit Rate@K (did we get
> the right chunk?), MRR (how high is it ranked?), NDCG (is graded relevance
> in the right order?). For generation: RAGAS Faithfulness (no hallucinations),
> Context Precision (no noisy retrieval), Answer Relevance (actually answering
> the question). In production I log all queries and run weekly automated evaluation
> with alerts when faithfulness drops below 0.8."
