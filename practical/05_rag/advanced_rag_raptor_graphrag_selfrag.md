# Advanced RAG — RAPTOR, GraphRAG, Self-RAG, CRAG, Agentic RAG

> Basic RAG gets you in the door. This file gets you the job.
> Every advanced RAG technique asked at 2-4 year GenAI Engineer level.

---

## WHY BASIC RAG FAILS (THE PROBLEM THIS FILE SOLVES)

Basic RAG (chunk → embed → retrieve → generate) breaks in 4 real situations:

```
PROBLEM 1: Multi-hop questions
  Q: "What did the CEO who founded the company that makes the GPU
      used in GPT-3 training say about AI safety?"
  Basic RAG: finds 1 chunk about GPT-3 GPUs, misses the rest
  Need: multi-step retrieval

PROBLEM 2: Document-level understanding
  Q: "Summarize the key themes across all 500 pages of this report"
  Basic RAG: retrieves 5 chunks, misses the big picture
  Need: hierarchical understanding

PROBLEM 3: Hallucination in retrieval
  Q: Retrieved document is partially relevant, model hallucinates the gap
  Basic RAG: no mechanism to catch this
  Need: self-verification

PROBLEM 4: Complex knowledge graphs
  Q: "How are Tesla, SpaceX, and Neuralink connected through their investors?"
  Basic RAG: finds isolated facts, misses relationships
  Need: graph-based retrieval
```

Each advanced RAG technique was designed to solve one of these problems.

---

## 1. RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval

**Paper:** Sarthi et al., Stanford, 2024
**Problem it solves:** Retrieving both fine-grained facts AND high-level themes from long documents

### The Core Idea

Standard RAG only retrieves at one granularity — the chunk level.
RAPTOR builds a **tree of summaries** at multiple levels of abstraction.

```
Level 0 (raw chunks):
  [Chunk 1][Chunk 2][Chunk 3][Chunk 4][Chunk 5][Chunk 6][Chunk 7][Chunk 8]

Level 1 (cluster summaries):
  [Summary A = clusters 1,2,3]  [Summary B = clusters 4,5]  [Summary C = 6,7,8]

Level 2 (high-level summary):
  [Root Summary = Summary A + B + C]
```

### How It Works — Step by Step

**Step 1: Chunk the document** (same as basic RAG)

**Step 2: Embed all chunks** using an embedding model

**Step 3: Cluster similar chunks** using Gaussian Mixture Models (GMM)
- Unlike K-Means (hard assignment), GMM allows soft assignments
- A chunk about "transformer attention" can partially belong to both
  "architecture" cluster and "optimization" cluster
- Use UMAP first to reduce dimensions before clustering (reduces noise)

**Step 4: Summarize each cluster** using an LLM
- Each cluster gets summarized into a single summary node
- Summary captures the main theme of that group of chunks

**Step 5: Repeat recursively** until you have one root summary
- Embed the summaries → cluster them → summarize clusters
- Repeat until only 1 node remains (the root)

**Step 6: Index ALL nodes** (raw chunks + all summary levels) in the vector DB

### Retrieval at Query Time

Two modes:

**Tree Traversal (top-down):**
```
Start at root → if relevant, go deeper → retrieve final leaf nodes
```
Good for: structured documents with clear hierarchy

**Collapsed Tree (flat search across all levels):**
```
Query → search ALL nodes at once (chunks + summaries)
```
Better in practice — finds the right abstraction level automatically

### Why RAPTOR Beats Basic RAG

```
Q: "What are the main themes in this 200-page annual report?"
Basic RAG: retrieves 5 random chunks → misses big picture
RAPTOR:    retrieves Level-2 summary → captures full document themes

Q: "What was the Q3 revenue for the North America segment?"
RAPTOR:    retrieves Level-0 chunk → precise fact
```

RAPTOR handles BOTH types of questions with one system.

### Production Usage
Used in long-document QA, legal document analysis, financial report systems.
LlamaIndex and LangChain both have RAPTOR implementations.

---

## 2. GraphRAG — Microsoft, 2024

**Paper:** Edge et al., Microsoft Research, 2024
**Problem it solves:** Questions requiring understanding of relationships and community-level patterns

### The Core Idea

Text documents contain entities and their relationships.
Standard RAG treats documents as bags of text.
GraphRAG builds a **knowledge graph** from the documents and queries it.

```
Document: "Elon Musk founded Tesla in 2003 with JB Straubel.
          Tesla supplies batteries to SpaceX for power storage."

Knowledge Graph:
  [Elon Musk] --FOUNDED--> [Tesla]
  [Tesla] --FOUNDED_WITH--> [JB Straubel]
  [Tesla] --SUPPLIES--> [SpaceX]
  [Elon Musk] --FOUNDED--> [SpaceX]
```

### How It Works — Step by Step

**Phase 1: Graph Construction (Offline)**

**Step 1: Entity & Relationship Extraction**
- Use LLM to extract entities (people, places, orgs, concepts) from each chunk
- Extract relationships between entities
- Prompt: "Extract all entities and relationships from this text as JSON"

**Step 2: Entity Resolution**
- Merge duplicate entities: "Musk", "Elon Musk", "Mr. Musk" → same node
- Use embedding similarity to detect duplicates

**Step 3: Community Detection (Leiden Algorithm)**
- Run Leiden community detection algorithm on the graph
- Finds clusters of highly interconnected entities
- Example communities: {Tesla, SpaceX, Neuralink, Elon Musk} = one community

**Step 4: Community Summarization**
- LLM generates a summary for each community
- "This community is about Elon Musk's technology ventures..."
- Summaries stored at multiple levels (hierarchical communities)

**Phase 2: Query (Online)**

**Two query modes:**

**Local Search:**
```
Q: "What did Elon Musk say about AI safety?"
→ Find Elon Musk entity in graph
→ Find all related entities and relationships
→ Retrieve chunks mentioning Elon Musk + neighbors
→ Feed to LLM with graph context
```
Good for: specific entity-focused questions

**Global Search:**
```
Q: "What are the major themes in this document collection?"
→ Retrieve community summaries at the right level
→ Map-reduce: LLM generates partial answers from each community
→ Reduce: LLM combines partial answers into final answer
```
Good for: holistic, thematic questions across the entire document collection

### When GraphRAG Beats Basic RAG

```
Q: "How are the major AI labs interconnected through funding?"
Basic RAG: finds 5 isolated chunks → misses connections
GraphRAG: traverses the entity graph → finds all connections directly

Q: "What are the key controversies surrounding AI development?"
Basic RAG: retrieves specific facts
GraphRAG: community summaries capture the full landscape
```

### Production Considerations

- Graph construction is expensive (LLM calls per chunk for extraction)
- Best for static or slowly changing document collections
- Microsoft open-sourced the full GraphRAG library (pip install graphrag)
- Latency: global queries are slower (map-reduce over community summaries)

---

## 3. SELF-RAG — Self-Reflective Retrieval Augmented Generation

**Paper:** Asai et al., University of Washington, 2023
**Problem it solves:** Retrieval is always done even when unnecessary; retrieved docs may be irrelevant or unsupported

### The Core Idea

Standard RAG always retrieves, always uses what it retrieves, never checks if it's correct.
Self-RAG trains the model to **decide when to retrieve** and **verify its own outputs**.

```
Standard RAG:  Question → Always Retrieve → Always Generate
Self-RAG:      Question → Decide if Retrieve Needed → Maybe Retrieve
                        → Check if Retrieved Doc is Relevant
                        → Generate → Check if Answer is Supported
                        → Check if Answer is Useful
```

### The Four Special Tokens (THIS IS THE KEY)

Self-RAG fine-tunes a model to generate 4 special "reflection tokens":

| Token | Type | Values | Meaning |
|-------|------|--------|---------|
| `[Retrieve]` | Retrieval decision | Yes / No / Continue | Should we retrieve? |
| `[ISREL]` | Relevance check | Relevant / Irrelevant | Is retrieved doc relevant? |
| `[ISSUP]` | Support check | Fully / Partially / No | Does doc support the claim? |
| `[ISUSE]` | Usefulness check | 5 / 4 / 3 / 2 / 1 | How useful is this response? |

### The Generation Process

```
Input: "What is the population of Tokyo?"

Model decides: [Retrieve=Yes]  ← retrieval is needed

Retrieve docs about Tokyo

For each retrieved doc:
  Model decides: [ISREL=Relevant]  ← this doc is relevant

Generate answer segment using doc

Model decides: [ISSUP=Fully Supported]  ← doc fully backs this claim
Model decides: [ISUSE=5]  ← this response is very useful

Final output: "Tokyo's population is approximately 13.96 million
              in the city proper, and 37.4 million in the
              greater metropolitan area."
```

### When Retrieval is SKIPPED:

```
Input: "What is 2 + 2?"
Model decides: [Retrieve=No]  ← don't waste time retrieving
Model generates directly: "4"
```

This is huge — standard RAG would retrieve documents about arithmetic. Wasteful.

### Training Self-RAG

Uses standard supervised fine-tuning on data augmented with reflection tokens:
1. Start with existing QA dataset
2. Run retrieval, check relevance, check support (with another LLM)
3. Insert appropriate reflection tokens into training examples
4. Fine-tune the model on this augmented data

At inference: the model generates reflection tokens naturally alongside content.

### Self-RAG Beam Search

At inference, you can do segment-level beam search:
- Generate multiple continuations
- Score them by ISUSE token
- Keep the highest-scoring path

---

## 4. CORRECTIVE RAG (CRAG)

**Paper:** Yan et al., 2024
**Problem it solves:** Retrieved documents are low-quality or incorrect

### The Core Idea

Standard RAG blindly trusts retrieved documents.
What if the top retrieved doc is wrong or outdated?
CRAG adds a **retrieval evaluator** and a **web search fallback**.

### The Three Actions

After retrieval, CRAG scores the relevance of each retrieved document:

```
Score = Relevance Evaluator (query, retrieved_doc)
```

Based on the score, one of three actions:

**CORRECT (high score):**
```
Retrieved doc is relevant and trustworthy
→ "Knowledge Refinement": extract key info, strip noise
→ Use refined knowledge for generation
```

**INCORRECT (low score):**
```
Retrieved doc is not relevant
→ Fallback to web search (Google/Bing API)
→ Use web search results instead
```

**AMBIGUOUS (medium score):**
```
Uncertain quality
→ Do BOTH: use local docs AND web search
→ Combine knowledge from both sources
```

### Knowledge Refinement

Even when the doc is correct, CRAG extracts only relevant parts:
```
Long document about Paris (10,000 words)
Query: "What year was the Eiffel Tower built?"

Without refinement: Send entire document to LLM
With refinement:    Extract → "The Eiffel Tower was built in 1887-1889"
                    Send only this snippet to LLM
```

Reduces noise, improves accuracy, saves context window.

### Production Value

CRAG is valuable when:
- Your vector DB might have outdated information
- User queries span topics not well covered in your knowledge base
- You want fallback to real-time web search for fresh information

---

## 5. AGENTIC RAG — The Most Powerful Pattern

**Problem it solves:** Complex multi-hop questions requiring multiple retrieval steps

### The Core Idea

Instead of one retrieval → one generation, use an LLM agent that:
1. Plans what to retrieve
2. Retrieves iteratively
3. Reasons over retrieved info
4. Decides if more retrieval is needed
5. Generates final answer

```
Standard RAG (1 round):
  Q: "Who is the CEO of the company that acquired DeepMind?"
  → Retrieve "DeepMind" → Answer "Google acquired it" → Done (incomplete)

Agentic RAG (multi-round):
  Q: "Who is the CEO of the company that acquired DeepMind?"
  Step 1: Retrieve "DeepMind acquisition"
          → Found: "Google/Alphabet acquired DeepMind in 2014"
  Step 2: Retrieve "Alphabet CEO"
          → Found: "Sundar Pichai is CEO of Alphabet"
  Answer: "Sundar Pichai"
```

### ReAct Loop for Agentic RAG

```
Thought: I need to find who acquired DeepMind, then find that company's CEO
Action: search[DeepMind acquisition]
Observation: Google/Alphabet acquired DeepMind in 2014 for ~$500M

Thought: Now I need to find Alphabet's current CEO
Action: search[Alphabet CEO 2024]
Observation: Sundar Pichai has been CEO of Alphabet since 2015

Thought: I have enough information to answer
Answer: Sundar Pichai is the CEO of Alphabet, the company that acquired DeepMind
```

### Sub-Query Decomposition

Complex questions broken into sub-queries:
```
Q: "Compare the attention mechanisms used in BERT and GPT-3"

Decompose:
  Sub-query 1: "What attention mechanism does BERT use?"
  Sub-query 2: "What attention mechanism does GPT-3 use?"
  Sub-query 3: "What are the key differences?"

Retrieve for each sub-query independently
Combine results
Generate comparative answer
```

### Tools Available to Agentic RAG

```python
tools = [
    search_vector_db,      # semantic search in your knowledge base
    web_search,            # real-time web search
    sql_query,             # query structured databases
    calculator,            # math operations
    code_executor,         # run Python code
    api_caller,            # call external APIs
]
```

### When to Use Agentic RAG

Use Agentic RAG when:
- Questions require 2+ retrieval steps
- Answers require combining info from multiple sources
- Some queries need real-time data (web) and others don't
- Complex reasoning over retrieved information is needed

Use Standard RAG when:
- Questions are single-hop
- Low latency is critical (each agent step adds 1-2 seconds)
- Cost matters (each step uses LLM tokens)

---

## 6. MODULAR RAG — The Framework That Unifies Everything

**Paper:** Gao et al., 2023

### The Idea

RAG is not a single system — it's a composition of modules:

```
[Query Module]     → Query rewriting, query decomposition, HyDE
     ↓
[Retrieval Module] → Dense, sparse, hybrid, web search
     ↓
[Reranking Module] → Cross-encoder, LLM reranker, ColBERT
     ↓
[Context Module]   → Compression, selection, fusion
     ↓
[Generation Module]→ LLM, structured output, citation
     ↓
[Feedback Module]  → Self-RAG verification, CRAG correction
```

Each module can be swapped independently.
This is how production RAG systems are built — composable, testable.

---

## 7. CONTEXTUAL RETRIEVAL (Anthropic, 2024)

### The Problem With Standard Chunking

A chunk without context loses meaning:
```
Chunk: "The company's revenue increased by 23% year-over-year."
Problem: Which company? Which year? What metric?
Without the document context, this chunk is ambiguous.
```

### The Solution

Before indexing, prepend each chunk with **LLM-generated context** about where it fits:

```
Original chunk:
"The company's revenue increased by 23% year-over-year."

Contextual chunk:
"[Context: This excerpt is from Apple Inc.'s Q3 2024 earnings report,
discussing the iPhone segment performance.]
The company's revenue increased by 23% year-over-year."
```

### Production Implementation

```python
system_prompt = """You are analyzing a document. Given the full document 
and a chunk, write a brief context (2-3 sentences) that situates this 
chunk within the document. Focus on what makes this chunk unique and 
what broader context is needed to understand it."""

for chunk in chunks:
    context = llm.generate(
        f"Document: {full_document}\n\nChunk: {chunk}\n\nContext:"
    )
    contextual_chunk = f"{context}\n\n{chunk}"
    embed_and_store(contextual_chunk)
```

**Cost:** ~800 tokens per chunk (document + chunk → context)
**Benefit:** 49% reduction in retrieval failures (per Anthropic's benchmarks)

---

## INTERVIEW BLAST — Advanced RAG

**"What is RAPTOR and when would you use it?"**
> "RAPTOR recursively clusters document chunks, summarizes each cluster with an LLM,
> then clusters and summarizes the summaries — building a tree of abstractions.
> At query time, you search across all levels. For 'What was the revenue?' you get a
> leaf chunk. For 'What are the main themes?' you get a high-level summary. I'd use
> it for long documents requiring both specific fact retrieval and thematic understanding —
> annual reports, legal documents, research papers."

**"What is GraphRAG and how is it different from standard RAG?"**
> "Standard RAG treats documents as isolated text chunks. GraphRAG first extracts
> entities and relationships to build a knowledge graph, then runs community detection
> to find clusters of related entities and summarizes each community. For local queries
> about specific entities, it traverses the graph. For global queries about themes,
> it does map-reduce over community summaries. It's superior for questions about
> relationships and patterns across a large document collection."

**"What is Self-RAG?"**
> "Self-RAG fine-tunes a model to generate special reflection tokens during generation:
> RETRIEVE (should I retrieve?), ISREL (is the doc relevant?), ISSUP (does the doc
> support my claim?), ISUSE (is my answer useful?). This makes retrieval adaptive —
> factual questions retrieve, math questions don't. And the model self-critiques its
> outputs, reducing hallucination. You get the best of RAG and direct generation."
