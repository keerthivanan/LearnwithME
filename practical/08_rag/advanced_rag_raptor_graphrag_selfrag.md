# Advanced RAG — RAPTOR, GraphRAG, Self-RAG, CRAG, Agentic RAG

> Basic RAG gets you in the door. This file gets you the job.
> Every advanced RAG technique asked at 2-4 year GenAI Engineer level.

---

## WHY BASIC RAG FAILS (THE PROBLEM THIS FILE SOLVES)

**What it is:** Basic RAG (chunk → embed → retrieve → generate) is good but has four specific failure modes. Each advanced technique was invented to solve one of these failures. Understanding the failure first makes every technique make sense.

Basic RAG (chunk → embed → retrieve → generate) breaks in 4 real situations:

```
PROBLEM 1: Multi-hop questions
  Q: "What did the CEO who founded the company that makes the GPU
      used in GPT-3 training say about AI safety?"
  Basic RAG: finds 1 chunk about GPT-3 GPUs, misses the rest
  Need: multi-step retrieval across multiple documents

PROBLEM 2: Document-level understanding
  Q: "Summarize the key themes across all 500 pages of this report"
  Basic RAG: retrieves 5 chunks, misses the big picture
  Need: hierarchical understanding at multiple granularities

PROBLEM 3: Hallucination in retrieval
  Q: Retrieved document is only partially relevant — model hallucinates the gap
  Basic RAG: no mechanism to catch or correct this
  Need: self-verification of retrieved documents and generated answers

PROBLEM 4: Complex knowledge graphs
  Q: "How are Tesla, SpaceX, and Neuralink connected through their investors?"
  Basic RAG: finds isolated facts about each company, misses the relationships
  Need: graph-based retrieval that traverses entity relationships
```

Each advanced RAG technique was designed to solve one of these problems.

---

## 1. RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval

**What it is:** RAPTOR builds a tree of summaries at multiple levels of abstraction. Instead of only retrieving at the raw chunk level, you can also retrieve cluster summaries (medium abstraction) or document-level summaries (high abstraction). This solves Problem 2: document-level understanding.

**Analogy:** A table of contents in a book has multiple levels: part → chapter → section → paragraph. RAPTOR builds this hierarchy automatically from any document. You can search at any level — find the exact paragraph for a specific fact, or find the high-level chapter summary for a broad theme question.

**Paper:** Sarthi et al., Stanford, 2024
**Problem it solves:** Retrieving both fine-grained facts AND high-level themes from long documents

### The Core Idea

Standard RAG only retrieves at one granularity — the raw chunk level.
RAPTOR builds a **tree of summaries** at multiple levels of abstraction.

```
Level 0 (raw chunks — most specific):
  [Chunk 1][Chunk 2][Chunk 3][Chunk 4][Chunk 5][Chunk 6][Chunk 7][Chunk 8]

Level 1 (cluster summaries — medium abstraction):
  [Summary A = summarizes chunks 1,2,3]  [Summary B = chunks 4,5]  [Summary C = 6,7,8]

Level 2 (high-level summary — most abstract):
  [Root Summary = Summary of Summary A + B + C — captures the whole document]
```

**WHY:** The same index now answers both "what was the Q3 revenue?" (needs a specific chunk) and "what were the major themes of the annual report?" (needs the root summary). One system, two levels of precision.

### How It Works — Step by Step

**Step 1: Chunk the document** (same as basic RAG — split into ~500 token pieces)

**Step 2: Embed all chunks** using an embedding model (same as basic RAG)

**Step 3: Cluster similar chunks** using Gaussian Mixture Models (GMM)
- Unlike K-Means (which assigns each chunk to exactly one cluster), GMM allows **soft assignments**
- A chunk about "transformer attention" can partially belong to both "architecture" cluster AND "optimization" cluster
- Use UMAP first to reduce embedding dimensions before clustering — reduces noise, improves cluster quality

**WHY GMM instead of K-Means:** Real chunks often straddle multiple topics. K-Means forces a hard choice. GMM assigns fractional membership, so a chunk about "attention optimization" contributes to both the "attention" cluster summary and the "optimization" cluster summary. The summaries are richer.

**Step 4: Summarize each cluster** using an LLM
- Each cluster of related chunks gets summarized into a single summary node
- The summary captures the main theme of that group of chunks in 2–5 sentences

**Step 5: Repeat recursively** until you have one root summary
- Embed the summaries → cluster them → summarize the clusters of summaries
- Repeat until only one node remains (the root — the full document summary)

**Step 6: Index ALL nodes** (raw chunks + all summary levels) in the vector DB
- Every node at every level of the tree gets its own embedding and is searchable

### Retrieval at Query Time

**Two modes:**

**Tree Traversal (top-down):**
```
Start at root → if root summary is relevant, go deeper into children
→ retrieve the final leaf chunks that are relevant
```
Good for: structured documents with clear hierarchy (legal contracts, technical manuals)

**Collapsed Tree (flat search across all levels):**
```
Query → search ALL nodes at once (chunks + all summary levels simultaneously)
→ the most relevant node at any abstraction level is returned
```
Better in practice — automatically finds the right abstraction level for each query

**WHY collapsed tree works better:** For most queries, you don't know in advance whether you need a specific fact (leaf) or a broad summary (root). Searching all levels simultaneously lets the query itself determine the right level — specific fact queries naturally score highest on leaf chunks, broad theme queries score highest on summary nodes.

### Why RAPTOR Beats Basic RAG

```
Q: "What are the main themes in this 200-page annual report?"
  Basic RAG: retrieves 5 random chunks → misses big picture → hallucinated summary
  RAPTOR:    retrieves Level-2 root summary → captures full document themes accurately

Q: "What was the Q3 revenue for the North America segment?"
  Basic RAG: might retrieve the right chunk (works fine for specific facts)
  RAPTOR:    also retrieves the exact Level-0 chunk → precise fact
```

RAPTOR handles BOTH types of questions with one system.

### Production Usage
Used in long-document QA, legal document analysis, financial report systems.
LlamaIndex and LangChain both have RAPTOR implementations.

---

## 2. GraphRAG — Microsoft, 2024

**What it is:** GraphRAG extracts entities (people, places, companies, concepts) and relationships from documents, builds a knowledge graph, and queries the graph instead of (or in addition to) raw text chunks. This solves Problem 4: questions about relationships and connections.

**Analogy:** Standard RAG stores documents as individual pages in a filing cabinet. GraphRAG is like building a relationship map — showing that "Elon Musk → founded → Tesla," "Tesla → supplies batteries to → SpaceX." When you ask "how are Tesla and SpaceX connected?", you query the map, not the cabinet.

**Paper:** Edge et al., Microsoft Research, 2024
**Problem it solves:** Questions requiring understanding of relationships and community-level patterns

### The Core Idea

Text documents contain entities and their relationships.
Standard RAG treats documents as bags of text (no structure).
GraphRAG builds a **knowledge graph** from the documents and queries that graph.

```
Document: "Elon Musk founded Tesla in 2003 with JB Straubel.
          Tesla supplies batteries to SpaceX for power storage."

Knowledge Graph built from this:
  [Elon Musk] --FOUNDED--> [Tesla]
  [Tesla] --FOUNDED_WITH--> [JB Straubel]
  [Tesla] --SUPPLIES--> [SpaceX]
  [Elon Musk] --FOUNDED--> [SpaceX]
```

### How It Works — Step by Step

**Phase 1: Graph Construction (Offline — done once)**

**Step 1: Entity & Relationship Extraction**
- Run an LLM on each chunk: "Extract all entities (people, places, organizations, concepts) and relationships from this text as JSON"
- Each chunk contributes entities and edges to the global graph

**Step 2: Entity Resolution**
- Merge duplicate entities that refer to the same thing: "Musk", "Elon Musk", "Mr. Musk" → one node
- Use embedding similarity to detect these near-duplicates automatically

**WHY entity resolution matters:** Without it, you have three separate nodes for the same person. Queries about "Elon Musk" will miss connections described using just "Musk." Merging ensures the graph is coherent.

**Step 3: Community Detection (Leiden Algorithm)**
- Run the Leiden community detection algorithm on the graph
- Finds clusters of highly interconnected entities
- Example community automatically discovered: {Tesla, SpaceX, Neuralink, Elon Musk, Boring Company} = one community

**Step 4: Community Summarization**
- LLM generates a human-readable summary for each community at multiple levels
- "This community is about Elon Musk's technology ventures, spanning electric vehicles (Tesla), space exploration (SpaceX), and neurotechnology (Neuralink)..."
- Summaries are stored at multiple hierarchy levels (communities can contain sub-communities)

**Phase 2: Query (Online — for every query)**

**Two query modes:**

**Local Search:**
```
Q: "What did Elon Musk say about AI safety?"
→ Find Elon Musk entity node in the graph
→ Find all entities connected to Elon Musk within 1–2 hops
→ Retrieve original text chunks mentioning Elon Musk and his neighbors
→ Feed to LLM with both the graph context and the retrieved chunks
```
Best for: specific entity-focused questions

**Global Search:**
```
Q: "What are the major themes in this document collection?"
→ Retrieve community summaries at the right level of hierarchy
→ Map step: LLM generates a partial answer from each community summary
→ Reduce step: LLM combines all partial answers into a final comprehensive answer
```
Best for: holistic, thematic questions that span the entire document collection

### When GraphRAG Beats Basic RAG

```
Q: "How are the major AI labs interconnected through funding?"
  Basic RAG: finds 5 isolated chunks about different labs → misses connections
  GraphRAG: traverses the entity graph → finds all funding relationships directly

Q: "What are the key controversies surrounding AI development?"
  Basic RAG: retrieves specific facts from a few chunks
  GraphRAG: community summaries capture the full landscape across all documents
```

### Production Considerations

- Graph construction is expensive (one LLM call per chunk for entity extraction — can be 1000s of calls)
- Best for static or slowly changing document collections (legal, financial reports, research papers)
- Microsoft open-sourced the full GraphRAG library: `pip install graphrag`
- Global queries are slower than basic RAG (map-reduce over community summaries requires many LLM calls)

---

## 3. SELF-RAG — Self-Reflective Retrieval Augmented Generation

**What it is:** Self-RAG fine-tunes a model to use special "reflection tokens" during generation. These tokens let the model decide: should I retrieve? Is this retrieved document actually relevant? Is my answer actually supported by the evidence? Does my answer actually help?

**Analogy:** Standard RAG is like a student who always opens the textbook, reads whatever page falls open, and writes their answer regardless of whether the textbook was helpful. Self-RAG is a student who first asks "do I need to look this up?", then checks "is this the right section?", then checks "does my answer match what I read?"

**Paper:** Asai et al., University of Washington, 2023
**Problem it solves:** Retrieval is always triggered even when unnecessary; retrieved docs may be irrelevant; generated answers may be unsupported by evidence

### The Core Idea

Standard RAG always retrieves, always uses what it retrieves, never checks if it is correct.
Self-RAG trains the model to **decide when to retrieve** and **verify its own outputs**.

```
Standard RAG:  Question → Always Retrieve → Always Generate
Self-RAG:      Question → [Decide if Retrieve Needed] → Maybe Retrieve
                        → [Check if Retrieved Doc is Relevant]
                        → Generate → [Check if Answer is Supported by Evidence]
                        → [Check if Answer is Useful]
```

### The Four Special Tokens (THIS IS THE KEY)

**What it is:** Self-RAG fine-tunes the model to generate four special "reflection tokens" naturally alongside its normal output. These tokens are the model's internal decision-making process made visible and trainable.

Self-RAG fine-tunes a model to generate 4 special "reflection tokens":

| Token | Type | Values | Meaning |
|-------|------|--------|---------|
| `[Retrieve]` | Retrieval decision | Yes / No / Continue | Should we retrieve for this question? |
| `[ISREL]` | Relevance check | Relevant / Irrelevant | Is this retrieved document actually relevant? |
| `[ISSUP]` | Support check | Fully / Partially / No | Does the document support my generated claim? |
| `[ISUSE]` | Usefulness check | 5 / 4 / 3 / 2 / 1 | How useful is this response overall? (1–5 scale) |

### The Generation Process — Factual Question

```
Input: "What is the population of Tokyo?"

1. Model generates: [Retrieve=Yes]   ← model decides retrieval IS needed for facts
2. System retrieves documents about Tokyo's population
3. For each retrieved document, model generates: [ISREL=Relevant]  ← doc is relevant
4. Model generates the answer text: "Tokyo's population is approximately 13.96 million..."
5. Model generates: [ISSUP=Fully Supported]  ← the retrieved doc fully backs this claim
6. Model generates: [ISUSE=5]  ← this is a complete, useful response
```

### When Retrieval is SKIPPED

```
Input: "What is 2 + 2?"
Model generates: [Retrieve=No]  ← math doesn't need document retrieval
Model generates directly: "4"
```

**WHY this matters:** Standard RAG would retrieve documents about arithmetic, wasting time and potentially retrieving irrelevant content. Self-RAG correctly identifies that no retrieval is needed. This makes the system more efficient and avoids the noise from unnecessary retrieval.

### Training Self-RAG

Uses standard supervised fine-tuning on data augmented with reflection tokens:
1. Start with an existing QA dataset (question, answer, source document)
2. Run retrieval for each question, check relevance, check support (using a separate LLM as the oracle)
3. Insert the appropriate reflection tokens into the training examples
4. Fine-tune the target model on this augmented data

At inference: the model generates reflection tokens naturally alongside content — no external oracle needed.

### Self-RAG Beam Search

**What it is:** At inference time, you can run segment-level beam search where multiple possible continuations are generated, scored by the ISUSE token, and the highest-scoring path is kept.

At inference, run segment-level beam search:
- Generate multiple possible next segments
- Score each by the ISUSE (usefulness) token
- Keep the highest-scoring path and continue generating from there

**WHY:** This is like a chess engine that evaluates multiple move sequences. Self-RAG can consider "what if I retrieve?" vs "what if I don't?" and keep the path that produces the most useful answer.

---

## 4. CORRECTIVE RAG (CRAG)

**What it is:** CRAG adds a quality check on retrieved documents. If the retrieved documents are not relevant, CRAG falls back to web search instead of blindly using bad documents. This solves Problem 3: hallucination when retrieved documents are wrong or outdated.

**Analogy:** Standard RAG is like a student who uses whatever book the librarian hands them, even if it is the wrong book. CRAG is a student who first checks "is this book actually about my topic?" and, if not, says "let me try Google instead."

**Paper:** Yan et al., 2024
**Problem it solves:** Retrieved documents are low-quality, outdated, or incorrect — and the system uses them anyway

### The Core Idea

Standard RAG blindly trusts retrieved documents, regardless of quality.
What if the top retrieved document is outdated or simply wrong?
CRAG adds a **retrieval evaluator** and a **web search fallback**.

### The Three Actions

After retrieval, CRAG scores the relevance of each retrieved document:

```
Score = Relevance_Evaluator(query, retrieved_doc)   ← a lightweight model scores relevance
```

Based on the score, one of three actions is taken:

**CORRECT (high relevance score):**
```
Retrieved doc is relevant and trustworthy
→ "Knowledge Refinement": extract only the key information from the doc, strip noise
→ Use refined knowledge as context for generation
```

**INCORRECT (low relevance score):**
```
Retrieved doc is not relevant to the query
→ Fallback to web search (Google/Bing API)
→ Use web search results as context instead of the local vector DB results
```

**AMBIGUOUS (medium relevance score):**
```
Uncertain quality — the doc might be relevant
→ Do BOTH: use local retrieved docs AND web search results
→ Combine knowledge from both sources for generation
```

### Knowledge Refinement

**What it is:** Even when the retrieved document IS correct, it is often long and contains much more than what is needed. Knowledge refinement extracts only the relevant sentences/paragraphs before passing to the LLM.

Even when the doc is correct, CRAG extracts only relevant parts:
```
Long document: entire Wikipedia article about Paris (10,000 words, covers history, culture, geography)
Query: "What year was the Eiffel Tower built?"

Without refinement: send all 10,000 words to LLM → wastes tokens, adds noise
With refinement:    extract → "The Eiffel Tower was built between 1887–1889"
                    send only this 8-word snippet → precise, cheap, accurate
```

**WHY:** Sending the full document wastes LLM context window and can confuse the model with irrelevant information (this is called "lost in the middle" — models pay less attention to information in the middle of long contexts). Refinement ensures only the needle is passed, not the entire haystack.

### Production Value

CRAG is valuable when:
- Your vector DB might have outdated information (docs not refreshed in months)
- User queries span topics not well covered in your knowledge base
- You want automatic fallback to real-time web search for fresh information

---

## 5. AGENTIC RAG — The Most Powerful Pattern

**What it is:** Instead of a single retrieval step, an LLM agent plans, retrieves multiple times, reasons over the results, and decides when it has enough information to answer. This solves Problem 1: multi-hop questions that require multiple retrieval steps.

**Analogy:** Standard RAG is like a student who reads one textbook and writes an answer. Agentic RAG is like a research team: they identify what they need to find, search multiple sources, cross-reference the results, and synthesize a comprehensive answer.

**Problem it solves:** Complex multi-hop questions requiring multiple retrieval steps

### The Core Idea

Instead of one retrieval → one generation (standard RAG), use an LLM agent that:
1. Plans what information is needed to answer the question
2. Retrieves iteratively, one step at a time
3. Reasons over the retrieved information
4. Decides if more retrieval is needed (or if enough info is available)
5. Generates the final comprehensive answer

```
Standard RAG (1 round):
  Q: "Who is the CEO of the company that acquired DeepMind?"
  → Retrieve "DeepMind" → Answer "Google acquired it" → Done (INCOMPLETE — missed the CEO)

Agentic RAG (multi-round):
  Q: "Who is the CEO of the company that acquired DeepMind?"
  Round 1: Retrieve "DeepMind acquisition"
           → Found: "Google/Alphabet acquired DeepMind in 2014"
           → Thought: "Now I need to find Alphabet's CEO, not Google's original CEO"
  Round 2: Retrieve "Alphabet CEO 2024"
           → Found: "Sundar Pichai is CEO of Alphabet since 2015"
  Answer: "Sundar Pichai (CEO of Alphabet, which acquired DeepMind in 2014)"
```

### ReAct Loop for Agentic RAG

**What it is:** ReAct (Reason + Act) is the pattern where the LLM alternates between reasoning (Thought) and acting (Action), observing the result, and continuing. This makes the reasoning process transparent and debuggable.

```
Thought: I need to find who acquired DeepMind, then find that company's current CEO.
         I'll start with the acquisition.
Action: search[DeepMind acquisition]
Observation: Google/Alphabet acquired DeepMind in 2014 for approximately $500M.

Thought: Google is the acquirer. But Alphabet is the parent company — I should check
         Alphabet's CEO specifically, not Google's original founders.
Action: search[Alphabet CEO 2024]
Observation: Sundar Pichai has been CEO of Alphabet since its formation in 2015.

Thought: I now have both pieces of information needed to answer completely.
Answer: Sundar Pichai is the CEO of Alphabet, the company that acquired DeepMind in 2014.
```

**WHY ReAct is powerful:** The "Thought" step before each action allows the LLM to reason about *what* to search for and *why*. This is fundamentally different from fixed retrieval — the agent adapts its search strategy based on what it has already found.

### Sub-Query Decomposition

**What it is:** Break a complex question into simpler sub-questions, retrieve for each independently, then combine all results to generate a comprehensive comparative answer.

Complex questions broken into sub-queries:
```
Q: "Compare the attention mechanisms used in BERT and GPT-3"

Decompose into independent sub-queries:
  Sub-query 1: "What attention mechanism does BERT use?"      → bidirectional self-attention
  Sub-query 2: "What attention mechanism does GPT-3 use?"    → causal (masked) self-attention
  Sub-query 3: "Key differences between bidirectional and causal attention"

Retrieve for each sub-query independently, then:
Combine all results + generate comparative answer
```

**WHY:** A single query "Compare BERT and GPT-3 attention" retrieves documents that happen to mention both in the same passage. Sub-query decomposition retrieves the best document for each aspect independently, giving the model richer information for each dimension of the comparison.

### Tools Available to Agentic RAG

**What it is:** An agentic RAG system can use multiple tools beyond just vector search. The agent decides which tool to use at each step.

```python
# The set of tools the agentic RAG system can call
tools = [
    search_vector_db,      # semantic search in your private knowledge base
    web_search,            # real-time web search (Google/Bing API) for fresh information
    sql_query,             # query structured databases for precise numerical data
    calculator,            # arithmetic operations (avoids LLM math errors)
    code_executor,         # run Python code and return results
    api_caller,            # call external APIs (weather, stock prices, etc.)
]
```

**WHY multiple tools:** Different question types need different tools. "What was our Q3 revenue?" needs SQL. "What is the latest news about OpenAI?" needs web search. "What is the integral of x²?" needs a calculator. The agent chooses the right tool for each step.

### When to Use Agentic RAG

Use Agentic RAG when:
- Questions require 2+ retrieval steps (multi-hop reasoning)
- Answers require combining information from multiple sources
- Some queries need real-time data (web) and others use the private knowledge base
- Complex reasoning over retrieved information is needed (not just fact lookup)

Use Standard RAG when:
- Questions are single-hop (one retrieval step is sufficient)
- Low latency is critical (each agent step adds 1–2 seconds)
- Cost matters (each step uses LLM tokens for reasoning)

**WHY:** Agentic RAG is slow and expensive because every "Thought" step requires an LLM call. A simple "what is the refund policy?" question does not need 4 retrieval steps and 4 LLM reasoning calls. Use agentic RAG for complex questions, standard RAG for simple ones.

---

## 6. MODULAR RAG — The Framework That Unifies Everything

**What it is:** Modular RAG is a conceptual framework that views RAG as a pipeline of swappable modules. Instead of one monolithic RAG system, you have separate, independently upgradeable modules for each stage.

**Analogy:** A car has a modular design — you can upgrade the engine, transmission, and brakes independently. You don't have to rebuild the entire car to get better brakes. Modular RAG applies this engineering philosophy to retrieval systems.

**Paper:** Gao et al., 2023

### The Idea

RAG is not a single system — it's a composition of swappable modules:

```
[Query Module]     → Query rewriting, query decomposition, HyDE
     ↓
[Retrieval Module] → Dense, sparse, hybrid, web search, SQL
     ↓
[Reranking Module] → Cross-encoder, LLM reranker, ColBERT
     ↓
[Context Module]   → Compression, selection, fusion
     ↓
[Generation Module]→ LLM, structured output, citation generation
     ↓
[Feedback Module]  → Self-RAG verification, CRAG correction
```

**WHY modular design matters:** Each module can be tested, benchmarked, and improved independently. You can swap the embedding model without touching the reranker. You can add a query expansion module without changing the generator. In production, this makes RAG systems maintainable and improvable.

---

## 7. CONTEXTUAL RETRIEVAL (Anthropic, 2024)

**What it is:** Before indexing, prepend each chunk with LLM-generated context explaining where that chunk fits in the document. This gives each chunk's embedding access to information it would otherwise not have.

**Analogy:** Imagine filing a newspaper article about revenue growth. Without context, the file just says "revenue grew 23%." With context, the file says "This is from Apple Inc.'s Q3 2024 earnings report, discussing iPhone segment performance. Revenue grew 23%." The second file is much easier to find when searching for Apple or Q3 2024.

### The Problem With Standard Chunking

A chunk without context loses meaning:
```
Chunk: "The company's revenue increased by 23% year-over-year."
Problem: Which company? Which year? What product segment? What revenue line?
Without document context, this chunk is ambiguous and retrieval is unreliable.
```

**WHY:** When you embed "The company's revenue increased by 23%", the embedding vector represents a general statement about revenue growth. It will match any query about revenue growth, regardless of which company or year. Adding "Apple Q3 2024" to the chunk changes the embedding to specifically represent Apple's Q3 2024 revenue, dramatically improving retrieval precision.

### The Solution

Before indexing, prepend each chunk with **LLM-generated context** about where it fits in the larger document:

```
Original chunk:
"The company's revenue increased by 23% year-over-year."

Contextual chunk (what gets embedded and stored):
"[Context: This excerpt is from Apple Inc.'s Q3 2024 earnings report,
discussing the iPhone segment performance. This comes after the discussion
of services revenue and before the geographic breakdown.]
The company's revenue increased by 23% year-over-year."
```

### Production Implementation

```python
# System prompt for the context generation LLM call
system_prompt = """You are analyzing a document. Given the full document 
and a specific chunk from that document, write a brief 2–3 sentence context 
that situates this chunk within the document. 
Focus on what makes this chunk unique and what broader context is needed 
to understand it correctly."""

# Loop over all chunks and generate context for each
for chunk in chunks:
    # Generate context: give the LLM the FULL document AND the specific chunk
    context = llm.generate(
        f"Document: {full_document}\n\nChunk: {chunk}\n\nContext:"   # full context + the specific chunk
    )
    # Prepend the generated context to the original chunk
    contextual_chunk = f"{context}\n\n{chunk}"   # context first, then the original text
    embed_and_store(contextual_chunk)              # embed the enriched chunk and store in vector DB
```

**WHY giving the full document to the LLM:** The context generation step needs to understand where the chunk fits in the whole document. "This is the third section, it follows the methodology and precedes the results" — this kind of contextual summary is only possible if the LLM can see the whole document at once.

**Cost:** ~800 tokens per chunk (full document + chunk → context generation)
**Benefit:** 49% reduction in retrieval failures (per Anthropic's published benchmarks)

---

## INTERVIEW BLAST — Advanced RAG

**"What is RAPTOR and when would you use it?"**
> "RAPTOR recursively clusters document chunks, summarizes each cluster with an LLM,
> then clusters and summarizes the summaries — building a tree of abstractions.
> At query time, you search across all levels simultaneously. For 'What was the revenue?' you get a
> specific leaf chunk. For 'What are the main themes?' you get a high-level summary node. I'd use
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
