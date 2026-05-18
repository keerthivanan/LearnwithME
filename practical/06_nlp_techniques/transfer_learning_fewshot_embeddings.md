# 07 — NLP Techniques: Transfer Learning, Few-Shot & More

> Key techniques from the JD: attention mechanisms, transfer learning, few-shot learning.

---

## 1. Transfer Learning

### What it is
Transfer learning takes knowledge learned from one task/domain and applies it to another task/domain. Instead of training a model from scratch for every new problem, you start from a model that already knows language and just teach it the new skill.

**Real-world analogy:** A person who is already a fluent English speaker doesn't need to re-learn grammar when asked to write formal business emails. They transfer their language knowledge and only learn the new style. That is exactly what transfer learning does for LLMs.

### The Pre-train → Fine-tune Paradigm

```
Stage 1: Pre-train on huge general corpus
         → Model learns: grammar, facts, reasoning, language patterns
         → Cost: $$$$ (millions of dollars, weeks of compute)

Stage 2: Fine-tune on small task-specific data
         → Adapt to specific task
         → Cost: $ (hours on a few GPUs)
```

**WHY this works:**
- Lower layers of the neural network learn general features (syntax, word meaning)
- Higher layers learn more abstract, task-specific features
- Fine-tuning updates the higher layers while preserving the lower-layer general knowledge
- Much less data is needed because the model already "knows" language — it only needs to learn the task

### In NLP History
| Era | Approach |
|-----|---------|
| Pre-2018 | Train from scratch per task, word2vec embeddings |
| 2018 (ELMo) | Contextual embeddings, still task-specific models |
| 2018 (BERT) | Unified encoder, fine-tune per task |
| 2020 (GPT-3) | Zero-shot and few-shot, no fine-tuning needed |
| 2023+ | Instruction-tuned LLMs, minimal prompting |

---

## 2. Few-Shot Learning

### What it is
The model learns how to do a new task from very few examples (sometimes zero), purely by reading those examples in the prompt. No training, no weight updates — just reading.

**Analogy:** If someone shows you 3 examples of "translate English to Pig Latin," you immediately understand the pattern and can apply it to new sentences. GPT-3 and larger LLMs can do the same thing — read a few examples and infer the pattern.

### Zero-Shot
Model performs a task without any examples. You only describe what you want.
```
"Classify if this review is positive or negative: 'I hated this product.'"
→ "negative"
```

### One-Shot
Model performs a task with exactly one example to guide it.
```
"Positive: 'Love it!' → positive
Classify: 'Worst purchase ever' →"
```

### Few-Shot (in-context learning)
Provide 3–10 examples in the prompt. No gradient update, no training.
```
"Translate English to French:
sea otter → loutre de mer
peppermint → menthe poivrée
plush girafe → girafe en peluche
cheese →"
```

### Why LLMs can do Few-Shot
GPT-3 (2020) showed that very large language models can learn from examples in context **without weight updates**. This is called **in-context learning (ICL)**.

**WHY:** The attention mechanism uses the example pairs to condition the generation. Each token in the examples becomes a key-value pair in the attention, and the model implicitly learns the pattern by attending to similar examples when generating the answer. No gradient is computed — it is all done through attention.

### Few-Shot vs Fine-Tuning Comparison
| Aspect | Few-Shot (ICL) | Fine-Tuning |
|--------|---------------|------------|
| Weight update? | No | Yes |
| Data needed | 0–10 examples in prompt | 100s–1000s |
| Cost | Just inference | Training cost |
| Flexibility | Change examples per call | Retrain to change |
| Performance | Good for large models | Better for small models |
| Persistent? | Only in context | Baked into weights |

**WHY:** For large models (70B+), few-shot often matches fine-tuning performance. For small models (7B and below), fine-tuning is significantly better because the model needs the explicit gradient signal.

---

## 3. Attention Mechanisms (Advanced)

### Why Attention is the Key Innovation

**What it is:** Before attention, encoder-decoder models (like early translation systems) compressed the entire input sentence into a single fixed-size vector. Attention lets the decoder look back at every word in the input when generating each output word.

**Analogy:** Without attention, a translator reads the whole sentence once, closes the book, and tries to translate from memory. With attention, the translator can glance back at any word of the original sentence as they translate. Obviously the second approach produces better translations.

Before attention, seq2seq models compressed the entire input into a fixed-size vector — an information bottleneck for long sequences.

Attention allows the decoder to "look back" at all encoder positions:
```
decoder_output = Σ attention_weight_i * encoder_output_i
```

### Bahdanau (Additive) Attention — Pre-Transformer
```
e_ij = v * tanh(W_a * h_i + W_b * s_j)    # alignment score: how relevant is encoder position i to decoder step j?
a_ij = softmax(e_ij)                        # normalize scores into probabilities (must sum to 1)
c_j  = Σ a_ij * h_i                        # context vector: weighted sum of all encoder states
```
- First attention mechanism (2015), used in neural machine translation
- Foundation for the Transformer's self-attention mechanism

### Self-Attention vs Cross-Attention
| Type | Q from | K,V from | Used in |
|------|--------|---------|---------|
| Self-attention | Same sequence | Same sequence | BERT encoder, GPT decoder |
| Cross-attention | Decoder | Encoder output | T5, BART |

**WHY:** Self-attention lets each token in a sequence look at all other tokens in the same sequence — this is how BERT builds up understanding of each word in context. Cross-attention connects the decoder to the encoder output — this is how T5 generates summaries by attending to the source document.

### Local vs Global Attention
| Type | Attends to | Use Case |
|------|-----------|---------|
| Global (full) | All tokens | Standard Transformers |
| Local/Sliding Window | Nearby N tokens | Long documents (Longformer) |
| Sparse | Selected positions | BigBird |
| Linear | Approximated globally | Efficient Transformers |

**WHY:** Full attention costs O(n²) in memory and compute because every token attends to every other token. For a 10,000-token document, that is 100 million attention scores. Local attention limits each token to attending only to its nearest neighbors, reducing cost to O(n × window_size).

---

## 4. Word Embeddings (Pre-Transformer NLP)

**What it is:** Word embeddings are dense numerical vectors that represent the meaning of words. The key idea is that words with similar meanings have similar vectors — so "king" and "queen" are close together, while "king" and "bicycle" are far apart.

**Analogy:** Imagine placing every word in a city. Words with similar meanings live in the same neighborhood. "Happy," "joyful," and "delighted" all live on the same street. "Bank" (financial) and "bank" (river) live in different neighborhoods entirely. That is what an embedding space looks like.

### Word2Vec (2013, Google)
Learn embeddings by predicting word context.

**CBOW** (Continuous Bag of Words): given the surrounding words, predict the center word.
**Skip-gram**: given the center word, predict the surrounding words.

```python
king - man + woman ≈ queen   # famous example showing analogical reasoning in vector space
```

**WHY:** Training word2vec forces words that appear in similar contexts to have similar vectors. "Doctor" and "nurse" often appear in the same contexts (hospitals, patients, medicine), so they end up close in vector space.

### GloVe (2014, Stanford)
Learn embeddings from global co-occurrence statistics — how often do word pairs appear together across the whole corpus? More principled than word2vec.

### FastText (2016, Facebook)
Embeddings for subword units — "eating" is made from the subword pieces "eat" + "ing." This handles out-of-vocabulary words: even if "unhappiness" was never in training, its subwords were.

### ELMo (2018, AllenNLP)
**Contextualized** embeddings — the same word gets a different embedding depending on the sentence it appears in.
```
"bank" in "river bank"    ≠ "bank" in "bank account"
```
Uses a bidirectional LSTM to read the sentence in both directions before producing the embedding.

### Limitation of Static Embeddings
**What it is:** Word2Vec and GloVe give the same single vector for every occurrence of a word, regardless of context. ELMo fixed this with context-sensitive vectors. Transformers (BERT, GPT) took this much further with dynamic, contextual embeddings computed through attention.

---

## 5. Sentence Embeddings

**What it is:** A sentence embedding is a single vector representing the meaning of an entire sentence (or paragraph). Unlike word embeddings (one vector per word), sentence embeddings collapse the whole text into one point in vector space.

**Use cases:**
- Semantic search: find documents that mean the same thing as the query, even with different words
- Sentence similarity: measure how alike two sentences are
- Clustering: group documents by topic
- RAG retrieval: find the most relevant chunks for a user question

### Models
```python
from sentence_transformers import SentenceTransformer  # specialized library for sentence embeddings

model = SentenceTransformer('all-MiniLM-L6-v2')  # load a small, fast sentence embedding model

# encode() takes a list of strings and returns a 2D array (one row per sentence)
embeddings = model.encode(["Hello world", "Hi there"])
# result: shape (2, 384) — two vectors of 384 dimensions each
```

### Bi-encoder vs Cross-encoder

**What it is:** Two different architectures for comparing text pairs. Bi-encoder embeds each text independently (fast). Cross-encoder reads both texts together (slow but more accurate).

**Analogy:**
- Bi-encoder: like sorting mail by city using zip codes — fast pre-processing, no need to open each envelope
- Cross-encoder: like reading each letter carefully to understand its meaning — slow but accurate

| Type | Speed | Accuracy | Use |
|------|-------|---------|-----|
| Bi-encoder | Fast (pre-compute) | Moderate | RAG retrieval (top-K) |
| Cross-encoder | Slow (pair-wise) | High | Re-ranking top-K |

**WHY:** In a RAG system with millions of documents, you cannot run a cross-encoder against every document for every query — it would be too slow. Instead, use the bi-encoder to quickly narrow to the top 50 candidates, then use the cross-encoder to carefully re-rank those 50. Two-stage pipeline: fast + accurate.

---

## 6. Sequence-to-Sequence Tasks

**What it is:** Any task that takes one sequence (text) as input and produces another sequence (text) as output. These are the core use cases for encoder-decoder models like T5 and BART.

All tasks that transform one sequence to another:

| Task | Input | Output |
|------|-------|--------|
| Translation | English text | French text |
| Summarization | Article | Summary |
| Question Answering | Question + Context | Answer |
| Text-to-SQL | Natural language | SQL query |
| Dialogue | Conversation history | Next response |

---

## 7. Text Preprocessing Techniques

### Tokenization
Split text into tokens (words, subwords, characters). The foundation of all NLP — covered in detail in the tokenization file.

### Stopword Removal
**What it is:** Removing extremely common words (the, is, at, which) that carry little meaning on their own. Less relevant now with LLMs but still used in BM25 keyword search to improve signal quality.

### Lemmatization vs Stemming
**What it is:** Both try to reduce words to their root form. Stemming is a fast, crude rule-based approach. Lemmatization is a linguistically correct approach that uses a dictionary.

| Method | "running" | "better" | Speed |
|--------|----------|---------|-------|
| Stemming | "run" (rough) | "better" (wrong!) | Fast |
| Lemmatization | "run" (accurate) | "good" (correct!) | Slower |

**WHY:** Stemming just chops off suffixes with rules ("running" → "run" by removing "ning"). Lemmatization actually looks up the word in a dictionary and finds the proper base form. For "better" → "good," only lemmatization gets this right because "better" and "good" share no morphological relationship.

### BPE Tokenization (modern)
```
"tokenization" → ["token", "ization"]
"unhappiness"  → ["un", "happiness"]
```

---

## 8. Named Entity Recognition (NER)

**What it is:** NER is the task of identifying named things in text and classifying them by type — people, organizations, locations, dates, and more. It is one of the most common NLP tasks in production systems.

**Analogy:** Like a yellow-highlighter pass through a document where you highlight every person in yellow, every company in blue, every location in green. NER automates this highlighting.

Tag each token with its entity type:
```
"Apple Inc. was founded by Steve Jobs in Cupertino."
 [ORG]          [PER: Steve Jobs]    [LOC: Cupertino]
```

Common tags: PERSON, ORG, LOC, DATE, MONEY, PRODUCT

Models: spaCy (fast, production-ready), BERT + token classification head (high accuracy).

---

## 9. Semantic Similarity

**What it is:** Measuring how similar two pieces of text are in *meaning*, not just in *words*. Two sentences can be semantically identical even if they use completely different words.

```python
from sentence_transformers import SentenceTransformer, util  # import sentence embedding tools

model = SentenceTransformer('all-MiniLM-L6-v2')  # load the sentence embedding model

# encode each sentence into a dense vector
e1 = model.encode("How do I reset my password?")
e2 = model.encode("I forgot my password and need to change it.")

# cosine similarity: 1.0 = identical meaning, 0.0 = unrelated, -1.0 = opposite
similarity = util.cos_sim(e1, e2)  # → 0.89 (very similar despite different words)
```

**WHY:** Cosine similarity measures the angle between two vectors. If two sentences mean the same thing, their embedding vectors point in the same direction (small angle, high cosine). Search engines use this to find documents that mean what the user asked, even when the exact words don't match.

---

## Matryoshka Representation Learning (MRL)

**What it is:** A training technique that produces embeddings where any prefix of dimensions is independently useful. You can use just the first 64 dimensions for a fast approximate search, or all 1536 dimensions for high-accuracy reranking — all from the same model.

**Analogy:** Traditional embeddings are like a JPEG where every pixel is equally necessary — you need the whole image or nothing. MRL embeddings are like a streaming video that gets progressively sharper — even the first few seconds give you a usable, blurry image. The more dimensions you use, the sharper the meaning gets.

Traditional embeddings: fixed size (e.g. 1536 dims) — you always use all dimensions.
MRL: train ONE model that produces embeddings where the FIRST N dimensions are already useful.

How it works:
- Train with loss computed at multiple truncation points: [32, 64, 128, 256, 512, 1024, 1536]
- First 32 dims capture most important info, next 32 add more detail, etc.
- Like Russian nesting dolls — a smaller embedding is inside every larger one

**WHY:** Computing loss at multiple truncation points forces early dimensions to be maximally informative on their own — the model cannot defer meaning to later dimensions. It must encode the most important information first.

Why it matters in production:
- Use 256-dim embeddings for fast ANN search (lower storage, faster query)
- Use full 1536-dim for final reranking (higher accuracy)
- OpenAI text-embedding-3 uses MRL — that is why you can set the `dimensions` parameter
- Nomic Embed also uses MRL

**Interview answer:** "What is Matryoshka embedding?" → "A training technique where the model learns embeddings where any prefix of dimensions is independently useful. Enables adaptive precision — use small dims for fast search, large dims for accurate reranking — all from one model."

---

## MTEB — Massive Text Embedding Benchmark

**What it is:** MTEB is the standard leaderboard for comparing embedding models. It runs every model on 56 different tasks and reports scores. It is the embedding model equivalent of MMLU for LLMs.

The standard way to evaluate and compare embedding models.

56 tasks across 8 categories:
- **Retrieval** (most important for RAG): nDCG@10 on MS-MARCO, BEIR
- **Semantic Textual Similarity (STS):** Spearman correlation
- **Classification:** accuracy
- **Clustering:** V-measure
- Reranking, Summarization, Pair Classification, Bitext Mining

**WHY:** A single MTEB overall score is misleading — a model that ranks #1 overall may underperform on Retrieval specifically, which is the only category that matters for RAG. Always filter by the Retrieval category when choosing a model for RAG.

Top models (2024-2025):
| Model | MTEB Score | Dims | Size |
|-------|-----------|------|------|
| text-embedding-3-large | 64.6 | 3072 | API |
| voyage-large-2 | 63.6 | 1536 | API |
| BAAI/bge-large-en-v1.5 | 63.5 | 1024 | 335M |
| Nomic-embed-text-v1.5 | 62.3 | 768 | 137M |
| all-MiniLM-L6-v2 | 56.3 | 384 | 22M (fast, small) |

**Interview answer:** "How do you choose an embedding model for RAG?" → "Check MTEB leaderboard specifically on the Retrieval category for your domain. For production, balance MTEB score vs model size vs inference cost. BAAI/bge-large is the best open-source option. For multilingual, use multilingual-e5-large."

---

## ColBERT — Late Interaction Retrieval

**What it is:** ColBERT is a retrieval architecture that keeps ALL token-level embeddings for documents (instead of compressing to a single vector) and computes a richer similarity score at query time using a "MaxSim" operation.

**Analogy:** A bi-encoder is like rating a restaurant with one star (1–5). ColBERT is like rating every dish separately. The "overall score" from ColBERT (MaxSim) captures nuances that a single-star rating misses.

Standard bi-encoder: embed query → embed doc → single cosine similarity score.
Problem: compresses full document meaning into one vector, losing fine-grained token-level matching.

ColBERT: keep ALL token embeddings, score at retrieval time:
- Query: [q1, q2, q3, q4] → 4 token embeddings (128-dim each)
- Document: [d1, d2, d3, d4, d5] → 5 token embeddings

Score = Σ over query tokens: max(sim(qi, dj)) for all doc tokens dj
= **MaxSim operation:** each query token independently finds its best matching document token

**WHY:** MaxSim is powerful because it doesn't force the entire document meaning into one vector. Each query token independently finds its best evidence in the document. "apple" in "Apple Inc" vs "apple fruit" — a bi-encoder collapses this into one vector, but ColBERT handles it because the token-level embeddings capture the context differently.

Why it is not always used:
- Storage cost: store ALL token embeddings (128-dim × avg_doc_length × num_docs) = huge
- 1M docs × 100 tokens × 128 dims × 4 bytes = 51GB just for the index

Production use: RAGatouille library makes ColBERT easy to use.

**Interview answer:** "What is late interaction retrieval?" → "ColBERT stores token-level embeddings for documents and computes MaxSim at query time — each query token finds its best matching document token. More accurate than single-vector retrieval but requires much more storage."

---

## 10. Interview Questions — NLP Techniques

**Q: What is transfer learning in NLP?**
> Pre-training a model on massive text data to learn general language representations, then fine-tuning on a smaller task-specific dataset. The pre-trained model transfers its language knowledge to the new task, drastically reducing the data and compute needed for good performance.

**Q: What is in-context learning?**
> The ability of large language models to learn new tasks from examples provided directly in the prompt, without any weight updates. GPT-3 demonstrated this — it can perform translation, summarization, or code generation from 0–10 prompt examples.

**Q: What is the difference between bi-encoder and cross-encoder?**
> A bi-encoder separately embeds both the query and document, then computes similarity — fast because documents can be pre-encoded. A cross-encoder jointly processes the query-document pair for a relevance score — more accurate but requires running inference for every pair. In RAG: use bi-encoder for retrieval, cross-encoder for re-ranking.

**Q: What are contextual embeddings and why are they better than Word2Vec?**
> Contextual embeddings (BERT, ELMo) generate different vectors for the same word depending on context — "bank" gets a different embedding in "river bank" vs "bank account." Word2Vec gives the same static vector regardless of context.

**Q: What is Chain-of-Thought prompting and when does it help?**
> Prompting the model to reason step-by-step before giving an answer. It significantly improves performance on multi-step reasoning, math problems, and complex instructions. It's most effective for larger models (>= 7B) — smaller models don't benefit as much.

**Q: What is the attention mechanism at a high level?**
> Attention computes a weighted sum of value vectors, where weights are based on the similarity between a query vector and key vectors. This allows each token to selectively focus on the most relevant other tokens when building its representation.

---

## Quick Reference Cheat Sheet

```
Transfer Learning:    Pre-train (huge data) → Fine-tune (small data)
Few-Shot:             Learn from examples in the prompt (no weight update)
Zero-Shot:            Task description only, no examples
Chain-of-Thought:     "Think step by step" → better reasoning
Contextual Embeddings: Same word, different vector based on context (BERT)
Bi-encoder:           Fast retrieval (pre-computed)
Cross-encoder:        Accurate re-ranking (pair-wise)
Word2Vec:             Static embeddings (pre-2018 NLP)
MRL:                  Truncatable embeddings — any prefix works independently
ColBERT:              Token-level late interaction — MaxSim scoring
MTEB:                 Embedding model benchmark — filter by Retrieval category for RAG
```
