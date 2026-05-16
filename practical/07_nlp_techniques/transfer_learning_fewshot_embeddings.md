# 07 — NLP Techniques: Transfer Learning, Few-Shot & More

> Key techniques from the JD: attention mechanisms, transfer learning, few-shot learning.

---

## 1. Transfer Learning

### What it is
Transfer learning takes knowledge learned from one task/domain and applies it to another task/domain.

### The Pre-train → Fine-tune Paradigm

```
Stage 1: Pre-train on huge general corpus
         → Model learns: grammar, facts, reasoning, language patterns
         → Cost: $$$$ (millions of dollars, weeks of compute)

Stage 2: Fine-tune on small task-specific data
         → Adapt to specific task
         → Cost: $ (hours on a few GPUs)
```

### Why Transfer Learning Works
- Lower layers learn general features (syntax, word meaning)
- Higher layers learn task-specific features
- Fine-tuning updates higher layers while preserving lower-layer knowledge
- Less data needed because foundation is already learned

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
The model learns a new task from very few examples (sometimes zero).

### Zero-Shot
Model performs task without any examples, just a description.
```
"Classify if this review is positive or negative: 'I hated this product.'"
→ "negative"
```

### One-Shot
Model performs task with one example.
```
"Positive: 'Love it!' → positive
Classify: 'Worst purchase ever' →"
```

### Few-Shot (in-context learning)
Provide 3–10 examples in the prompt. No gradient update.
```
"Translate English to French:
sea otter → loutre de mer
peppermint → menthe poivrée
plush girafe → girafe en peluche
cheese →"
```

### Why LLMs can do Few-Shot
GPT-3 (2020) showed that very large language models can learn from examples in context **without weight updates**. This is called **in-context learning (ICL)**.

The mechanism: the attention mechanism uses the examples to condition the generation.

### Few-Shot vs Fine-Tuning Comparison
| Aspect | Few-Shot (ICL) | Fine-Tuning |
|--------|---------------|------------|
| Weight update? | No | Yes |
| Data needed | 0–10 examples in prompt | 100s–1000s |
| Cost | Just inference | Training cost |
| Flexibility | Change examples per call | Retrain to change |
| Performance | Good for large models | Better for small models |
| Persistent? | Only in context | Baked into weights |

---

## 3. Attention Mechanisms (Advanced)

### Why Attention is the Key Innovation

Before attention, seq2seq models compressed the entire input into a fixed-size vector — information bottleneck for long sequences.

Attention allows the decoder to "look back" at all encoder positions:
```
decoder_output = Σ attention_weight_i * encoder_output_i
```

### Bahdanau (Additive) Attention — Pre-Transformer
```
e_ij = v * tanh(W_a * h_i + W_b * s_j)    # alignment score
a_ij = softmax(e_ij)                        # attention weights
c_j  = Σ a_ij * h_i                        # context vector
```
- First attention mechanism (2015)
- Used in neural machine translation
- Foundation for Transformer attention

### Self-Attention vs Cross-Attention
| Type | Q from | K,V from | Used in |
|------|--------|---------|---------|
| Self-attention | Same sequence | Same sequence | BERT encoder, GPT decoder |
| Cross-attention | Decoder | Encoder output | T5, BART |

### Local vs Global Attention
| Type | Attends to | Use Case |
|------|-----------|---------|
| Global (full) | All tokens | Standard Transformers |
| Local/Sliding Window | Nearby N tokens | Long documents (Longformer) |
| Sparse | Selected positions | BigBird |
| Linear | Approximated globally | Efficient Transformers |

---

## 4. Word Embeddings (Pre-Transformer NLP)

### Word2Vec (2013, Google)
Learn embeddings by predicting word context.

**CBOW** (Continuous Bag of Words): predict center word from context.
**Skip-gram**: predict context words from center word.

```python
king - man + woman ≈ queen   # famous example
```

### GloVe (2014, Stanford)
Learn embeddings from global co-occurrence statistics.

### FastText (2016, Facebook)
Embeddings for subword units — handles out-of-vocabulary words.

### ELMo (2018, AllenNLP)
**Contextualized** embeddings — same word gets different embedding based on context.
```
"bank" in "river bank" ≠ "bank" in "bank account"
```
Uses bidirectional LSTM.

### Limitation of Static Embeddings
Word2Vec/GloVe give the same vector regardless of context.
Transformers (BERT, GPT) gave rise to **dynamic, contextual embeddings**.

---

## 5. Sentence Embeddings

Single vector representing meaning of entire sentence.

### Use Cases
- Semantic search
- Sentence similarity
- Clustering documents
- RAG retrieval

### Models
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(["Hello world", "Hi there"])
```

### Bi-encoder vs Cross-encoder
| Type | Speed | Accuracy | Use |
|------|-------|---------|-----|
| Bi-encoder | Fast (pre-compute) | Moderate | RAG retrieval (top-K) |
| Cross-encoder | Slow (pair-wise) | High | Re-ranking top-K |

---

## 6. Sequence-to-Sequence Tasks

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
Split text into tokens (words, subwords, characters).

### Stopword Removal
Remove common words (the, is, at) — less relevant now with LLMs but used in BM25.

### Lemmatization vs Stemming
| Method | "running" | "better" |
|--------|----------|---------|
| Stemming | "run" (rough) | "better" |
| Lemmatization | "run" (accurate) | "good" |

Lemmatization is linguistically correct; stemming is faster but crude.

### BPE Tokenization (modern)
```
"tokenization" → ["token", "ization"]
"unhappiness"  → ["un", "happiness"]
```

---

## 8. Named Entity Recognition (NER)

Tag each token with its entity type:
```
"Apple Inc. was founded by Steve Jobs in Cupertino."
 [ORG]          [PER: Steve Jobs]    [LOC: Cupertino]
```

Common tags: PERSON, ORG, LOC, DATE, MONEY, PRODUCT

Models: spaCy, BERT + token classification head.

---

## 9. Semantic Similarity

Measure how similar two pieces of text are in meaning.

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')
e1 = model.encode("How do I reset my password?")
e2 = model.encode("I forgot my password and need to change it.")

similarity = util.cos_sim(e1, e2)  # → 0.89 (very similar)
```

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
```
