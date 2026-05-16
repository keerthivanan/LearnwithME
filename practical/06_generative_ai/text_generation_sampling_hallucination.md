# 06 — Generative AI: Text Generation, Tasks & Concepts

> Understand what Generative AI is, how text generation works, and the key NLP tasks you'll work on.

---

## 1. What is Generative AI?

**Generative AI** creates new content (text, images, code, audio) by learning patterns from training data.

### Generative vs Discriminative Models
| Type | What it does | Examples |
|------|-------------|---------|
| **Discriminative** | Maps input to label (classifies) | BERT, SVM, Logistic Regression |
| **Generative** | Models the data distribution, can generate new samples | GPT, DALL-E, Stable Diffusion |

### How Generative LLMs Work
LLMs are **autoregressive** — they generate one token at a time:

```
"The cat sat on the" → predict → "mat"
"The cat sat on the mat" → predict → "."
"The cat sat on the mat." → predict → [EOS]
```

At each step: output a probability distribution over the vocabulary, sample a token.

---

## 2. Text Generation — How it Works

### Autoregressive Generation
```python
# Pseudocode
context = tokenize("The cat sat on the")
while not done:
    logits = model(context)          # shape: [vocab_size]
    next_token = sample(logits)      # choose next token
    context = context + [next_token]
    if next_token == EOS: break
```

### The Vocabulary Distribution
At each step, the model outputs a logit for each vocabulary token.
Logits → Softmax → Probability distribution → Sample

```
vocab: ["the", "a", "mat", "floor", "chair", ...]
probs: [0.02, 0.01, 0.45, 0.23, 0.12, ...]
sample: "mat"
```

---

## 3. Sampling Strategies (Decoding Methods)

How we choose the next token from the probability distribution matters a lot for output quality.

### Greedy Decoding
Always pick the most probable token.
```
next_token = argmax(logits)
```
- Fast
- Can get stuck in repetitive loops
- Deterministic

### Beam Search
Keep top-K (beam width) sequences at each step, pick the best complete sequence.
```
beam_width=3: track top 3 sequences at each step
```
- Better quality than greedy
- Deterministic, structured
- Used in machine translation, summarization
- Too "safe" for open-ended generation

### Temperature Sampling
Scale logits before softmax to control randomness.
```
scaled_logits = logits / temperature
probs = softmax(scaled_logits)
next_token = sample(probs)
```
| Temperature | Effect |
|-------------|--------|
| < 1.0 (e.g., 0.3) | More focused, less random, repetitive |
| = 1.0 | Unchanged distribution |
| > 1.0 (e.g., 1.5) | More random, creative, diverse |

### Top-K Sampling
Only sample from the top K most probable tokens (zero out the rest).
```python
top_k = 50  # only consider 50 most probable tokens
```
Prevents very low probability (nonsense) tokens being selected.

### Top-P (Nucleus) Sampling
Sample from the smallest set of tokens whose cumulative probability exceeds p.
```
top_p = 0.9  # top tokens summing to 90% probability
```
- Dynamically adjusts the number of candidates
- More natural than top-K
- Most commonly used in practice

### Typical Production Settings
```
temperature=0.7, top_p=0.9, top_k=50  # balanced creativity + coherence
temperature=0.2, top_p=0.95           # factual, deterministic answers
temperature=1.0, top_p=0.8            # creative writing
```

---

## 4. Key NLP Tasks

### Text Generation
Produce free-form text given a prompt.
```
Prompt: "Write a blog post about..."
Output: [Generated blog post]
```
Used in: ChatGPT, Copilot, creative writing tools.

### Text Completion
Complete a partial piece of text.
```
Input:  "The capital of France is"
Output: " Paris."
```

### Summarization
Condense long text into a shorter version.
```
Input:  [500-word article]
Output: [2-sentence summary]
```

**Extractive**: Extract key sentences from original text.
**Abstractive**: Generate new sentences that capture the meaning (LLMs do this).

### Question Answering
Answer a question, optionally given a context passage.
```
Context: "The Eiffel Tower was built in 1889..."
Question: "When was the Eiffel Tower built?"
Answer: "1889"
```

**Open-domain QA**: No context, model answers from training knowledge.
**Closed-domain QA**: Given context (RAG is used here).

### Machine Translation
Convert text from one language to another.
```
EN: "Hello, how are you?"
FR: "Bonjour, comment allez-vous?"
```

### Text Classification
Assign a label to a piece of text.
```
"I love this product!" → positive
"Terrible experience" → negative
```

### Named Entity Recognition (NER)
Tag entities in text.
```
"Elon Musk founded Tesla in 2003."
  [PER]          [ORG]   [DATE]
```

### Code Generation
Generate code from natural language.
```
"Write a Python function to sort a list of dictionaries by value"
```

---

## 5. Prompt Engineering

The art of crafting inputs to get better outputs from LLMs.

### Zero-Shot Prompting
```
"Classify the sentiment of this tweet: 'This is amazing!'"
```

### Few-Shot Prompting
```
"positive: 'Great product!'
negative: 'Terrible experience'
Classify: 'Best purchase ever!'"
```

### Chain-of-Thought (CoT)
Add "Let's think step by step" or provide reasoning examples.
```
"Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls each.
How many does he have now? Let's think step by step."
```
Significantly improves reasoning on complex tasks.

### System Prompts
```
System: "You are an expert SQL engineer. Always respond with valid SQL."
User: "Get all users who signed up in 2024"
```

### ReAct Prompting
Combine reasoning (thought) + action (tool use) + observation:
```
Thought: I need to search for this.
Action: Search[Paris population]
Observation: 2.1 million
Thought: I have the answer.
Answer: Paris has 2.1 million people.
```

---

## 6. LLM Evaluation

### Automatic Metrics

**Perplexity (PPL)**
Measures how well the model predicts a text. Lower = better.
```
PPL = exp(-1/N * Σ log P(token_i))
```

**BLEU (Bilingual Evaluation Understudy)**
- Compare n-gram overlap between generated and reference text
- Used for translation, summarization

**ROUGE**
- Recall-Oriented Understudy for Gisting Evaluation
- Measures overlap of n-grams between summary and reference
- ROUGE-1, ROUGE-2, ROUGE-L

**BERTScore**
- Use BERT embeddings to measure semantic similarity
- Better than BLEU/ROUGE for semantic tasks

### LLM-as-Judge
Use a stronger LLM (e.g., GPT-4) to evaluate outputs on a rubric:
```
"Rate this response on a scale of 1-5 for helpfulness, accuracy, and safety."
```
MT-Bench, AlpacaEval use this approach.

### Benchmarks
| Benchmark | Tests |
|-----------|-------|
| MMLU | 57 academic subjects (knowledge) |
| HumanEval | Code generation |
| GSM8K | Math word problems |
| TruthfulQA | Avoiding hallucinations |
| HellaSwag | Common sense reasoning |

---

## 7. Hallucination

When an LLM generates factually incorrect but confident-sounding text.

### Why it happens
- Trained to generate fluent text, not necessarily true text
- May extrapolate/fill gaps in knowledge
- No explicit separation between known and unknown

### Mitigation Strategies
- **RAG**: Ground responses in retrieved documents
- **Low temperature**: More deterministic, less creative
- **Self-consistency**: Sample multiple outputs, take majority vote
- **Chain-of-thought**: Forces step-by-step reasoning
- **Calibration**: Train model to say "I don't know"

---

## 8. System Design: LLM Application Architecture

```
User Request
     ↓
API Gateway / Rate Limiting
     ↓
Prompt Builder
   - System prompt
   - Retrieved context (RAG)
   - Conversation history
   - User query
     ↓
LLM Inference
     ↓
Output Parser / Guardrails
     ↓
Response to User
```

---

## 9. Interview Questions — Generative AI

**Q: What is the difference between discriminative and generative models?**
> Discriminative models learn a decision boundary to classify inputs (e.g., "is this spam?"). Generative models learn the data distribution and can sample new examples from it (e.g., "generate a spam email"). LLMs are generative — they model P(text) and sample from it.

**Q: How does temperature affect text generation?**
> Temperature scales the logits before softmax. Temperature < 1 sharpens the distribution (more deterministic, repetitive). Temperature > 1 flattens it (more random, diverse, creative). Temperature = 1 leaves the distribution unchanged.

**Q: What is the difference between top-K and top-P sampling?**
> Top-K always considers the K most probable tokens regardless of their probability spread. Top-P (nucleus sampling) dynamically selects the smallest set of tokens whose cumulative probability exceeds P — this adapts to the actual distribution shape and tends to produce more natural results.

**Q: What is Chain-of-Thought prompting?**
> A technique where the LLM is prompted (or fine-tuned) to generate intermediate reasoning steps before the final answer. It significantly improves performance on reasoning, math, and multi-step tasks, especially for models >= 7B parameters.

**Q: What is perplexity and what does it measure?**
> Perplexity measures how surprised a language model is by a given text — it's the exponentiated average negative log-likelihood of the tokens. Lower perplexity means the model assigns higher probability to the actual next tokens, indicating better language modeling.

**Q: What are hallucinations in LLMs and how do you address them?**
> Hallucinations are confident but factually incorrect statements. They happen because LLMs are trained for fluency, not factual accuracy. Mitigation: RAG (ground in retrieved documents), low temperature, self-consistency sampling, and training with factual preference data.

---

## Quick Reference Cheat Sheet

```
Autoregressive:    Generate one token at a time, left to right
Temperature:       < 1 = focused, > 1 = creative, 1 = unchanged
Top-P:             Sample from top tokens summing to P probability
CoT:               "Let's think step by step" → better reasoning
Hallucination:     Fluent but wrong — fix with RAG or grounding
Evaluation:        BLEU/ROUGE (overlap), BERTScore (semantic), PPL (fluency)
Perplexity:        Lower = model better predicts the text
```
