# 06 — Generative AI: Text Generation, Sampling & Hallucination

> Understand what Generative AI is, how text generation works, and the key NLP tasks you'll work on.

---

## 1. What is Generative AI?

**What it is:** Generative AI is a type of AI that *creates* new content — text, images, code, audio — by learning patterns from massive training data. Think of it like a student who reads millions of books and then writes their own.

**Generative AI** creates new content (text, images, code, audio) by learning patterns from training data.

### Generative vs Discriminative Models

**What it is:** These are two fundamentally different approaches to learning from data.
- A **discriminative** model is like a judge: it looks at something and decides what category it belongs to.
- A **generative** model is like an author: it learns *how things are written* so it can write new things itself.

| Type | What it does | Examples |
|------|-------------|---------|
| **Discriminative** | Maps input to label (classifies) | BERT, SVM, Logistic Regression |
| **Generative** | Models the data distribution, can generate new samples | GPT, DALL-E, Stable Diffusion |

### How Generative LLMs Work

**What it is:** LLMs generate text one word-piece (token) at a time, always looking at what came before to predict what comes next — like autocomplete on steroids.

LLMs are **autoregressive** — they generate one token at a time:

```
"The cat sat on the" → predict → "mat"
"The cat sat on the mat" → predict → "."
"The cat sat on the mat." → predict → [EOS]
```

**Analogy:** It is like a game of "what word comes next?" — the model plays this game billions of times during training until it gets very good at predicting natural language.

At each step: output a probability distribution over the vocabulary, sample a token.

---

## 2. Text Generation — How it Works

### Autoregressive Generation

**What it is:** The model takes a running context and keeps predicting the next token until it decides to stop (generates an End-Of-Sequence token).

```python
# Pseudocode showing the generation loop
context = tokenize("The cat sat on the")  # convert text to token IDs
while not done:                            # keep generating until EOS or max length
    logits = model(context)               # run model forward pass → raw scores for every vocab word
    next_token = sample(logits)           # pick the next token (various strategies explained below)
    context = context + [next_token]      # append chosen token to the running context
    if next_token == EOS: break           # stop if model generated end-of-sequence token
```

**WHY:** The loop runs because the model only produces one token at a time. Each new token becomes part of the context for the next prediction, so every token influences what comes after it.

### The Vocabulary Distribution

**What it is:** At every step the model outputs a score (called a "logit") for every word in the vocabulary. Those scores are turned into probabilities with softmax. The model then picks (samples) one token.

**Analogy:** Imagine a roulette wheel with 50,000 slots — one per vocabulary word. The model assigns a weight to each slot. Bigger weight = bigger slice of the wheel = more likely to be picked.

At each step, the model outputs a logit for each vocabulary token.
Logits → Softmax → Probability distribution → Sample

```
vocab: ["the", "a", "mat", "floor", "chair", ...]
probs: [0.02, 0.01, 0.45, 0.23, 0.12, ...]
sample: "mat"
```

---

## 3. Sampling Strategies (Decoding Methods)

**What it is:** Once we have the probability distribution, we need to decide *how* to pick the next token. This choice dramatically affects whether output is creative or boring, coherent or random.

**Analogy:** Think of the distribution like a menu. "How you order" is the sampling strategy. You can always order the most popular dish (greedy), explore a few top choices (beam search), or randomly try anything with some dishes weighted higher (temperature sampling).

How we choose the next token from the probability distribution matters a lot for output quality.

### Greedy Decoding

**What it is:** The simplest possible strategy — always pick the word with the highest probability. No randomness at all.

Always pick the most probable token.
```
next_token = argmax(logits)  # argmax means "index of the maximum value"
```
- Fast
- Can get stuck in repetitive loops
- Deterministic (same input always gives same output)

**WHY this fails:** If the model is 40% confident "the" is next and 38% confident "a" is next, greedy always picks "the" — but sometimes "a" would lead to a better overall sentence. Greedy is short-sighted.

### Beam Search

**What it is:** Instead of committing to one token at a time, keep track of the K most promising partial sequences simultaneously. At the end, pick the best complete sequence.

**Analogy:** You are writing a story. Instead of committing to your first word choice, you try 3 different first words, then 3 different second words for each of those, and so on. At the end you pick the storyline that scored best overall.

Keep top-K (beam width) sequences at each step, pick the best complete sequence.
```
beam_width=3: track top 3 sequences at each step
```
- Better quality than greedy
- Deterministic, structured
- Used in machine translation, summarization
- Too "safe" for open-ended generation

**WHY:** Beam search finds high-probability complete sequences rather than high-probability single tokens. It is the standard for translation and summarization where correctness matters more than creativity.

### Temperature Sampling

**What it is:** A knob that controls how "spiky" or "flat" the probability distribution is before sampling. Low temperature = model sticks to its favorites. High temperature = model considers many options more equally.

**Analogy:** Temperature is like how brave a chef is. Low temperature = always makes the classic dish. High temperature = experiments wildly with unusual combinations.

Scale logits before softmax to control randomness.
```
scaled_logits = logits / temperature   # divide every logit by temperature value
probs = softmax(scaled_logits)         # convert to probabilities
next_token = sample(probs)             # randomly pick one token weighted by these probs
```
| Temperature | Effect |
|-------------|--------|
| < 1.0 (e.g., 0.3) | More focused, less random, repetitive |
| = 1.0 | Unchanged distribution |
| > 1.0 (e.g., 1.5) | More random, creative, diverse |

**WHY:** Dividing by a small number (< 1) makes large logits relatively larger and small ones relatively smaller, sharpening the distribution. Dividing by a large number (> 1) flattens it.

### Top-K Sampling

**What it is:** Before sampling, throw away all tokens except the top K most probable ones. Only sample from those K candidates.

**Analogy:** You are at an ice cream shop with 200 flavors. Top-K with K=10 means you only consider the 10 most popular flavors — no sampling from exotic flavors with 0.001% probability.

Only sample from the top K most probable tokens (zero out the rest).
```python
top_k = 50  # only consider 50 most probable tokens — zero out the rest
```
Prevents very low probability (nonsense) tokens being selected.

**WHY:** Without top-K, there is always a tiny chance the model picks a completely random, incoherent word. Top-K eliminates that tail risk while preserving some diversity.

### Top-P (Nucleus) Sampling

**What it is:** Instead of a fixed number of candidates (like top-K), dynamically pick the *smallest* set of tokens whose probabilities add up to at least P. If the model is very confident, this set is small; if the model is uncertain, this set is large.

**Analogy:** Top-P is like saying "order from restaurants that together cover 90% of all customer demand" — some days that is 2 restaurants, other days it is 15, depending on how popular each one is.

Sample from the smallest set of tokens whose cumulative probability exceeds p.
```
top_p = 0.9  # top tokens summing to 90% probability
```
- Dynamically adjusts the number of candidates
- More natural than top-K
- Most commonly used in practice

**WHY:** Top-K has a flaw — when the model is very confident, the top-50 still includes some bad options. When uncertain, top-50 misses many good options. Top-P adapts to the actual confidence level.

### Typical Production Settings

**What it is:** Common combinations of these settings tuned for different use cases.

```
temperature=0.7, top_p=0.9, top_k=50  # balanced creativity + coherence (general chat)
temperature=0.2, top_p=0.95           # factual, deterministic answers (customer support)
temperature=1.0, top_p=0.8            # creative writing (stories, marketing copy)
```

**WHY:** There is no universal best setting. Low temperature for facts (you want the right answer), high temperature for creativity (you want surprising output).

---

## 4. Key NLP Tasks

**What it is:** The specific types of problems LLMs are applied to in production. Each has slightly different prompt formats and evaluation methods.

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

**WHY this matters in production:** Text completion powers autocomplete in code editors (GitHub Copilot) and document editors.

### Summarization

**What it is:** Condense long text into a shorter version that preserves the key information.

Condense long text into a shorter version.
```
Input:  [500-word article]
Output: [2-sentence summary]
```

**Extractive**: Extract key sentences from original text. (No new words invented — like highlighting.)
**Abstractive**: Generate new sentences that capture the meaning. (LLMs do this — like writing a summary in your own words.)

### Question Answering
Answer a question, optionally given a context passage.
```
Context: "The Eiffel Tower was built in 1889..."
Question: "When was the Eiffel Tower built?"
Answer: "1889"
```

**Open-domain QA**: No context — model answers from its training knowledge alone. Risky for hallucinations.
**Closed-domain QA**: Given context (RAG is used here) — model answers based only on provided text. Much safer.

### Machine Translation
Convert text from one language to another.
```
EN: "Hello, how are you?"
FR: "Bonjour, comment allez-vous?"
```

### Text Classification

**What it is:** Assign a label from a fixed set to a piece of text. The most classic NLP task.

Assign a label to a piece of text.
```
"I love this product!" → positive
"Terrible experience" → negative
```

### Named Entity Recognition (NER)

**What it is:** Find and tag real-world entities (people, places, organizations, dates) in text.

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

**What it is:** The art and science of designing the input text (prompt) to get the best possible output from an LLM. Because LLMs are sensitive to exact wording, small changes can dramatically affect output quality.

**Analogy:** Prompt engineering is like knowing how to ask a very smart but literal genie for wishes. The genie will do exactly what you ask — so you have to ask precisely.

The art of crafting inputs to get better outputs from LLMs.

### Zero-Shot Prompting

**What it is:** Ask the model to do a task without giving any examples. Just a direct instruction.

```
"Classify the sentiment of this tweet: 'This is amazing!'"
```

**WHY:** Works for simple tasks where the model already understands what to do from its training.

### Few-Shot Prompting

**What it is:** Give 2–5 examples of input→output pairs before the actual question. Teaches the model the exact format and style you want.

```
"positive: 'Great product!'
negative: 'Terrible experience'
Classify: 'Best purchase ever!'"
```

**WHY:** Examples dramatically help the model understand your desired output format and reasoning style, especially for unusual tasks.

### Chain-of-Thought (CoT)

**What it is:** Instead of asking the model to jump straight to an answer, prompt it to show its work step-by-step. This improves accuracy on complex reasoning tasks.

**Analogy:** Instead of asking a student "what is 15% of 340?" and hoping they get it right in their head, you ask them to show every calculation step. They are much less likely to make mistakes.

Add "Let's think step by step" or provide reasoning examples.
```
"Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls each.
How many does he have now? Let's think step by step."
```
Significantly improves reasoning on complex tasks.

**WHY:** Generating intermediate reasoning steps forces the model to work through logic sequentially rather than guessing. Research shows this helps on math, logic, and multi-step reasoning — especially for models 7B+ parameters.

### System Prompts

**What it is:** A special prompt given at the start of a conversation that defines the model's role, rules, and behaviour. Users cannot see it but it shapes every response.

```
System: "You are an expert SQL engineer. Always respond with valid SQL."
User: "Get all users who signed up in 2024"
```

**WHY:** System prompts let you customize the model's persona and constraints for your specific application without retraining.

### ReAct Prompting

**What it is:** A framework that combines reasoning (thinking out loud) with acting (calling tools). The model alternates between writing thoughts and taking actions until it reaches an answer.

**Analogy:** Watching a detective work — they think "I need to check the alibi", then they go check it, observe what they find, think about what it means, then act again.

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

**What it is:** How do you know if a language model is actually good? There are several automated metrics and human evaluation approaches used in production and research.

### Automatic Metrics

**Perplexity (PPL)**

**What it is:** A measure of how "surprised" the model is by a text. If the model assigns high probability to each actual next word, it has low perplexity. Lower perplexity = better language model.

**Analogy:** Perplexity is like a test score where the model has to predict every word. If it predicts correctly most of the time, it gets a low perplexity score (good). If it keeps being surprised, perplexity is high (bad).

Measures how well the model predicts a text. Lower = better.
```
PPL = exp(-1/N * Σ log P(token_i))
```

**BLEU (Bilingual Evaluation Understudy)**
- Compare n-gram overlap between generated and reference text
- Used for translation, summarization
- **Limitation:** Two sentences can mean the same thing with different words — BLEU would score them low.

**ROUGE**

**What it is:** Similar to BLEU but focused on recall (how much of the reference text appears in the generated text). More suited to summarization.

- Recall-Oriented Understudy for Gisting Evaluation
- Measures overlap of n-grams between summary and reference
- ROUGE-1 (single words), ROUGE-2 (pairs), ROUGE-L (longest common subsequence)

**BERTScore**

**What it is:** Instead of counting exact word matches, use BERT embeddings to compare the *meaning* of sentences. More sophisticated than BLEU/ROUGE.

- Use BERT embeddings to measure semantic similarity
- Better than BLEU/ROUGE for semantic tasks

**WHY BERTScore is better:** "The car is red" and "The automobile is crimson" score near-zero with BLEU (different words) but high with BERTScore (same meaning).

### LLM-as-Judge

**What it is:** Use a stronger LLM (like GPT-4) to automatically evaluate the outputs of a weaker model. More nuanced than metrics but still automated and scalable.

Use a stronger LLM (e.g., GPT-4) to evaluate outputs on a rubric:
```
"Rate this response on a scale of 1-5 for helpfulness, accuracy, and safety."
```
MT-Bench, AlpacaEval use this approach.

### Benchmarks

**What it is:** Standardized test sets used to compare models. Each benchmark tests a specific capability.

| Benchmark | Tests |
|-----------|-------|
| MMLU | 57 academic subjects (knowledge) |
| HumanEval | Code generation |
| GSM8K | Math word problems |
| TruthfulQA | Avoiding hallucinations |
| HellaSwag | Common sense reasoning |

---

## 7. Hallucination

**What it is:** When an LLM generates text that sounds confident and fluent but is factually incorrect or completely made up. The model is "hallucinating" facts.

**Analogy:** Hallucination is like a student who did not study but writes a confident-sounding essay full of plausible-sounding but wrong facts. They do not know they are wrong — they just pattern-matched from what they have seen.

When an LLM generates factually incorrect but confident-sounding text.

### Why it happens
- Trained to generate fluent text, not necessarily true text (fluency ≠ accuracy)
- May extrapolate or fill in gaps in its knowledge by pattern-matching from related information
- No explicit separation between what it knows and what it does not know

### Mitigation Strategies

**What it is:** Techniques to reduce hallucinations in production systems.

- **RAG**: Ground responses in retrieved documents — model must answer based on real sources, not imagination
- **Low temperature**: More deterministic output sticks closer to high-probability (usually correct) answers
- **Self-consistency**: Sample the same question multiple times, take the majority vote answer — outlier hallucinations get voted out
- **Chain-of-thought**: Forces step-by-step reasoning, makes it easier to catch logical errors
- **Calibration**: Train the model to say "I don't know" when uncertain, rather than guessing

**WHY self-consistency helps:** If you ask the same math question 20 times and 17 answers say "51" but 3 say "52", the outliers are probably hallucinations. Majority vote gives you the most reliable answer.

---

## 8. System Design: LLM Application Architecture

**What it is:** The typical end-to-end architecture of a production LLM application. Each layer has a specific job.

```
User Request
     ↓
API Gateway / Rate Limiting         ← controls traffic, prevents abuse
     ↓
Prompt Builder
   - System prompt                  ← defines model's role and rules
   - Retrieved context (RAG)        ← grounding facts from your database
   - Conversation history           ← past turns so model remembers context
   - User query                     ← what the user just asked
     ↓
LLM Inference                       ← the actual model doing generation
     ↓
Output Parser / Guardrails          ← validate output format, filter harmful content
     ↓
Response to User
```

**WHY this layered design:** Each layer handles one concern. Rate limiting protects your budget. The prompt builder assembles the right context. Guardrails prevent unsafe output from reaching users.

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
