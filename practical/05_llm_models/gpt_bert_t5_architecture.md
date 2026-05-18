# 03 — LLM Models: GPT, BERT, T5 & Modern LLMs

> Know the architecture, training objective, and use cases of each model family.

---

## 1. The Three Paradigms

**What it is:** The three fundamental ways to build a language model, each reflecting a different philosophy about what the model should do and what kind of data it should learn from.

| Model | Architecture | Training Objective | Best For |
|-------|-------------|-------------------|---------|
| **BERT** | Encoder-only | Masked Language Modeling | Understanding (classification, NER, QA) |
| **GPT** | Decoder-only | Causal Language Modeling | Generation (text, code, chat) |
| **T5** | Encoder-Decoder | Text-to-Text (masked spans) | Seq2seq (translation, summarization) |

**Analogy:**
- BERT is like a book editor who reads the whole document before marking anything
- GPT is like a novelist writing one word at a time, never reading ahead
- T5 is like a translator who reads the source document fully, then writes the translation

---

## 2. BERT — Bidirectional Encoder Representations from Transformers

**What it is:** Google's 2018 model that proved you can pre-train a Transformer on unlabeled text using bidirectional context, then fine-tune it on almost any NLP task with excellent results.

### Released
Google, 2018

### Architecture
- **Encoder-only** Transformer — no generation capability, only understanding
- **Bidirectional**: each token attends to ALL other tokens (left AND right context simultaneously)
- 12 layers (base) / 24 layers (large)
- d_model: 768 (base) / 1024 (large)
- Parameters: 110M (base) / 340M (large)

**WHY encoder-only?** BERT's goal is understanding, not generation. To understand a word like "bank" in context, you need to see everything around it — words before AND after. An encoder with bidirectional attention provides exactly this full-context view.

### Training Objectives

**1. Masked Language Modeling (MLM)**

**What it is:** A fill-in-the-blank task — mask some words, train the model to predict them from surrounding context.

- Randomly mask 15% of tokens with [MASK]
- Model predicts the original masked token
- Teaches the model to understand context

```
Input:  "The cat [MASK] on the mat"
Target: "sat"
← The model sees all other words bidirectionally
← It must figure out that "sat" fits here based on all context
```

**WHY masking (not next-token prediction)?** Next-token prediction forces left-to-right processing. Masking allows bidirectional attention — the model sees both "The cat" AND "on the mat" to predict [MASK]. This gives much richer contextual representations for understanding tasks.

**2. Next Sentence Prediction (NSP)**

**What it is:** A binary classification task — given two sentences A and B, predict whether B actually follows A in the original document.

- Given two sentences A and B, predict: is B the actual next sentence after A?
- Teaches understanding of sentence relationships
- (Later found to be less useful — RoBERTa removed it after showing it didn't help)

```
Positive example: A="The dog barked." B="The cat ran away." → IsNext
Negative example: A="The dog barked." B="The stock market fell." → NotNext
```

**WHY NSP was later abandoned:** RoBERTa's ablation study showed that removing NSP and training longer on MLM alone produced better representations. NSP's task was too easy — the model learned to distinguish topics rather than sentence coherence.

### Special Tokens

**What it is:** The reserved tokens BERT uses for specific structural purposes — you must know these for BERT fine-tuning.

```
[CLS]  - Classification token (first token, used for classification tasks)
         ← aggregates the whole sequence's meaning into one vector
[SEP]  - Separator between sentences
         ← tells BERT where sentence A ends and sentence B begins
[MASK] - Masked token placeholder during MLM training
         ← the model predicts what goes here
[PAD]  - Padding
         ← makes all sequences the same length in a batch
```

### How to Use BERT

```python
# Classification task: use [CLS] token output as the sentence representation
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize: adds [CLS] at start, [SEP] at end
inputs = tokenizer("The bank was robbed", return_tensors='pt')
# inputs['input_ids']: [[101, 1996, 2924, 2001, 19027, 102]]
#   101 = [CLS], 102 = [SEP]

outputs = model(**inputs)
# outputs.last_hidden_state: shape (1, seq_len, 768) — representations for ALL tokens
# outputs.pooler_output: shape (1, 768) — the [CLS] token representation specifically

# For classification: take [CLS] token and add a linear head
cls_representation = outputs.pooler_output  # shape: (batch, 768)
# Add a linear layer on top: nn.Linear(768, num_classes)
# Train only this head (or fine-tune entire model)

# For NER (Named Entity Recognition): use all token outputs
token_representations = outputs.last_hidden_state  # shape: (batch, seq_len, 768)
# Add a token-level classifier: nn.Linear(768, num_entity_types)
# Each token gets its own label

# For semantic similarity: compare [CLS] embeddings using cosine similarity
```

### Variants

**What it is:** Improved versions of BERT that addressed its various limitations.

| Model | Difference |
|-------|-----------|
| RoBERTa | Longer training, no NSP, dynamic masking (recomputed each epoch), larger batches |
| DistilBERT | 40% smaller, 60% faster, 97% of BERT performance (knowledge distillation) |
| ALBERT | Parameter sharing across layers (reduces overfitting), factorized embeddings |
| DeBERTa | Disentangled attention (separate content + position attention matrices) |

**WHY DeBERTa matters:** It separately models "what a word means" and "where a word is" in attention, rather than mixing them. This gives better performance on understanding benchmarks.

---

## 3. GPT Family — Generative Pre-trained Transformers

**What it is:** OpenAI's decoder-only Transformer series, trained to predict the next word — leading to the discovery that a single large model trained this way can do virtually anything.

### Released
OpenAI: GPT (2018) → GPT-2 (2019) → GPT-3 (2020) → GPT-4 (2023)

### Architecture
- **Decoder-only** Transformer
- **Causal (unidirectional)**: each token attends only to previous tokens
- Trained to predict the next token

**WHY decoder-only for generation?** Generation is inherently left-to-right — you write one word at a time, each word based only on what came before. The causal mask enforces this: each position can only see its own past. This is also why GPT can't look at the future during training — it must learn to predict it.

### Training Objective — Causal Language Modeling (CLM)

**What it is:** A self-supervised task — given all previous words, predict the next one. The label is the text itself, shifted by one position.

```
Input:  "The cat sat on the"
Target: "mat"
← No human labels needed — the next word IS the label
← Self-supervised: billions of books and webpages become free training data
```
Predict the next token given all previous tokens.
Loss = Cross-entropy over next-token predictions.

### GPT-2
- 1.5B parameters (largest version)
- 48 layers, d_model=1600, 25 heads
- Context: 1024 tokens
- Showed GPT can generate coherent long-form text
- OpenAI initially withheld it, fearing misuse (later released)

### GPT-3
- **175B parameters** — 100× larger than GPT-2
- 96 layers, d_model=12288, 96 heads
- Context: 2048 tokens
- Demonstrated **few-shot learning** (task description + examples in prompt)
- No fine-tuning needed for many tasks — just describe the task in the prompt

### GPT-3.5 / InstructGPT

**What it is:** GPT-3 further trained with human feedback to follow instructions and be helpful, rather than just predicting the next token on web text.

- GPT-3 fine-tuned with **RLHF** (Reinforcement Learning from Human Feedback)
- Aligns model to follow instructions and be helpful/harmless
- Human raters rank model outputs → reward model trained on rankings → model optimized via RL

**WHY RLHF matters:** Raw GPT-3 continues text in whatever direction the training data went. RLHF redirects it to be helpful. Without RLHF, asking GPT-3 "What is the capital of France?" might generate: "What is the capital of Germany? Berlin. What is the capital of Italy? Rome..." — it pattern-matches rather than answering.

### GPT-4
- Multimodal (text + images)
- Longer context (8k / 32k / 128k tokens)
- Better reasoning (possibly uses mixture-of-experts internally)

### Few-Shot Learning in GPT

**What it is:** GPT's ability to learn a new task from just a few examples shown in the prompt — no gradient updates required, just reading context.

```
Prompt:
"Translate English to French:
sea otter => loutre de mer      ← example 1 (shows the pattern)
peppermint => menthe poivrée    ← example 2 (reinforces the pattern)
cheese => fromage               ← example 3 (confirms the task)
plush toy =>"                   ← the actual question

GPT output: "jouet en peluche"
← No fine-tuning — just reading examples in the prompt!
```
No fine-tuning — just examples in the prompt!

**WHY in-context learning works:** GPT has learned patterns of (input, output) pairs from its training data. When you show it examples in the prompt, it recognizes "this is a translation task" and continues the pattern. The model has implicitly learned to learn from context.

---

## 4. T5 — Text-to-Text Transfer Transformer

**What it is:** Google's 2019 model with a unifying framework — convert every NLP task into a text-to-text problem, then use one model to solve all of them.

**Analogy:** Before T5, you needed a different hammer for each nail. T5 said: "every NLP task is a translation problem — translate input text to output text." One universal hammer.

### Released
Google, 2019

### Architecture
- **Encoder-Decoder** Transformer
- Everything is framed as text-to-text

### Key Innovation: Unified Text-to-Text Framework

**What it is:** The elegant insight that classification, translation, summarization, and QA can all be expressed as "given this text, produce this text."

Every NLP task is cast as sequence-to-sequence:

```
Translation:    "translate English to German: The house is wonderful." → "Das Haus ist wunderbar."
                ← task is part of the input text!
Summarization:  "summarize: [article text]" → "[summary]"
                ← same model, different prefix
Classification: "sentiment: This movie is great." → "positive"
                ← output is just text ("positive"), not a special classification head
QA:             "question: What is the capital of France? context: France's capital is Paris." → "Paris"
                ← answer is generated as text, not extracted from a pointer
```

**WHY this unification matters:** You can train ONE model on ALL tasks simultaneously. Multi-task training means the model learns shared representations across tasks, and each task benefits from the others. You also get a consistent interface — every task looks the same.

### Training Objective

**What it is:** T5's version of masked language modeling — but it masks entire spans of text, not individual tokens, which is better for seq2seq tasks.

- **Span Corruption**: randomly mask spans of tokens (not individual tokens)
- Model predicts the masked spans

```
Original: "Thank you for inviting me to your party last week."
Input:    "Thank you <X> me to your party <Y> week."
           ← <X> and <Y> are "sentinel tokens" marking masked spans
Target:   "<X> for inviting <Y> last <Z>"
           ← model must generate the full masked content in order
           ← <Z> signals end of sequence (no more masked spans)
```

**WHY span masking instead of token masking?** Masking whole spans forces the model to generate multiple tokens — practicing the seq2seq generation it will do at inference. Token-level masking (BERT-style) is too easy for a model designed to generate sequences.

### Variants
| Model | Parameters |
|-------|-----------|
| T5-small | 60M |
| T5-base | 220M |
| T5-large | 770M |
| T5-XL | 3B |
| T5-XXL | 11B |
| Flan-T5 | Instruction-tuned T5 |

### Flan-T5

**What it is:** T5 that has been instruction-tuned on 1000+ different NLP tasks written as natural language prompts — dramatically improving its zero-shot and few-shot generalization.

- T5 fine-tuned on **1000+ NLP tasks** with natural language instructions
- Significantly better zero-shot and few-shot performance
- Following instructions in natural language rather than task-specific formatting

**WHY instruction tuning works:** Flan-T5 has seen so many different phrasings of tasks that when it sees a new instruction, it can recognize the pattern. "Classify the sentiment" and "Is this review positive or negative?" both lead to the same type of output — the model learns this generalization.

---

## 5. Modern Open-Source LLMs

**What it is:** The major families of open-weights models that democratized access to frontier-quality LLMs.

### LLaMA Family (Meta)
| Model | Params | Context | Notes |
|-------|--------|---------|-------|
| LLaMA 1 | 7B–65B | 2048 | First major open-source LLM — sparked the open-source movement |
| LLaMA 2 | 7B–70B | 4096 | Commercially usable license — companies could deploy it |
| LLaMA 3 | 8B–70B | 8192 | Strong performance, better tokenizer (128K vocab) |
| LLaMA 3.1 | 8B–405B | 128K | Long context, multilingual, frontier-quality at 405B |

Key changes from original Transformer:
- **RoPE** positional encoding — relative positions, better length generalization
- **SwiGLU** activation — gated FFN, better than GELU
- **RMSNorm** instead of LayerNorm — faster, simpler
- **Pre-normalization** (norm before attention, not after) — more stable
- **GQA** (Grouped Query Attention) in larger models — 8× smaller KV cache

### Mistral (Mistral AI)
- Sliding Window Attention (SWA) — attends to nearest 4096 tokens, propagates through layers
- Grouped Query Attention (GQA) — 8× KV cache compression
- 7B model outperforms LLaMA 2 13B — better efficiency, not just size

### Falcon (TII)
- Multi-query attention — even more aggressive KV cache compression
- Strong performance on code
- Built by Technology Innovation Institute in UAE

### GPT-NeoX / Pythia (EleutherAI)
- Fully open source (weights + training code + data — all public)
- Good for research and reproducibility studies

---

## 6. Key Architectural Innovations in Modern LLMs

**What it is:** The specific technical choices that distinguish modern LLMs from the 2017 Transformer — every one of these is a potential interview question.

### Grouped Query Attention (GQA)

**What it is:** A memory optimization where multiple Query heads share a single Key-Value pair, drastically reducing the KV cache size without significant quality loss.

- Multiple query heads share a single Key-Value head
- Reduces KV cache memory significantly
- Used in LLaMA 2 70B, Mistral, LLaMA 3

```
Multi-Head (standard): Q=h heads, K=h heads, V=h heads
← 64 query heads + 64 key heads + 64 value heads = huge KV cache
GQA:                   Q=h heads, K=g groups, V=g groups (g << h)
← 64 query heads + 8 key heads + 8 value heads = 8× smaller KV cache
MQA:                   Q=h heads, K=1 head, V=1 head
← 64 query heads + 1 key head + 1 value head = 64× smaller KV cache (quality drops)
```

```python
# GQA implementation — groups of Q heads share one K,V head:
num_heads = 64       # query heads
num_kv_heads = 8     # key-value heads (much fewer)
d_k = d_model // num_heads  # dimension per head

# Project queries: separate for each of the 64 heads
Q = linear_Q(x).reshape(batch, seq, num_heads, d_k)      # (B, S, 64, d_k)

# Project keys/values: only 8 per layer (not 64)
K = linear_K(x).reshape(batch, seq, num_kv_heads, d_k)    # (B, S, 8, d_k)
V = linear_V(x).reshape(batch, seq, num_kv_heads, d_k)    # (B, S, 8, d_k)

# Repeat K, V to match num_heads for attention computation:
K = K.repeat_interleave(num_heads // num_kv_heads, dim=2)  # (B, S, 64, d_k)
V = V.repeat_interleave(num_heads // num_kv_heads, dim=2)  # (B, S, 64, d_k)
# Each KV head "serves" 8 query heads (64/8 = 8)
# Query heads 0-7 all use KV head 0; heads 8-15 use KV head 1; etc.
# WHY repeat_interleave? It's equivalent to having 64 KV heads — just memory-shared
# BUT: we only CACHE the 8 KV heads, not the repeated 64 — that's the memory saving

# Standard attention computation (same as always):
scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
weights = F.softmax(scores, dim=-1)
output = weights @ V
```

### SwiGLU Activation

**What it is:** A gated activation function where one linear projection controls information flow of another — more expressive than a single activation.

```
SwiGLU(x) = Swish(xW) ⊙ (xV)
← Swish = SiLU = x * sigmoid(x): smooth, slightly negative values pass
← ⊙ = element-wise multiply
← Two separate learned projections, multiplied element-wise
```
- Better than GELU in practice for LLMs
- Used in LLaMA, PaLM

### RMSNorm

**What it is:** A simplified Layer Norm that only divides by root mean square — removing the mean-subtraction step that turns out to be unimportant.

```
RMSNorm(x) = x / RMS(x) * γ
← RMS(x) = sqrt(mean(x²)) — just the magnitude, no centering
← γ: learned scale parameter per dimension
```
- Simpler than LayerNorm (no mean subtraction)
- Faster, similar performance
- Used in LLaMA, T5

---

## 7. Tokenization

**What it is:** The process of converting raw text into integer IDs that the model can process — how text becomes numbers.

**Analogy:** Like converting sheet music into MIDI numbers. You need a mapping from symbols (notes/words) to numbers before a computer can process them. Tokenization is that mapping.

Before text enters an LLM, it must be **tokenized** (split into tokens with integer IDs).

### BPE (Byte-Pair Encoding)

**What it is:** A compression-inspired algorithm that iteratively merges the most common character pairs into a single token, building a vocabulary of subword units.

- Used by GPT-2, GPT-3, LLaMA
- Iteratively merges most frequent character pairs into subword units
- Vocabulary size: typically 32K–100K

```
Training BPE on text:
Step 1: Start with characters: ["u", "n", "h", "a", "p", "p", "i", "n", "e", "s", "s"]
Step 2: Find most common pair: "p"+"p" → merge to "pp"
Step 3: Find next most common: "un"+"happiness" → "unhappiness"
Result: "unhappiness" → ["un", "happiness"]  # BPE splits at learned subword boundaries

At inference:
"unhappiness" → ["un", ##"happiness"] → [IDs: 4895, 7842]
"unfamiliar"  → ["un", ##"familiar"]  → [IDs: 4895, 9321]  ← "un" reused!
← Subword sharing means rare words are built from common pieces
← "unknowing" works even if the model never saw it — "un" + "knowing"
```

### WordPiece

**What it is:** BPE's cousin used by BERT — similar approach but uses likelihood instead of frequency to decide which merges to make.

- Used by BERT
- Similar to BPE but uses likelihood instead of frequency
- Tokens not in vocabulary get split into known subwords, prefixed with ##

### SentencePiece

**What it is:** A language-agnostic tokenizer that works directly on raw text without language-specific preprocessing — ideal for multilingual models.

- Language-agnostic tokenizer (no language-specific rules)
- Used by T5, LLaMA 3
- Handles whitespace differently — treats it as a regular character

### Example
```
"unhappiness" → ["un", "happiness"]  # BPE — split into morphemes
"tokenization" → ["token", "##ization"]  # WordPiece — ## marks continuation
```

```python
from transformers import AutoTokenizer

# LLaMA 3 tokenizer (BPE + SentencePiece, 128K vocabulary):
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokens = tokenizer("Hello world!")
# {'input_ids': [128000, 9906, 1917, 0]}
# 128000 = <|begin_of_text|> (BOS token — required by LLaMA 3)
# 9906   = "Hello"
# 1917   = " world"
# 0      = "!"

# Note: tokens ≠ words exactly
text = "ChatGPT is powerful"
tokens = tokenizer(text)
# Might give: ["Chat", "G", "PT", " is", " powerful"]
# "ChatGPT" is a rare compound — split into subwords
# Tokenization affects how rare terms are handled
```

Tokens ≠ Words. A word can be 1-3 tokens. English: ~1.3 tokens/word on average.
Code can be 2-5 tokens/word (unfamiliar syntax).

---

## 8. Context Window Comparison

**What it is:** How different models' maximum context lengths have grown over time — understanding this progression shows how the field has evolved.

| Model | Context Window |
|-------|---------------|
| BERT base | 512 tokens |
| GPT-2 | 1024 tokens |
| GPT-3 | 2048 tokens |
| LLaMA 2 | 4096 tokens |
| LLaMA 3.1 | 128K tokens |
| Claude 3 | 200K tokens |
| GPT-4 Turbo | 128K tokens |
| Gemini 1.5 | 1M tokens |

**WHY context length matters:** 512 tokens = about 1 page. 128K tokens = about 250 pages. The jump from 2K to 128K context unlocks: entire codebases, full research papers, long meetings, complete books.

---

## 9. Model Parameters & Memory

**What it is:** A practical formula for estimating how much GPU memory you need to load a model — essential for production planning.

Rule of thumb:
- 1B parameters ≈ 2GB memory (in FP16/BF16)
- 7B model ≈ 14GB GPU RAM (FP16), 7GB (INT8), 4GB (INT4)

```python
# Memory estimation for model loading:
def estimate_memory_gb(num_params_billions, precision='fp16'):
    bytes_per_param = {'fp32': 4, 'fp16': 2, 'bf16': 2, 'int8': 1, 'int4': 0.5}
    return num_params_billions * 1e9 * bytes_per_param[precision] / 1e9

# LLaMA 3 8B:
print(estimate_memory_gb(8, 'bf16'))  # 16 GB — needs one A100 40GB
print(estimate_memory_gb(8, 'int8'))  # 8 GB  — fits on most modern GPUs
print(estimate_memory_gb(8, 'int4'))  # 4 GB  — fits on a gaming GPU

# LLaMA 3 70B:
print(estimate_memory_gb(70, 'bf16'))  # 140 GB — needs 2× A100 80GB
print(estimate_memory_gb(70, 'int4'))  # 35 GB  — fits on 1× A100 40GB
```

---

## Structured Output — Making LLMs Return Valid JSON

**What it is:** The production engineering challenge of getting LLMs to reliably return structured data rather than free-form text — critical for building applications on top of LLMs.

**Analogy:** Asking a human to fill out a form vs asking them to describe what the form would say. The form gives you structured, parseable output. The free description gives you something you have to parse manually.

Problem: LLMs sometimes return malformed JSON, breaking your application.

Solutions:

**1. OpenAI JSON mode:**
```python
response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},  # force JSON output
    messages=[{
        "role": "user",
        "content": "Extract name and age from: John Smith is 25 years old."
    }]
)
# Forces valid JSON output — model is constrained to produce parseable JSON
# But: schema is not validated — could return {"anything": "really"}
parsed = json.loads(response.choices[0].message.content)
```

**2. OpenAI Structured Output (stronger):**
```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str   # model must include this field with a string value
    age: int    # model must include this field with an integer value

completion = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "John Smith is 25 years old."}],
    response_format=Person,  # schema is enforced at the token level
)
person = completion.choices[0].message.parsed  # typed Python object — no json.loads needed
print(person.name)  # "John Smith" — guaranteed to be a string
print(person.age)   # 25 — guaranteed to be an integer
# WHY stronger: schema is enforced, not just JSON validity
```

**3. Outlines library (open-source, works with any model):**
```python
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B")
# Load any open-source model

generator = outlines.generate.json(model, Person)
# Create a constrained generator for this Pydantic schema

result = generator("Extract person from: John is 25 years old")
# Uses constrained decoding — at each step, ONLY tokens that form valid JSON
# matching the schema are allowed in the sampling distribution
# Invalid tokens get probability 0 before sampling → physically impossible to generate invalid output
# WHY this is the strongest approach: violation is literally impossible at the token level
```

**Interview answer:** "How do you ensure LLM always returns valid JSON?" → "Three approaches: OpenAI's response_format parameter for cloud models, Pydantic integration for typed outputs, or Outlines library for open-source models which uses grammar-constrained decoding at the token level."

---

## 10. Interview Questions — LLM Models

**Q: What is the difference between BERT and GPT?**
> BERT is an encoder-only model trained with masked language modeling — it sees full bidirectional context and is best for understanding tasks (classification, NER). GPT is a decoder-only model trained with causal language modeling — it only sees past tokens and is designed for generation tasks. BERT can't generate text; GPT isn't directly used for classification without adaptation.

**Q: What is the training objective of GPT?**
> Causal Language Modeling (CLM): predict the next token given all previous tokens. The loss is cross-entropy between the predicted probability distribution and the actual next token.

**Q: Why is T5's text-to-text framing powerful?**
> It provides a unified interface for any NLP task — the same model architecture and fine-tuning approach works for translation, summarization, classification, QA, etc. This eliminates task-specific architectural changes.

**Q: What is the difference between GPT-3 and InstructGPT?**
> InstructGPT fine-tunes GPT-3 using RLHF (Reinforcement Learning from Human Feedback) to align it with human preferences — making it follow instructions, be helpful, and avoid harmful outputs.

**Q: What improvements does LLaMA have over the original Transformer?**
> RoPE positional encoding, SwiGLU activation, RMSNorm instead of LayerNorm, pre-normalization architecture, and Grouped Query Attention — all improving efficiency and performance.

**Q: What is tokenization and why does it matter?**
> Tokenization converts raw text into integer token IDs using a vocabulary. It matters because it determines how the model represents text — the vocabulary size affects model capacity, and subword tokenization handles rare/unknown words gracefully.

---

## Quick Reference Cheat Sheet

```
BERT:    Encoder-only | Bidirectional | MLM + NSP | Understanding tasks
         [CLS] token for classification | [SEP] between sentences
         110M (base), 340M (large) | 512 token context

GPT:     Decoder-only | Causal | CLM (next token) | Generation tasks
         In-context learning (few-shot) without fine-tuning
         175B (GPT-3) | 2048 token context → 128K (GPT-4)

T5:      Encoder-Decoder | Text-to-Text | Span corruption | Seq2seq tasks
         All tasks framed as "translate input text to output text"
         60M–11B | Flan-T5 = instruction-tuned variant

LLaMA improvements: RoPE, SwiGLU, RMSNorm, Pre-norm, GQA
Tokenization: BPE (GPT), WordPiece (BERT), SentencePiece (T5, LLaMA3)
Memory: 1B params ≈ 2GB in FP16 | 7B → 14GB FP16, 7GB INT8, 4GB INT4
```
