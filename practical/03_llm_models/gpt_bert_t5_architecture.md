# 03 — LLM Models: GPT, BERT, T5 & Modern LLMs

> Know the architecture, training objective, and use cases of each model family.

---

## 1. The Three Paradigms

| Model | Architecture | Training Objective | Best For |
|-------|-------------|-------------------|---------|
| **BERT** | Encoder-only | Masked Language Modeling | Understanding (classification, NER, QA) |
| **GPT** | Decoder-only | Causal Language Modeling | Generation (text, code, chat) |
| **T5** | Encoder-Decoder | Text-to-Text (masked spans) | Seq2seq (translation, summarization) |

---

## 2. BERT — Bidirectional Encoder Representations from Transformers

### Released
Google, 2018

### Architecture
- **Encoder-only** Transformer
- **Bidirectional**: each token attends to ALL other tokens (left + right context)
- 12 layers (base) / 24 layers (large)
- d_model: 768 (base) / 1024 (large)
- Parameters: 110M (base) / 340M (large)

### Training Objectives

**1. Masked Language Modeling (MLM)**
- Randomly mask 15% of tokens with [MASK]
- Model predicts the original masked token
- Teaches the model to understand context

```
Input:  "The cat [MASK] on the mat"
Target: "sat"
```

**2. Next Sentence Prediction (NSP)**
- Given two sentences A and B, predict: is B the actual next sentence after A?
- Teaches understanding of sentence relationships
- (Later found to be less useful — RoBERTa removed it)

### Special Tokens
```
[CLS]  - Classification token (first token, used for classification tasks)
[SEP]  - Separator between sentences
[MASK] - Masked token placeholder
[PAD]  - Padding
```

### How to Use BERT
- Add a classification head on [CLS] token for classification
- Add token-level head for NER, POS tagging
- Use embeddings for semantic similarity

### Variants
| Model | Difference |
|-------|-----------|
| RoBERTa | Longer training, no NSP, dynamic masking |
| DistilBERT | 40% smaller, 60% faster, 97% of BERT performance |
| ALBERT | Parameter sharing across layers, factorized embeddings |
| DeBERTa | Disentangled attention (separate position + content attention) |

---

## 3. GPT Family — Generative Pre-trained Transformers

### Released
OpenAI: GPT (2018) → GPT-2 (2019) → GPT-3 (2020) → GPT-4 (2023)

### Architecture
- **Decoder-only** Transformer
- **Causal (unidirectional)**: each token attends only to previous tokens
- Trained to predict the next token

### Training Objective — Causal Language Modeling (CLM)
```
Input:  "The cat sat on the"
Target: "mat"
```
Predict the next token given all previous tokens.
Loss = Cross-entropy over next-token predictions.

### GPT-2
- 1.5B parameters (largest version)
- 48 layers, d_model=1600, 25 heads
- Context: 1024 tokens
- Showed GPT can generate coherent long-form text

### GPT-3
- **175B parameters**
- 96 layers, d_model=12288, 96 heads
- Context: 2048 tokens
- Demonstrated **few-shot learning** (task description + examples in prompt)
- No fine-tuning needed for many tasks

### GPT-3.5 / InstructGPT
- GPT-3 fine-tuned with **RLHF** (Reinforcement Learning from Human Feedback)
- Aligns model to follow instructions and be helpful/harmless

### GPT-4
- Multimodal (text + images)
- Longer context (8k / 32k / 128k tokens)
- Better reasoning

### Few-Shot Learning in GPT
```
Prompt:
"Translate English to French:
sea otter => loutre de mer
peppermint => menthe poivrée
cheese => fromage
plush toy =>"

GPT output: "jouet en peluche"
```
No fine-tuning — just examples in the prompt!

---

## 4. T5 — Text-to-Text Transfer Transformer

### Released
Google, 2019

### Architecture
- **Encoder-Decoder** Transformer
- Everything is framed as text-to-text

### Key Innovation: Unified Text-to-Text Framework
Every NLP task is cast as sequence-to-sequence:

```
Translation:    "translate English to German: The house is wonderful." → "Das Haus ist wunderbar."
Summarization:  "summarize: [article text]" → "[summary]"
Classification: "sentiment: This movie is great." → "positive"
QA:             "question: What is the capital of France? context: France's capital is Paris." → "Paris"
```

### Training Objective
- **Span Corruption**: randomly mask spans of tokens (not individual tokens)
- Model predicts the masked spans

```
Original: "Thank you for inviting me to your party last week."
Input:    "Thank you <X> me to your party <Y> week."
Target:   "<X> for inviting <Y> last <Z>"
```

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
- T5 fine-tuned on **1000+ NLP tasks** with natural language instructions
- Significantly better zero-shot and few-shot performance

---

## 5. Modern Open-Source LLMs

### LLaMA Family (Meta)
| Model | Params | Context | Notes |
|-------|--------|---------|-------|
| LLaMA 1 | 7B–65B | 2048 | First major open-source LLM |
| LLaMA 2 | 7B–70B | 4096 | Commercially usable |
| LLaMA 3 | 8B–70B | 8192 | Strong performance, better tokenizer |
| LLaMA 3.1 | 8B–405B | 128K | Long context, multilingual |

Key changes from original Transformer:
- **RoPE** positional encoding
- **SwiGLU** activation (instead of ReLU/GELU)
- **RMSNorm** instead of LayerNorm (faster)
- **Pre-normalization** (norm before attention, not after)
- **GQA** (Grouped Query Attention) in larger models

### Mistral (Mistral AI)
- Sliding Window Attention (SWA) for efficient long context
- Grouped Query Attention (GQA)
- 7B model outperforms LLaMA 2 13B

### Falcon (TII)
- Multi-query attention
- Strong performance on code

### GPT-NeoX / Pythia (EleutherAI)
- Fully open source (weights + training code + data)
- Good for research

---

## 6. Key Architectural Innovations in Modern LLMs

### Grouped Query Attention (GQA)
- Multiple query heads share a single Key-Value head
- Reduces KV cache memory significantly
- Used in LLaMA 2 70B, Mistral, LLaMA 3

```
Multi-Head:  Q=h heads, K=h heads, V=h heads
GQA:         Q=h heads, K=g groups, V=g groups (g << h)
MQA:         Q=h heads, K=1 head, V=1 head
```

### SwiGLU Activation
```
SwiGLU(x) = Swish(xW) ⊙ (xV)
```
- Better than GELU in practice for LLMs
- Used in LLaMA, PaLM

### RMSNorm
```
RMSNorm(x) = x / RMS(x) * γ
```
- Simpler than LayerNorm (no mean subtraction)
- Faster, similar performance
- Used in LLaMA, T5

---

## 7. Tokenization

Before text enters an LLM, it must be **tokenized** (split into tokens with integer IDs).

### BPE (Byte-Pair Encoding)
- Used by GPT-2, GPT-3, LLaMA
- Iteratively merges most frequent character pairs into subword units
- Vocabulary size: typically 32K–100K

### WordPiece
- Used by BERT
- Similar to BPE but uses likelihood instead of frequency

### SentencePiece
- Language-agnostic tokenizer
- Used by T5, LLaMA 3

### Example
```
"unhappiness" → ["un", "happiness"]  # BPE
"tokenization" → ["token", "##ization"]  # WordPiece
```

Tokens ≠ Words. A word can be 1-3 tokens. English: ~1.3 tokens/word on average.

---

## 8. Context Window Comparison

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

---

## 9. Model Parameters & Memory

Rule of thumb:
- 1B parameters ≈ 2GB memory (in FP16)
- 7B model ≈ 14GB GPU RAM (FP16), 7GB (INT8), 4GB (INT4)

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
GPT:     Decoder-only | Causal | CLM (next token) | Generation tasks
T5:      Encoder-Decoder | Text-to-Text | Span corruption | Seq2seq tasks

LLaMA improvements: RoPE, SwiGLU, RMSNorm, Pre-norm, GQA
Tokenization: BPE (GPT), WordPiece (BERT), SentencePiece (T5, LLaMA3)
Memory: 1B params ≈ 2GB in FP16
```
