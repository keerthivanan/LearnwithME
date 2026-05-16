# Tokenization Deep Dive — BPE, SentencePiece, Vocabulary, Multilingual

> Tokenization is asked in every LLM interview. Most people give a surface answer.
> This file gives you the deep answer that stands out.

---

## WHY TOKENIZATION MATTERS MORE THAN YOU THINK

Tokenization is not just "split text into pieces."
It directly affects:
- **Model capacity:** vocabulary size = embedding table size
- **Context efficiency:** 1 token per word vs 3 tokens per rare word
- **Multilingual performance:** English-centric tokenizers are unfair to other languages
- **Math/code performance:** Numbers and operators tokenized poorly = poor reasoning
- **Cost:** More tokens = more money on API calls

---

## THE EVOLUTION OF TOKENIZATION

```
1990s:  Word tokenization     → "don't" → ["don't"] (can't handle unknowns)
2000s:  Character tokenization → "hello" → ["h","e","l","l","o"] (too many tokens)
2015:   BPE                    → "unhappiness" → ["un","happiness"] (SWEET SPOT)
2016:   WordPiece (Google)     → "tokenization" → ["token","##ization"]
2018:   SentencePiece          → Language-agnostic, treats text as raw bytes
2019:   Byte-level BPE (GPT-2) → Never fails on any input, handles any Unicode
```

---

## BYTE PAIR ENCODING (BPE) — Step by Step

### The Algorithm

**Input:** Large text corpus
**Output:** Vocabulary of subword tokens + merge rules

**Step 1: Start with character vocabulary**
```
Initial vocab: {a, b, c, d, ..., z, A, ..., Z, 0-9, !, ?, ...} + special tokens
Initial text representation:
  "low" → ["l", "o", "w"]
  "lower" → ["l", "o", "w", "e", "r"]
  "newest" → ["n", "e", "w", "e", "s", "t"]
```

**Step 2: Count all adjacent pairs**
```
Pairs: (l,o)=3, (o,w)=3, (w,e)=1, (w,_)=2, (e,r)=1, (n,e)=1, ...
```

**Step 3: Merge most frequent pair**
```
Most frequent: (l,o) with count 3
Merge: l + o → lo
New vocab: {..., lo, ...}
Update all occurrences: "low" → ["lo", "w"]
```

**Step 4: Repeat until vocabulary size reached**
```
Round 2: most frequent now = (lo, w) with count 3
  Merge: lo + w → low
  "low" → ["low"]
  "lower" → ["low", "e", "r"]

Round 3: most frequent = (low, e) with count 2
  Merge: low + e → lowe
  "lower" → ["lowe", "r"]

Continue until vocab_size = 50,000 (or whatever target)
```

**Final merge rules (learned):**
```
l + o → lo
lo + w → low
low + e → lowe
lowe + r → lower
n + e → ne
ne + w → new
new + e → newe
newe + s → newes
...
```

### At Inference Time
Apply merge rules in order to tokenize new text:
```
New word: "lowest"
Apply rules:
  l,o,w,e,s,t → lo,w,e,s,t (rule 1: l+o)
  → low,e,s,t (rule 2: lo+w)
  → lowe,s,t (rule 3: low+e)
  → lowe,s,t (rule for "e"+"s" doesn't exist, stop here)
  → ["lowe", "s", "t"]
```

---

## BYTE-LEVEL BPE (GPT-2, LLaMA) — The Modern Standard

### The Problem With Character-Level BPE
Unicode has ~150,000+ characters. Starting vocabulary is huge.
Some Unicode characters appear so rarely they never merge.

### The Solution: Start From Bytes
Every text = sequence of UTF-8 bytes (0-255).
**Starting vocabulary: only 256 tokens (bytes 0-255).**

```
"café" → UTF-8 bytes: [99, 97, 102, 195, 169]
         In hex:       [0x63, 0x61, 0x66, 0xC3, 0xA9]
```

**Then BPE merges as usual, but on bytes.**

**Benefits:**
- Handles ANY Unicode input — impossible to have an unknown character
- No `[UNK]` token needed
- Works for code, emojis, math symbols, every language

**GPT-2 vocabulary:** 50,257 tokens (50,000 BPE merges + 256 byte tokens + `<|endoftext|>`)
**LLaMA 3 vocabulary:** 128,000 tokens (much larger → better multilingual, fewer tokens per text)

---

## WORDPIECE (BERT) vs BPE (GPT)

| Property | BPE | WordPiece |
|----------|-----|-----------|
| Merge criterion | Most frequent pair | Highest likelihood increase |
| Notation | No prefix | `##` prefix for continuation |
| Example | `["token", "ization"]` | `["token", "##ization"]` |
| Training | Bottom-up merging | Maximum likelihood |
| Unknown handling | Falls back to bytes (byte-BPE) | Uses [UNK] token |

### WordPiece Example
```
"tokenization" → ["token", "##ization"]
"unrelated"    → ["un", "##related"]
"##" means "this piece continues a word (no space before it)"
```

The `##` prefix helps BERT understand word boundaries — useful for NER.

---

## SENTENCEPIECE — LANGUAGE-AGNOSTIC TOKENIZATION

**Used by:** T5, LLaMA 1/2, ALBERT, XLNet

### Key Difference
SentencePiece treats the entire input as a raw character stream.
It doesn't rely on word boundaries (spaces) — crucial for languages like Chinese, Japanese, Arabic.

```
English (space-separated): "Hello world" → clear word boundaries
Chinese (no spaces): "你好世界" → no boundaries — SentencePiece handles this natively
```

### Two SentencePiece Algorithms

**SentencePiece + BPE:** Apply BPE on raw text (including spaces as special characters)
```
"Hello world" → "▁Hello▁world" (▁ = space character)
               → ["▁Hello", "▁world"]
```

**SentencePiece + Unigram Language Model:**
Instead of merging, start with a large vocabulary and PRUNE tokens.
Train a unigram language model, remove tokens that contribute least to likelihood.
More mathematically principled but slightly more complex.

---

## TIKTOKEN (OpenAI)

**Used by:** GPT-3.5, GPT-4, GPT-4o, text-embedding-ada-002

Tiktoken is a fast BPE tokenizer (Rust implementation, 10-100× faster than HuggingFace).

```python
import tiktoken

# Different encoding for different models
enc = tiktoken.encoding_for_model("gpt-4")
# or: enc = tiktoken.get_encoding("cl100k_base")

text = "Hello, I am learning about tokenization!"
tokens = enc.encode(text)
print(tokens)   # [9906, 11, 358, 1097, 6975, 922, 47058, 0]
print(len(tokens))  # 8 tokens

decoded = enc.decode(tokens)
print(decoded)  # "Hello, I am learning about tokenization!"

# Count tokens for an API call (important for cost estimation!)
def count_tokens(text, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))
```

---

## VOCABULARY SIZE — THE ENGINEERING TRADE-OFF

| Vocab Size | Examples | Tokens/Word | Pros | Cons |
|-----------|---------|------------|------|------|
| 8K | Early BERT | ~2.5 | Less memory | Many rare words split badly |
| 32K | BERT, LLaMA 1/2 | ~1.5 | Balanced | Weak multilingual |
| 50K | GPT-2 | ~1.3 | Good English | Poor multilingual |
| 100K | GPT-4, LLaMA 3 | ~1.1 | Efficient | Large embedding table |
| 256K | Gemma | ~1.05 | Very efficient | More memory |

### Memory Impact of Vocabulary Size
```
Embedding table size = vocab_size × d_model × bytes_per_param

LLaMA 2 (32K vocab, d=4096, BF16):
  32,000 × 4,096 × 2 bytes = 256 MB

LLaMA 3 (128K vocab, d=4096, BF16):
  128,000 × 4,096 × 2 bytes = 1 GB

For a 7B model, the embedding table is ~15% of total memory.
```

---

## MULTILINGUAL TOKENIZATION PROBLEM

### The Fertility Problem

**Tokenizer fertility:** average tokens per word in a language.
A tokenizer trained mostly on English data is unfair to other languages.

```
English: "Hello, my name is Claude"     → 6 tokens (1:1 ratio)
French:  "Bonjour, je m'appelle Claude" → 8 tokens (still ok)
Arabic:  "مرحبا، اسمي كلود"              → 14 tokens (2.3x more expensive!)
Thai:    "สวัสดีฉันชื่อคลอด"              → 25 tokens (4x more expensive!)
```

**Implications:**
- Non-English languages cost more API tokens
- Context window used less efficiently for non-English
- Model might have less "space" to reason in non-English contexts

### Solutions
- Larger vocabularies (LLaMA 3: 128K vs 32K — better multilingual)
- Language-specific tokenizers (Qwen uses extra Chinese tokens)
- SentencePiece (language-agnostic design)

---

## SPECIAL TOKENS — THE CONTROL SIGNALS

Every model has special tokens that control behavior:

### Common Special Tokens
| Token | Purpose | Example |
|-------|---------|---------|
| `<|endoftext|>` | End of document (GPT-2) | Marks document boundary |
| `<s>` | Start of sequence (LLaMA) | Beginning of any sequence |
| `</s>` | End of sequence (LLaMA) | End generation here |
| `[BOS]` | Beginning of sentence | Same as `<s>` |
| `[EOS]` | End of sentence | Model stops here |
| `[PAD]` | Padding | Fill shorter sequences in batch |
| `[UNK]` | Unknown token | WordPiece for unknown chars |
| `[MASK]` | Masked token (BERT) | Position to predict |
| `[CLS]` | Classification token (BERT) | First token, used for classification |
| `[SEP]` | Separator (BERT) | Between sentence pairs |

### Chat Templates — How Special Tokens Enable Conversations

**LLaMA 2 Chat format:**
```
<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

What is 2+2? [/INST] 4 </s>
<s>[INST] And 3+3? [/INST] 6 </s>
```

**LLaMA 3 format (ChatML-based):**
```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What is 2+2?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
4
<|eot_id|>
```

**Why this matters for fine-tuning:**
You MUST use the correct chat template when fine-tuning chat models.
Wrong template = model doesn't understand conversation structure = poor results.

```python
# HuggingFace handles this automatically
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 2+2?"},
]

# Apply correct template automatically
formatted = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)
```

---

## THE TOKENIZATION PROBLEM FOR MATH & CODE

### Why LLMs Are Bad At Arithmetic

```
"1234 + 5678 = ?"

Tokenization: "1234" → ["12", "34"]   (split mid-number!)
              "5678" → ["56", "78"]

The model can't "see" individual digits — it sees chunks.
Column-by-column addition is impossible when digits are merged.
```

**Solutions:**
- Chain-of-Thought: "4+8=12, write 2 carry 1; 3+7+1=11..."
- Reasoning models (o1): learn to decompose into steps
- Future: token-free models that process raw bytes

### Code Tokenization

Python code: generally tokenizes well (spaces preserved)
```python
"def hello():" → ["def", " hello", "():", ...]
```

Whitespace-sensitive: indent matters in Python
```python
"    return x" → ["    ", "return", " x"]  # leading spaces preserved
```

---

## INTERVIEW BLAST — Tokenization

**"How does BPE tokenization work?"**
> "BPE starts with a character vocabulary and iteratively merges the most frequent
> adjacent pairs into new tokens, repeating until we reach the target vocabulary size.
> GPT-2 uses byte-level BPE — starting from 256 byte tokens — which handles any
> Unicode input without unknown token issues. The learned merge rules are applied
> sequentially at inference to tokenize new text."

**"Why does vocabulary size matter?"**
> "Vocabulary size affects three things: memory (embedding table = vocab × d_model),
> tokenization efficiency (larger vocab → fewer tokens per text → more fits in context
> window), and multilingual fairness (small English-centric vocabulary needs 4× more
> tokens for Thai than English). LLaMA 3 doubled vocabulary from 32K to 128K specifically
> to improve multilingual efficiency and coding."

**"What is the fertility problem in multilingual tokenization?"**
> "Fertility is the average number of tokens per word for a language. Tokenizers trained
> mostly on English have high fertility for other languages — Arabic might need 4× more
> tokens than English for the same semantic content. This means non-English users get
> less effective context window, pay more API costs, and the model has less capacity to
> reason in their language."
