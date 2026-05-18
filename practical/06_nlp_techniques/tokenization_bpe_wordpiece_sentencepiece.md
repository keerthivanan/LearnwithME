# Tokenization Deep Dive — BPE, SentencePiece, Vocabulary, Multilingual

> Tokenization is asked in every LLM interview. Most people give a surface answer.
> This file gives you the deep answer that stands out.

---

## WHY TOKENIZATION MATTERS MORE THAN YOU THINK

**What it is:** Tokenization is the process of breaking raw text into pieces (called tokens) that the model can understand. It is the very first step before any language model sees your text.

Think of it like this: before you can teach a child to read, you first have to agree on what the "letters" and "words" are. Tokenization is making that agreement for an LLM.

It directly affects everything downstream:
- **Model capacity:** vocabulary size = the size of the embedding lookup table in memory
- **Context efficiency:** if every rare word costs 3 tokens instead of 1, you waste context window space
- **Multilingual performance:** tokenizers trained mostly on English waste tokens on other languages
- **Math/code performance:** if numbers are split mid-digit, the model can't do column arithmetic
- **Cost:** every extra token = extra money on API calls

---

## THE EVOLUTION OF TOKENIZATION

**What it is:** A timeline showing how the field moved from naive word-splitting to the modern byte-level approach used in GPT-4 and LLaMA 3.

```
1990s:  Word tokenization     → "don't" → ["don't"]         (fails on unseen words)
2000s:  Character tokenization → "hello" → ["h","e","l","l","o"] (too many tokens)
2015:   BPE                    → "unhappiness" → ["un","happiness"]  (SWEET SPOT)
2016:   WordPiece (Google)     → "tokenization" → ["token","##ization"]
2018:   SentencePiece          → Language-agnostic, treats text as raw bytes
2019:   Byte-level BPE (GPT-2) → Never fails on any input, handles any Unicode
```

**WHY:** The core insight driving every step: we want tokens that are small enough to handle rare/new words, but large enough that common words cost only 1 token. BPE nails this balance.

---

## BYTE PAIR ENCODING (BPE) — Step by Step

**What it is:** BPE is an algorithm that starts with individual characters and repeatedly merges the two most common adjacent characters into a new token. You keep merging until you have the vocabulary size you want.

**Real-world analogy:** Imagine you are typing abbreviations. You notice you always type "re" together, so you make a macro for it. Then you notice "pre" (p + re) is common, so you make a macro for that too. BPE is the same idea — it learns the most useful "macros" (merge rules) from your training data.

### The Algorithm

**Input:** Large text corpus
**Output:** Vocabulary of subword tokens + merge rules

**Step 1: Start with character vocabulary**
```
Initial vocab: {a, b, c, d, ..., z, A, ..., Z, 0-9, !, ?, ...} + special tokens
Initial text representation:
  "low"    → ["l", "o", "w"]
  "lower"  → ["l", "o", "w", "e", "r"]
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
  "low"   → ["low"]
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

**WHY:** You save these merge rules as the tokenizer's "memory." At inference, you replay these rules in order to tokenize any new text — even words the tokenizer never saw during training.

### At Inference Time
Apply merge rules in order to tokenize new text:
```
New word: "lowest"
Apply rules:
  l,o,w,e,s,t → lo,w,e,s,t  (rule 1: l+o)
  → low,e,s,t               (rule 2: lo+w)
  → lowe,s,t                (rule 3: low+e)
  → lowe,s,t                (rule for "e"+"s" doesn't exist, stop here)
  → ["lowe", "s", "t"]
```

---

## BYTE-LEVEL BPE (GPT-2, LLaMA) — The Modern Standard

**What it is:** Instead of starting BPE from individual characters (there are 150,000+ Unicode characters), start from raw bytes. There are only 256 possible byte values (0–255), so the starting vocabulary is tiny and perfectly complete.

**Analogy:** Instead of learning to read every language's alphabet separately, you learn to read binary. Every text in every language is ultimately made of bytes, so starting from bytes means you can handle literally anything.

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

**WHY:** A larger vocabulary means common multi-character sequences get their own token, so you use fewer total tokens per document. LLaMA 3 quadrupled the vocabulary size specifically to make non-English languages and code more token-efficient.

---

## WORDPIECE (BERT) vs BPE (GPT)

**What it is:** WordPiece is Google's alternative to BPE, used in BERT. Instead of merging the most *frequent* pair, it merges the pair that most *increases the likelihood* of the training corpus under a language model. The difference is subtle but produces different tokenizations.

**Key difference you need to know:** WordPiece uses a `##` prefix on continuation pieces. This prefix tells BERT: "this piece does NOT start a new word — it continues the previous one." BPE has no such prefix.

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

**WHY:** The `##` prefix helps BERT understand word boundaries — useful for tasks like Named Entity Recognition where you need to know where words start and end.

---

## SENTENCEPIECE — LANGUAGE-AGNOSTIC TOKENIZATION

**What it is:** SentencePiece is a tokenizer library (not an algorithm) that works on raw text streams without relying on spaces to define word boundaries. It can use either BPE or a Unigram language model internally.

**Used by:** T5, LLaMA 1/2, ALBERT, XLNet

**Analogy:** Most tokenizers assume words are separated by spaces — that is an English-centric assumption. Chinese, Japanese, Thai, and Arabic do not use spaces. SentencePiece treats the whole text as one continuous stream of characters and finds meaningful pieces without needing spaces.

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
"Hello world" → "▁Hello▁world"   (▁ = space character, encoded as a real token)
               → ["▁Hello", "▁world"]
```

**WHY:** Encoding the space as `▁` is clever — it means the tokenizer can encode "Hello world" and "Helloworld" differently even though both have the same characters. The space is information.

**SentencePiece + Unigram Language Model:**
Instead of merging, start with a large vocabulary and PRUNE tokens.
Train a unigram language model, remove tokens that contribute least to likelihood.
More mathematically principled but slightly more complex.

---

## TIKTOKEN (OpenAI)

**What it is:** Tiktoken is OpenAI's fast BPE tokenizer, written in Rust. It uses the same BPE algorithm as GPT-2, but is 10–100× faster than the HuggingFace Python implementation.

```python
import tiktoken  # import OpenAI's tokenizer library

# Different models use different tokenizers — always match model to encoding
enc = tiktoken.encoding_for_model("gpt-4")
# Alternative: enc = tiktoken.get_encoding("cl100k_base")  # specify encoding by name

text = "Hello, I am learning about tokenization!"
tokens = enc.encode(text)     # convert string to list of integer token IDs
print(tokens)                  # [9906, 11, 358, 1097, 6975, 922, 47058, 0]
print(len(tokens))             # 8 tokens — tells you the length

decoded = enc.decode(tokens)   # convert token IDs back to readable string
print(decoded)                  # "Hello, I am learning about tokenization!"

# CRITICAL PRODUCTION USE: count tokens before sending to API to predict cost
def count_tokens(text, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)   # get the right encoder for the model
    return len(enc.encode(text))               # encode and count the resulting tokens
```

**WHY:** Before making an OpenAI API call, always count tokens. GPT-4 charges per token. A document you think is "short" might tokenize into thousands of tokens and cost dollars per call. `count_tokens()` should be in every production pipeline.

---

## VOCABULARY SIZE — THE ENGINEERING TRADE-OFF

**What it is:** The vocabulary size is how many distinct tokens the model knows. It is a fundamental design choice with direct trade-offs between memory usage, tokenization efficiency, and multilingual capability.

**Analogy:** Think of vocabulary size like the number of keys on a keyboard. A tiny keyboard (26 letters) requires you to type many keystrokes to express ideas. A huge keyboard (one key per common word) is faster to type but the keyboard itself is enormous. Vocabulary size is that same trade-off.

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

**WHY:** The embedding table maps every token ID to a high-dimensional vector. It is literally a giant lookup table. Doubling vocabulary from 32K to 128K quadruples the embedding table size. This is a real engineering cost, but the payoff is fewer tokens per document, which means cheaper inference and more text fits in the context window.

---

## MULTILINGUAL TOKENIZATION PROBLEM

### The Fertility Problem

**What it is:** "Fertility" means the average number of tokens needed to represent one word in a given language. A tokenizer trained mostly on English has low fertility for English but high fertility for other languages — it takes many more tokens to represent the same meaning.

**Analogy:** Imagine a filing system designed for English books. To file a Chinese book, you have to spell out every character phonetically in English letters — it takes 3× more space. The filing cabinet (context window) is the same size, but Chinese books fill it 3× faster.

```
English: "Hello, my name is Claude"     → 6 tokens  (1:1 ratio — efficient)
French:  "Bonjour, je m'appelle Claude" → 8 tokens  (still ok)
Arabic:  "مرحبا، اسمي كلود"              → 14 tokens (2.3x more expensive!)
Thai:    "สวัสดีฉันชื่อคลอด"              → 25 tokens (4x more expensive!)
```

**Implications:**
- Non-English languages cost more API tokens (real money)
- Context window used less efficiently for non-English text
- Model might have less "thinking space" for non-English reasoning

### Solutions
- Larger vocabularies (LLaMA 3: 128K vs 32K — specifically to improve multilingual efficiency)
- Language-specific tokenizers (Qwen adds extra Chinese tokens to its vocabulary)
- SentencePiece (language-agnostic design, no space dependency)

---

## SPECIAL TOKENS — THE CONTROL SIGNALS

**What it is:** Special tokens are reserved tokens that are not part of normal language — they are control codes the model uses to understand structure. They are like punctuation marks for the model's internal "grammar" of conversations.

**Analogy:** In HTML, `<b>` and `</b>` are not displayed text — they are control codes that tell the browser to make text bold. Special tokens are the same thing for LLMs: invisible control codes that say "conversation starts here," "system prompt ends here," "generate your response here."

Every model has special tokens that control behavior:

### Common Special Tokens
| Token | Purpose | Example |
|-------|---------|---------|
| `<\|endoftext\|>` | End of document (GPT-2) | Marks document boundary |
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

**What it is:** A chat template is the specific format in which a chat model expects its input. Different model families use completely different formats. Using the wrong format produces garbage output.

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

**WHY this matters for fine-tuning:**
You MUST use the correct chat template when fine-tuning chat models.
Wrong template = model doesn't understand conversation structure = poor results.

```python
# HuggingFace handles chat templates automatically — always use this
from transformers import AutoTokenizer  # import HuggingFace tokenizer

# Load the tokenizer for a specific model — this also loads the correct chat template
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Build a conversation as a list of message dictionaries
messages = [
    {"role": "system", "content": "You are helpful."},   # system prompt
    {"role": "user",   "content": "What is 2+2?"},       # user message
]

# Apply the model's specific chat template automatically
# tokenize=False means return string (not token IDs) — useful for inspection
# add_generation_prompt=True adds the opening assistant header so the model knows to respond
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

**WHY:** `apply_chat_template` is essential. Each model has its own template stored in its tokenizer config. This function reads that config and formats the conversation correctly. Never hardcode templates manually — they change between model versions.

---

## THE TOKENIZATION PROBLEM FOR MATH & CODE

### Why LLMs Are Bad At Arithmetic

**What it is:** This is a concrete, testable consequence of tokenization — numbers get split mid-digit, so the model literally cannot "see" individual digits. It sees multi-digit chunks instead.

```
"1234 + 5678 = ?"

Tokenization: "1234" → ["12", "34"]    (split mid-number!)
              "5678" → ["56", "78"]

The model can't "see" individual digits — it sees chunks.
Column-by-column addition is impossible when digits are merged.
```

**WHY:** To do addition correctly, you need to line up digits by place value (ones, tens, hundreds). But if "1234" becomes ["12", "34"], the model sees two two-digit chunks, not four single digits. It cannot carry out column arithmetic on those chunks.

**Solutions:**
- Chain-of-Thought: "4+8=12, write 2 carry 1; 3+7+1=11..." — let the model do arithmetic step by step in text
- Reasoning models (o1/R1): trained to decompose into verifiable steps
- Future: token-free models that process raw bytes — solves this permanently

### Code Tokenization

Python code: generally tokenizes well because spaces are preserved as tokens.
```python
"def hello():" → ["def", " hello", "():", ...]    # function keyword, space+name, punctuation
```

Whitespace-sensitive: Python indentation IS meaning, and it is preserved:
```python
"    return x" → ["    ", "return", " x"]  # 4 leading spaces preserved as a single token
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
