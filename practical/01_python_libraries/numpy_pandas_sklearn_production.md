
# 01 — Python & Core ML Libraries

> JD: "Proficiency in Python, NumPy, Pandas, Scikit-learn, and others." Know these for data processing and ML tasks.

---

## 1. Python Essentials for ML

### List Comprehensions & Generators

**What it is:** A short way to create a list in one line.

```python
squares = [x**2 for x in range(10)]
# Output: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
# x**2 means x to the power of 2. range(10) gives 0,1,2...9

filtered = [x for x in data if x > 0]
# Only keeps values greater than 0
# Same as: for x in data: if x > 0: result.append(x)
```

**What is a Generator and why use it?**
- A list loads ALL items into memory at once → crashes with 1 million texts
- A generator loads ONE item at a time → never crashes, memory safe

```python
def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]
# yield = give one batch, pause, wait for next call

# Example:
# data = [1,2,3,4,5,6], batch_size=2
# batch 1 → [1,2]
# batch 2 → [3,4]
# batch 3 → [5,6]
# WHY: You can't feed 1 million texts to a model at once.
#      Feed 32 at a time (batch_size=32) → safe
```

---

### Decorators

**What it is:** A decorator wraps a function and adds behavior before/after it — without changing the original function.

```python
import time

def timer(func):          # receives a function as input
    def wrapper(*args, **kwargs):
        start = time.time()          # runs BEFORE original function
        result = func(*args, **kwargs)  # runs original function
        print(f"{func.__name__} took {time.time()-start:.2f}s")  # runs AFTER
        return result
    return wrapper

@timer                    # @timer is shortcut for: train_epoch = timer(train_epoch)
def train_epoch(model, dataloader):
    ...
# Now every time train_epoch() runs, it automatically prints how long it took
# Example output: "train_epoch took 3.24s"
```

**Real ML use:** `@torch.no_grad()` is a decorator. Before your function it turns OFF gradient tracking (saves memory). After it turns it back ON.

---

### Context Managers

**What it is:** `with` block — automatically runs setup before and cleanup after your code.

```python
with torch.no_grad():        # turns OFF gradient tracking before this block
    output = model(input)    # inference runs here
                             # gradients automatically turned back ON after block ends

# WHY: During inference you don't train, so storing gradients wastes memory.
#      no_grad() → 50% less memory, faster speed

with torch.cuda.amp.autocast():   # switches to FP16 (half precision) automatically
    output = model(input)
# WHY: FP16 uses half the memory of FP32. Faster on modern GPUs.
#      autocast() handles the conversion automatically — no manual casting needed
```

---

### Type Hints

**What it is:** Optional labels that tell you what type of data a function expects and returns.

```python
from typing import List, Dict, Optional, Tuple
import numpy as np

def embed_texts(texts: List[str], model: str = "bge-large") -> np.ndarray:
    ...
# texts: List[str]  → expects a list of strings
# model: str        → expects a string, default is "bge-large"
# -> np.ndarray     → returns a NumPy array (the embeddings)

# WHY: Makes code readable. You instantly know what goes in and what comes out.
#      Also catches bugs early — editor warns if you pass wrong type
```

---

### Async/Await (For API Servers)

**What it is:** Async lets your server handle multiple users at the same time without blocking.

```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/generate")
async def generate(prompt: str):
    result = await async_llm_call(prompt)  # pause here, let other users run
    return {"text": result}

# WHY: LLM takes 5 seconds to respond.
#      SYNC → 1 user blocks all others for 5 seconds
#      ASYNC → 1000 users all wait together, server handles all simultaneously
#
# await = "pause this function, go handle other requests, come back when done"
```

---

### `logging` — Production Code Never Uses `print()`

```python
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info("Model loaded successfully")        # INFO: normal events
logger.warning("Sequence too long, truncating") # WARNING: something unexpected
logger.error("GPU out of memory")               # ERROR: something broke

# WHY: print() disappears in production. logging saves to files, adds timestamps,
#      can be turned on/off per level without changing code
```

---

### `functools.lru_cache` — Cache Repeated Computations

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(text: str):
    return model.encode(text)

# First call: actually runs model.encode() → slow
get_embedding("hello world")

# Second call with same text: returns cached result instantly → fast
get_embedding("hello world")

# WHY: If 1000 users ask the same question, you compute embedding only ONCE
#      Cache stores up to 1000 recent results (maxsize=1000)
```

---

### `collections.Counter` — Count Token Frequencies

```python
from collections import Counter

tokens = ["apple", "banana", "apple", "cherry", "apple", "banana"]
counts = Counter(tokens)
# Counter({'apple': 3, 'banana': 2, 'cherry': 1})

print(counts.most_common(2))   # [('apple', 3), ('banana', 2)]

# Real use: analyze vocabulary in your training data
all_tokens = [token for text in dataset for token in tokenizer.encode(text)]
freq = Counter(all_tokens)
print(f"Vocab size: {len(freq)}")
print(f"Top 10 tokens: {freq.most_common(10)}")
```

---

### `re` (Regular Expressions) — Text Cleaning for NLP

```python
import re

text = "  Hello,   World!!! Visit https://example.com   "

# Remove URLs
text = re.sub(r'http\S+', '', text)

# Remove extra whitespace
text = re.sub(r'\s+', ' ', text).strip()

# Remove special characters, keep letters and spaces
text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

# WHY: Raw text is messy. Before training, you must clean it.
#      URLs, HTML tags, extra spaces all hurt model quality.
```

---

### `dataclasses` — HuggingFace Uses This Everywhere

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    model_name: str = "meta-llama/Llama-3.1-8B"
    learning_rate: float = 2e-4
    batch_size: int = 4
    max_seq_length: int = 2048
    num_epochs: int = 3
    output_dir: Optional[str] = None

config = TrainingConfig(learning_rate=1e-4, batch_size=8)
print(config.learning_rate)   # 0.0001

# WHY: HuggingFace TrainingArguments is a dataclass.
#      @dataclass auto-generates __init__, __repr__ — no boilerplate needed
```

---

### `pathlib.Path` — Modern File Handling

```python
from pathlib import Path

# Better than os.path — cleaner syntax
output_dir = Path("models") / "llama-finetuned" / "checkpoint-500"
output_dir.mkdir(parents=True, exist_ok=True)  # creates all folders

# Check if file exists
if Path("dataset.jsonl").exists():
    print("Dataset ready")

# List all .jsonl files in a folder
files = list(Path("data").glob("*.jsonl"))

# WHY: os.path is old and verbose. pathlib is the modern standard.
#      HuggingFace Trainer uses Path internally.
```

---

## 2. NumPy

### What it is
NumPy = fast math on arrays. Every ML library (PyTorch, TensorFlow) is built on top of it.
Think of it as Excel for Python — but 1000x faster and works with millions of numbers.

### Core Operations

```python
import numpy as np

# --- CREATING ARRAYS ---
a = np.array([1, 2, 3, 4, 5])      # simple 1D array, shape: (5,)
b = np.zeros((3, 4))               # 3 rows, 4 cols, all zeros → used to initialize weights
c = np.ones((2, 3))                # 2 rows, 3 cols, all ones
d = np.random.randn(100, 768)      # 100 embeddings, each 768 dims → typical embedding shape

# --- SHAPE (most important concept) ---
print(d.shape)       # (100, 768) → 100 rows, 768 columns
d.reshape(200, 384)  # same data, different shape (100*768 = 200*384)
d.T                  # transpose → flips rows and columns, shape becomes (768, 100)

# --- MATH (vectorized = no for loop, very fast) ---
a + b                # adds element by element
a @ b                # matrix multiplication (used in attention: Q @ K.T)
np.dot(a, b)         # dot product (same as @)
np.sum(a, axis=0)    # sum down each column
np.mean(a, axis=1)   # average across each row

# --- INDEXING ---
a[0]                 # first item
a[1:4]               # items at index 1,2,3 (not 4)
d[:, 0]              # ALL rows, only column 0 → first feature of every embedding
d[d > 0]             # only keep values greater than 0 (boolean filter)
```

### Broadcasting — Automatic Shape Matching

**Problem:** You want to add two arrays with different shapes.
**Solution:** NumPy automatically stretches the smaller one to match.

```python
a = np.array([[1, 2, 3]])      # shape (1, 3) → 1 row, 3 cols
b = np.array([[1], [2], [3]])  # shape (3, 1) → 3 rows, 1 col
result = a + b                 # shape (3, 3) — NumPy stretches both to match

# Result:
# [[1+1, 2+1, 3+1],   = [[2, 3, 4],
#  [1+2, 2+2, 3+2],      [3, 4, 5],
#  [1+3, 2+3, 3+3]]      [4, 5, 6]]

# REAL USE: subtract mean from 1000 embeddings
docs = np.random.randn(1000, 768)   # shape (1000, 768)
mean = np.mean(docs, axis=0)        # shape (768,)
docs = docs - mean                  # NumPy broadcasts mean to (1000,768) automatically
# No for loop needed! This runs 100x faster than looping
```

### Combining Arrays — You Do This Constantly

```python
# np.concatenate — join arrays along existing axis
a = np.random.randn(100, 768)   # 100 embeddings
b = np.random.randn(200, 768)   # 200 more embeddings
combined = np.concatenate([a, b], axis=0)   # shape: (300, 768)
# WHY: You process data in batches, then concatenate all results at the end

# np.stack — join arrays along NEW axis
batch1 = np.random.randn(768)   # one embedding
batch2 = np.random.randn(768)   # another embedding
stacked = np.stack([batch1, batch2], axis=0)  # shape: (2, 768)
# WHY: Converting list of embeddings into a matrix for batch processing

# np.vstack / np.hstack — shorthand
matrix = np.vstack([a, b])   # same as concatenate axis=0 (vertical stack)
```

### Top-K — Used in Vector Search and Sampling

```python
scores = np.array([0.2, 0.9, 0.4, 0.7, 0.1])

# Get indices of top 3 highest scores
top3_indices = np.argsort(scores)[-3:][::-1]
# argsort → [4,0,2,3,1] (sorted low to high)
# [-3:] → take last 3 (highest): [2,3,1]
# [::-1] → reverse to get high→low: [1,3,2]
print(top3_indices)   # [1, 3, 2] — indices of top 3
print(scores[top3_indices])  # [0.9, 0.7, 0.4] — their scores

# WHY: This is literally how vector databases return top-K results
```

### NumPy Dtypes — Critical for Memory Calculation

```python
a_f32 = np.random.randn(1000, 768).astype(np.float32)  # 4 bytes per number
a_f16 = np.random.randn(1000, 768).astype(np.float16)  # 2 bytes per number
a_i8  = np.random.randn(1000, 768).astype(np.int8)     # 1 byte per number

print(a_f32.nbytes / 1e6, "MB")   # 3.07 MB
print(a_f16.nbytes / 1e6, "MB")   # 1.53 MB (half!)
print(a_i8.nbytes  / 1e6, "MB")   # 0.76 MB (quarter!)

# WHY THIS MATTERS:
# 7B model parameters × 4 bytes (FP32) = 28GB → won't fit on most GPUs
# 7B model parameters × 2 bytes (FP16) = 14GB → fits on A100 (80GB)
# 7B model parameters × 0.5 bytes (INT4) = 3.5GB → fits on laptop GPU!
# This is why quantization matters so much in production
```

### np.percentile — Find Right max_seq_length

```python
token_counts = np.array([128, 256, 512, 1024, 2048, 300, 450, 600])

print(np.percentile(token_counts, 50))   # median: 50% are below this
print(np.percentile(token_counts, 95))   # 95th percentile
print(np.percentile(token_counts, 99))   # 99th percentile

# WHY: Set max_seq_length at 95th percentile of your dataset
# If 95% of texts are < 512 tokens → use max_seq_length=512
# Truncating top 5% saves massive memory vs padding everything to 2048
```

### Most Important ML Functions

```python
# --- COSINE SIMILARITY ---
# Used in vector search: find documents most similar to query
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
# np.linalg.norm = length of vector
# dot product of two unit vectors = cosine similarity
# Output range: -1 (opposite) to 1 (identical)

# --- SOFTMAX ---
# Converts raw scores (logits) into probabilities that sum to 1
# Used in: attention weights, output token probabilities
def softmax(x):
    e_x = np.exp(x - np.max(x))   # subtract max FIRST → prevents infinity (nan safety)
    return e_x / e_x.sum()
# Example: [3.0, 1.0, 0.5] → [0.70, 0.19, 0.11] (probabilities)

# --- EINSUM (batch matrix multiply) ---
# Used in attention: multiply Q,K,V matrices across batch and heads
result = np.einsum("bsd,dh->bsh", x, W)
# b=batch, s=sequence, d=model_dim, h=head_dim
# For every batch(b) and position(s): multiply x[b,s,:] with W[:,h]
# Output shape: (batch, seq, head_dim)
```

---

## 3. Pandas

### What it is
Pandas = spreadsheet in Python. You load datasets, clean them, filter them, and prepare them for training.
Think of it as Excel but in code — every row is a sample, every column is a feature.

### Core Operations

```python
import pandas as pd

# --- LOAD DATA ---
df = pd.read_csv("dataset.csv")               # load CSV file
df = pd.read_json("data.jsonl", lines=True)   # load JSONL file (common for LLM datasets)
# df is a DataFrame — like a table with rows and columns

# --- INSPECT (always do this first) ---
df.head(5)       # show first 5 rows → check what data looks like
df.info()        # show column names, types, missing values count
df.describe()    # show min, max, mean, std for numeric columns
df.shape         # (num_rows, num_columns) → e.g. (50000, 3)

# --- SELECT DATA ---
df["text"]               # get ONE column → returns a Series (list of values)
df[["text", "label"]]    # get MULTIPLE columns → returns a DataFrame
df.iloc[0:5]             # get rows 0,1,2,3,4 by position
df.loc[df["label"] == 1] # get all rows where label equals 1

# --- MODIFY DATA ---
df["new_col"] = df["col"].apply(lambda x: x.lower())
# apply() runs a function on every row — here: converts text to lowercase
# lambda x: x.lower() means: for each value x, return x.lower()

df.drop("col", axis=1, inplace=True)          # delete a column
df.rename(columns={"old": "new"}, inplace=True) # rename a column

# --- HANDLE MISSING DATA ---
df.isnull().sum()           # count missing values per column
df.dropna(subset=["text"])  # remove rows where "text" column is empty
df.fillna("unknown")        # replace all missing values with "unknown"

# --- GROUP AND COUNT ---
df.groupby("category")["score"].mean()  # average score per category
df.value_counts("label")                # how many rows per label value
```

### Critical Operations You'll Use in Every Project

```python
# --- DEDUPLICATION — remove duplicate rows ---
df = df.drop_duplicates(subset=["text"])
# WHY: Training on duplicate data → model memorizes instead of learning
#      Common issue with scraped web data

# --- SHUFFLE — randomize order before training ---
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# frac=1 → keep all rows (100%), just shuffle them
# reset_index → fix row numbers after shuffle
# WHY: If data is sorted by category, model sees all category A then B → bad training

# --- MERGE DATASETS — combine multiple data sources ---
df1 = pd.read_json("dataset1.jsonl", lines=True)
df2 = pd.read_json("dataset2.jsonl", lines=True)
combined = pd.concat([df1, df2], ignore_index=True)
# ignore_index=True → renumber rows from 0 after merging

# --- CONVERT TO LIST OF DICTS — for custom processing ---
records = df.to_dict("records")
# [{"text": "...", "label": 1}, {"text": "...", "label": 0}, ...]
# WHY: HuggingFace Dataset.from_list() takes this format

# --- STRING OPERATIONS ---
df["text"] = df["text"].str.lower()          # lowercase all text
df["text"] = df["text"].str.strip()          # remove leading/trailing spaces
df["text"] = df["text"].str.replace("\n", " ")  # replace newlines
df = df[df["text"].str.len() > 20]           # filter short texts
```

### For NLP/LLM Datasets — Real Production Use

```python
# Load a conversation dataset
df = pd.read_json("conversations.jsonl", lines=True)

# Add a column: count characters in each text
df["text_len"] = df["text"].apply(len)
# apply(len) runs len() on every row → gives character count

# Add a column: count tokens (what the model actually sees)
df["token_count"] = df["text"].apply(lambda x: len(tokenizer.encode(x)))
# tokenizer.encode() converts text → token IDs → count them

# Filter: remove texts that are too long for the model
df = df[df["token_count"] < 2048]
# Keep only rows where token_count is less than 2048
# WHY: LLaMA has 2048 context window — longer texts get cut off anyway

# Convert to HuggingFace Dataset format (needed for training)
from datasets import Dataset
hf_dataset = Dataset.from_pandas(df)
# Now you can use it with HuggingFace Trainer
```

---

## 4. Scikit-learn

### What it is
Scikit-learn = classical ML toolkit. You don't use it to build LLMs, but you use it every day for:
- Splitting data into train/test
- Measuring model performance (accuracy, F1)
- Building quick baselines before training a big model

### Key Uses in LLM Projects

**Train/Test Split — Always do this before training**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels,
    test_size=0.2,        # 20% goes to test, 80% to train
    random_state=42,      # same random seed = same split every time (reproducible)
    stratify=labels       # keep same % of each label in both train and test
)
# WHY stratify: if 90% of data is class A and 10% class B,
#               without stratify your test might have 0% class B
```

**Evaluation Metrics — How good is your model?**
```python
from sklearn.metrics import accuracy_score, f1_score, classification_report

y_pred = model.predict(X_test)   # model makes predictions on test data

print(accuracy_score(y_test, y_pred))
# accuracy = correct predictions / total predictions
# PROBLEM: if 95% data is class A, model that always predicts A gets 95% accuracy
#          but it never learns class B! → use F1 instead

print(f1_score(y_test, y_pred, average="weighted"))
# F1 = balance of precision and recall
# weighted = accounts for class imbalance
# Always use F1 for imbalanced datasets (common in NLP)

print(classification_report(y_test, y_pred))
# Shows precision, recall, F1 for EACH class separately
```

**TF-IDF — Simple text → numbers (before LLM era)**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
# max_features=10000 → keep only top 10000 most important words
# ngram_range=(1,2)  → use single words AND pairs of words

X_train_tfidf = vectorizer.fit_transform(train_texts)
# fit = learn vocabulary from training data
# transform = convert texts to numbers

X_test_tfidf = vectorizer.transform(test_texts)
# transform ONLY (never fit on test data — that's data leakage)
# WHY: TF-IDF is the backbone of BM25 (keyword search in RAG)
```

**Baseline Models — Always build a simple model first**
```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_tfidf, y_train)       # train on TF-IDF features
print(f"Baseline: {clf.score(X_test_tfidf, y_test):.3f}")

# WHY: Before training a 7B LLM for 3 days, check if a simple model gets 95%.
#      If logistic regression gets 95%, you don't need a big LLM.
#      If it gets 60%, you know you need something more powerful.
```

**Cross-Validation — More reliable than single train/test split**
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, X, y, cv=5, scoring="f1_weighted")
print(f"CV F1: {scores.mean():.3f} ± {scores.std():.3f}")
# cv=5 → split data into 5 parts, train on 4 test on 1, repeat 5 times
# scores.mean() → average F1 across all 5 runs
# scores.std()  → how consistent the model is (low std = stable)
# WHY: Single split can get lucky/unlucky. CV gives honest performance estimate
```

**Clustering — Group similar embeddings together**
```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(embeddings)
# Groups your 1000 embeddings into 5 clusters
# WHY: Used in RAPTOR (RAG) to cluster document chunks before summarizing

pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)
# Compresses 768-dim embeddings → 2 dimensions for plotting
# WHY: You can't visualize 768 dimensions. PCA lets you see clusters in 2D
```

---

## 5. Matplotlib — Visualize Training

```python
import matplotlib.pyplot as plt

# Plot training loss curve — shows if model is learning or overfitting
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.show()
# If train loss goes down but val loss goes up → overfitting
# If both go down → model is learning correctly

# Plot token length distribution — find right max_length for your dataset
plt.hist(df["token_count"], bins=50)
plt.xlabel("Token Count")
plt.title("How long are your texts?")
plt.show()
# Look at the 95th percentile → set max_length there
# If 95% of texts are < 512 tokens, use max_length=512 (saves memory)
```

---

## 6. Environment Setup — First Thing on Any New Machine

```bash
# Create isolated environment (so packages don't conflict)
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows

# Install everything you need for LLM work
pip install torch transformers datasets peft trl accelerate
pip install langchain chromadb sentence-transformers
pip install fastapi uvicorn vllm
pip install numpy pandas scikit-learn matplotlib
pip install wandb mlflow

# Save your environment so others can reproduce it
pip freeze > requirements.txt

# Someone else installs your exact same environment
pip install -r requirements.txt
```

---

## 7. Production Code Patterns You'll Use Every Day

### Batch Processing with Progress Bar
```python
from tqdm import tqdm

results = []
for batch in tqdm(dataloader, desc="Processing"):
    # tqdm shows a progress bar: Processing: 45%|████▌     | 450/1000
    with torch.no_grad():           # no gradients during inference
        output = model(**batch)     # ** unpacks batch dict into model
    results.extend(output.logits.cpu().numpy())
    # .cpu() → move from GPU to CPU memory
    # .numpy() → convert PyTorch tensor to NumPy array
```

### JSONL — Standard Format for LLM Training Data
```python
import json

# Each line is ONE complete JSON object
# {"instruction": "...", "output": "..."}
# {"instruction": "...", "output": "..."}
# WHY JSONL: can stream line by line without loading entire file

# Write JSONL
with open("dataset.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")  # one JSON per line

# Read JSONL
data = []
with open("dataset.jsonl") as f:
    for line in f:
        data.append(json.loads(line.strip()))  # parse each line as JSON
```

### GPU Memory Management — Critical for LLM Work
```python
import torch, gc

# Always check before loading a model
print(torch.cuda.memory_allocated() / 1024**3, "GB")  # currently used
print(torch.cuda.memory_reserved() / 1024**3, "GB")   # reserved by PyTorch

# Free GPU memory after you're done with a model
del model               # delete Python reference
gc.collect()            # Python garbage collector
torch.cuda.empty_cache() # release GPU cache back to OS
# WHY: If you load a second model without freeing first → OOM (Out of Memory) crash

# Rule of thumb for memory needed:
# 7B model in FP16  = ~14GB  (fits on 1x A100 80GB)
# 70B model in FP16 = ~140GB (needs 2x A100 80GB)
# 7B model in INT4  = ~4GB   (fits on consumer GPU!)
```

### Set Seed — Always Do This for Reproducible Results
```python
import random, numpy as np, torch

def set_seed(seed=42):
    random.seed(seed)              # Python random
    np.random.seed(seed)           # NumPy random
    torch.manual_seed(seed)        # PyTorch CPU
    torch.cuda.manual_seed_all(seed) # PyTorch GPU
    torch.backends.cudnn.deterministic = True  # makes CUDA ops deterministic

set_seed(42)
# WHY: Without this, every run gives different results.
#      With this, you can reproduce the exact same model every time.
#      42 is convention — any number works
```

---

## 8. Full Data Processing Pipeline — Real Production Example

This is what you do BEFORE training a model. Every step explained:

```python
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# STEP 1: Load raw data
df = pd.read_csv("raw_data.csv")
print(df.shape)   # check how many rows you have

# STEP 2: Clean — remove bad rows
df = df.dropna(subset=["instruction", "output"])
# Remove rows where instruction OR output is empty (can't train on empty data)

df = df[df["instruction"].str.len() > 10]
# Remove very short instructions (less than 10 characters = probably garbage)

# STEP 3: Format — LLM needs specific input format
def format_example(row):
    return {
        "text": f"### Instruction:\n{row['instruction']}\n\n### Response:\n{row['output']}"
    }
df["text"] = df.apply(format_example, axis=1)
# apply() runs format_example on every row
# Creates a "text" column with: "### Instruction:\n...\n\n### Response:\n..."
# WHY this format: LLM learns to complete the Response part when it sees Instruction

# STEP 4: Count tokens and filter by length
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
df["token_count"] = df["text"].apply(lambda x: len(tokenizer.encode(x)))
# tokenizer.encode("hello world") → [101, 7592, 2088] → len = 3 tokens
df = df[df["token_count"] < 2048]
# Remove texts longer than 2048 tokens (model's context window limit)

# 5. Convert to HF Dataset
dataset = Dataset.from_pandas(df[["text"]])
dataset = dataset.train_test_split(test_size=0.1, seed=42)
print(dataset)
```

---

## 9. Interview Questions — Python & Libraries

**Q: What is the difference between `.apply()` and vectorized operations in Pandas?**
> `.apply()` runs a Python function row by row (slow for large datasets). Vectorized operations use NumPy under the hood and operate on entire columns at once (much faster). Prefer `df["col"] * 2` over `df["col"].apply(lambda x: x * 2)` whenever possible.

**Q: How do you handle GPU memory issues in PyTorch?**
> Use `torch.no_grad()` during inference, delete tensors when done, call `torch.cuda.empty_cache()`, use gradient checkpointing for training, switch to FP16/BF16 mixed precision, reduce batch size and use gradient accumulation.

**Q: What is broadcasting in NumPy?**
> Automatically expanding arrays of different shapes to compatible shapes for element-wise operations. A (3,) array and (4,3) array can be added — the (3,) is broadcast to (4,3). Enables efficient vectorized code without explicit loops.

**Q: Why is Scikit-learn still relevant when you have deep learning?**
> Scikit-learn is essential for: quick baseline models, evaluation metrics, data splitting, preprocessing (TF-IDF, scaling), clustering for analysis, and cross-validation. It's not used for the main LLM, but for surrounding infrastructure, evaluation, and classical ML components.

---

## Quick Reference Cheat Sheet

```
NumPy:       Vectorized math on arrays, foundation of all ML
Pandas:      Tabular data manipulation, dataset preprocessing  
Scikit-learn: Metrics, train/test split, baselines, clustering
Matplotlib:  Training curves, data visualization
tqdm:        Progress bars for training loops
JSONL:       Standard format for LLM training data
set_seed():  Always set seeds for reproducibility
torch.no_grad(): Disable gradient for inference (memory + speed)
```
