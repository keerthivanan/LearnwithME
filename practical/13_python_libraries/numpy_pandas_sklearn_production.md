# 13 — Python & Core ML Libraries

> JD: "Proficiency in Python, NumPy, Pandas, Scikit-learn, and others." Know these for data processing and ML tasks.

---

## 1. Python Essentials for ML

### List Comprehensions & Generators
```python
# List comprehension — fast, Pythonic
squares = [x**2 for x in range(10)]
filtered = [x for x in data if x > 0]

# Generator — memory efficient for large datasets
def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]
```

### Decorators
```python
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time()-start:.2f}s")
        return result
    return wrapper

@timer
def train_epoch(model, dataloader):
    ...
```

### Context Managers
```python
# Used extensively in PyTorch
with torch.no_grad():      # disable gradient computation
    output = model(input)

with torch.cuda.amp.autocast():   # mixed precision
    output = model(input)
```

### Type Hints
```python
from typing import List, Dict, Optional, Tuple
import numpy as np

def embed_texts(texts: List[str], model: str = "bge-large") -> np.ndarray:
    ...
```

### Async/Await (For API Servers)
```python
import asyncio
from fastapi import FastAPI

app = FastAPI()

@app.post("/generate")
async def generate(prompt: str):
    result = await async_llm_call(prompt)
    return {"text": result}
```

---

## 2. NumPy

### What it is
Efficient numerical computation with N-dimensional arrays. Foundation of all ML libraries.

### Core Operations
```python
import numpy as np

# Create arrays
a = np.array([1, 2, 3, 4, 5])
b = np.zeros((3, 4))               # 3x4 zeros
c = np.ones((2, 3))
d = np.random.randn(100, 768)      # normal random, 100×768

# Shape operations
print(d.shape)         # (100, 768)
d.reshape(200, 384)
d.T                    # transpose

# Math operations (vectorized — fast)
a + b                  # element-wise add
a @ b                  # matrix multiply
np.dot(a, b)           # dot product
np.sum(a, axis=0)      # sum along axis

# Indexing and slicing
a[0]                   # first element
a[1:4]                 # slice
d[:, 0]                # all rows, first column
d[d > 0]               # boolean indexing
```

### Broadcasting
```python
a = np.array([[1, 2, 3]])   # shape (1, 3)
b = np.array([[1], [2], [3]])  # shape (3, 1)
a + b   # shape (3, 3) — broadcast
```

### Important for ML
```python
# Cosine similarity
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))   # subtract max for numerical stability
    return e_x / e_x.sum()

# Batch matrix multiply
# (batch, seq, d_model) × (batch, d_model, d_k) = (batch, seq, d_k)
result = np.einsum("bsd,dh->bsh", x, W)
```

---

## 3. Pandas

### What it is
Data manipulation and analysis library. Essential for working with tabular datasets.

### Core Operations
```python
import pandas as pd

# Load data
df = pd.read_csv("dataset.csv")
df = pd.read_json("data.jsonl", lines=True)

# Inspect
df.head(5)
df.info()
df.describe()
df.shape

# Select
df["column"]                    # single column → Series
df[["col1", "col2"]]           # multiple columns → DataFrame
df.iloc[0:5]                    # by index
df.loc[df["label"] == 1]       # by condition

# Modify
df["new_col"] = df["col"].apply(lambda x: x.lower())
df.drop("col", axis=1, inplace=True)
df.rename(columns={"old": "new"}, inplace=True)

# Handle missing data
df.isnull().sum()
df.dropna(subset=["text"])
df.fillna("unknown")

# Group and aggregate
df.groupby("category")["score"].mean()
df.value_counts("label")
```

### For NLP/LLM Datasets
```python
# Load a conversation dataset
df = pd.read_json("conversations.jsonl", lines=True)

# Inspect text lengths
df["text_len"] = df["text"].apply(len)
df["token_count"] = df["text"].apply(lambda x: len(tokenizer.encode(x)))

# Filter by length
df = df[df["token_count"] < 2048]

# Convert to Hugging Face Dataset
from datasets import Dataset
hf_dataset = Dataset.from_pandas(df)
```

---

## 4. Scikit-learn

### What it is
The standard library for classical ML. Useful for evaluation, preprocessing, and baselines.

### Key Uses in LLM Projects

**Train/Test Split**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
```

**Evaluation Metrics**
```python
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average="weighted"))
print(classification_report(y_test, y_pred))
```

**TF-IDF & Sparse Features (for BM25-like retrieval)**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(train_texts)
X_test_tfidf = vectorizer.transform(test_texts)
```

**Baseline Models**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Quick baseline for text classification
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_tfidf, y_train)
print(f"Baseline accuracy: {clf.score(X_test_tfidf, y_test):.3f}")
```

**Cross-Validation**
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, X, y, cv=5, scoring="f1_weighted")
print(f"CV F1: {scores.mean():.3f} ± {scores.std():.3f}")
```

**Clustering (for analyzing embeddings)**
```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# Cluster embeddings
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(embeddings)

# Visualize embeddings (2D)
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)
```

---

## 5. Matplotlib & Seaborn (Visualization)

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Training curves
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Token length distribution
plt.hist(df["token_count"], bins=50, edgecolor="black")
plt.xlabel("Token Count")
plt.title("Dataset Token Length Distribution")
```

---

## 6. Environment Management

### Virtual Environments
```bash
# Create environment
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows

# Conda
conda create -n llm-env python=3.11
conda activate llm-env
```

### Requirements
```bash
# Save
pip freeze > requirements.txt

# Install
pip install -r requirements.txt

# Common LLM packages
pip install torch transformers datasets peft trl accelerate
pip install langchain chromadb sentence-transformers
pip install fastapi uvicorn vllm
pip install numpy pandas scikit-learn matplotlib
pip install wandb mlflow
```

---

## 7. Useful Code Patterns in LLM Projects

### Batch Processing with Progress
```python
from tqdm import tqdm

results = []
for batch in tqdm(dataloader, desc="Processing"):
    with torch.no_grad():
        output = model(**batch)
    results.extend(output.logits.cpu().numpy())
```

### JSON Lines (JSONL) — Standard for LLM Datasets
```python
import json

# Write JSONL
with open("dataset.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")

# Read JSONL
data = []
with open("dataset.jsonl") as f:
    for line in f:
        data.append(json.loads(line.strip()))
```

### Hugging Face Dataset from JSONL
```python
from datasets import load_dataset

dataset = load_dataset("json", data_files={
    "train": "train.jsonl",
    "test": "test.jsonl"
})
```

### GPU Memory Management
```python
import torch
import gc

# Check GPU memory
print(torch.cuda.memory_allocated() / 1024**3, "GB")
print(torch.cuda.memory_reserved() / 1024**3, "GB")

# Free memory
del model
gc.collect()
torch.cuda.empty_cache()
```

### Seed Everything for Reproducibility
```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)
```

---

## 8. Data Processing Pipeline Example

```python
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# 1. Load raw data
df = pd.read_csv("raw_data.csv")

# 2. Clean
df = df.dropna(subset=["instruction", "output"])
df = df[df["instruction"].str.len() > 10]

# 3. Format as instruction-response pairs
def format_example(row):
    return {
        "text": f"### Instruction:\n{row['instruction']}\n\n### Response:\n{row['output']}"
    }
df["text"] = df.apply(format_example, axis=1)

# 4. Tokenize and filter by length
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
df["token_count"] = df["text"].apply(lambda x: len(tokenizer.encode(x)))
df = df[df["token_count"] < 2048]

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
