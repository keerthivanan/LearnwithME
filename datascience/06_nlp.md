# Natural Language Processing (NLP)

---

## 1. Text Preprocessing Pipeline

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess(text):
    # 1. Lowercase
    text = text.lower()
    # 2. Remove special chars / numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # 3. Tokenize (split into words)
    words = text.split()
    # 4. Remove stopwords (the, is, at, ...)
    stop = set(stopwords.words('english'))
    words = [w for w in words if w not in stop]
    # 5. Lemmatize (running → run, better → good)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

clean = preprocess("I am running to the store quickly!")
# → "running store quickly"
```

---

## 2. Text Representation

### Bag of Words (BoW)
```
Represent text as word frequency counts.
"I love cats and cats love me" → {I:1, love:2, cats:2, and:1, me:1}

Ignores word order, grammar. Simple but works.
```

### TF-IDF (Most Used Classic Method)
```
TF  = term frequency         (how often word appears in doc)
IDF = inverse document freq  (log(N/df) — rare words get higher weight)
TF-IDF = TF × IDF

"the" has high TF but low IDF (appears in all docs) → low score
"quantum" has lower TF but high IDF → high score

Gives higher weight to words that are important in this doc but rare overall.
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=10000,  # top 10K words
    ngram_range=(1, 2),  # unigrams + bigrams
    min_df=5,            # ignore words in < 5 docs
    max_df=0.95,         # ignore words in > 95% of docs
    stop_words='english'
)

X_train_tfidf = vectorizer.fit_transform(train_texts)
X_test_tfidf  = vectorizer.transform(test_texts)

# Use with any model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
```

### Word Embeddings (Word2Vec, GloVe)
```
Represent words as dense vectors (e.g., 300 dimensions).
Similar words have similar vectors.

king - man + woman ≈ queen  ← famous example

Word2Vec learns embeddings from word co-occurrence in large text corpus.
GloVe: similar but uses global statistics.
FastText: subword embeddings (handles out-of-vocabulary words).
```

```python
from gensim.models import Word2Vec

# Train on your corpus
sentences = [["I", "love", "data"], ["data", "science", "is", "great"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# Get word vector
model.wv['data']          # 100-dim vector
model.wv.most_similar('data', topn=5)  # most similar words
model.wv.similarity('king', 'queen')   # cosine similarity

# Use pretrained (better)
import gensim.downloader
vectors = gensim.downloader.load('word2vec-google-news-300')
```

---

## 3. Text Classification Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Full pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ('clf', LogisticRegression(C=1.0, max_iter=1000))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## 4. Sentiment Analysis

### Using pretrained VADER (simple, fast)
```python
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
score = sia.polarity_scores("I love this product! It's amazing.")
# {'neg': 0.0, 'neu': 0.25, 'pos': 0.75, 'compound': 0.8316}
# compound: -1 (most negative) to +1 (most positive)
```

### Using BERT (best quality)
```python
from transformers import pipeline

sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
result = sentiment("I love this product!")
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

---

## 5. Named Entity Recognition (NER)

```python
import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("Apple was founded by Steve Jobs in California.")
for ent in doc.ents:
    print(ent.text, ent.label_)
# Apple   ORG
# Steve Jobs  PERSON
# California  GPE

# Using transformers
ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
results = ner("Apple was founded by Steve Jobs")
```

---

## 6. BERT & Transformers (Modern NLP)

### What is BERT?
- **B**idirectional **E**ncoder **R**epresentations from **T**ransformers
- Pretrained on massive text (Wikipedia + BookCorpus)
- Bidirectional: reads sentence left-to-right AND right-to-left simultaneously
- **Fine-tune** on your specific task with much less data

### Key BERT variants
| Model | Size | Use |
|-------|------|-----|
| BERT-base | 110M params | General NLP tasks |
| BERT-large | 340M params | Better accuracy, slower |
| DistilBERT | 66M params | 60% smaller, 97% performance |
| RoBERTa | 125M params | Better training than BERT |
| ALBERT | 12M params | Very small, faster |
| Sentence-BERT | - | Sentence similarity, embeddings |

### Fine-tuning BERT
```python
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    TrainingArguments, Trainer
)
from datasets import Dataset
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(examples):
    return tokenizer(examples['text'], truncation=True,
                     max_length=128, padding='max_length')

# Prepare dataset
dataset = Dataset.from_pandas(df[['text', 'label']])
dataset = dataset.map(tokenize, batched=True)
train_ds, val_ds = dataset.train_test_split(test_size=0.2).values()

# Load model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2
)

# Training
args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

trainer.train()
```

### Sentence Embeddings (SentenceTransformers)
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # fast, good quality

sentences = ["I love machine learning", "AI is amazing", "Football is fun"]
embeddings = model.encode(sentences)   # shape (3, 384)

# Similarity between sentences
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(embeddings)
```

---

## 7. Text Similarity

```python
# Cosine similarity (most used for text)
from sklearn.metrics.pairwise import cosine_similarity

# TF-IDF similarity
vec = TfidfVectorizer()
tfidf = vec.fit_transform(documents)
similarity = cosine_similarity(tfidf[0:1], tfidf)

# Semantic similarity (using embeddings)
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
emb1 = model.encode("I love cats")
emb2 = model.encode("I adore felines")
score = util.cos_sim(emb1, emb2)  # 0.85 (high similarity!)
```

---

## 8. Information Retrieval / RAG

```python
# Build a semantic search system
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# Index your documents
documents = ["Doc 1 text...", "Doc 2 text...", "Doc 3 text..."]
embeddings = model.encode(documents, convert_to_numpy=True)

# Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Search
query = "What is machine learning?"
query_emb = model.encode([query], convert_to_numpy=True)
D, I = index.search(query_emb, k=3)  # top 3 results
for i in I[0]:
    print(documents[i])
```

---

## 9. NLP Quick Reference

| Task | Approach |
|------|---------|
| Text classification | TF-IDF + LR (baseline) → BERT (best) |
| Sentiment analysis | VADER (fast) → DistilBERT (accurate) |
| NER | spaCy (fast) → BERT-NER (accurate) |
| Text similarity | TF-IDF cosine → Sentence-BERT |
| Question answering | BERT QA / RoBERTa |
| Summarization | T5, BART, Pegasus |
| Translation | Helsinki-NLP models |
| Generation | GPT-2 / GPT-3/4 |
| Semantic search | FAISS + Sentence-BERT |
