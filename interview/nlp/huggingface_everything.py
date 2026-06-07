"""
HuggingFace Transformers — Everything
=======================================
pip install transformers datasets accelerate sentence-transformers
"""

from transformers import (
    pipeline,
    AutoTokenizer, AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    TrainingArguments, Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset, load_dataset
import torch
import numpy as np
import pandas as pd

# ════════════════════════════════════════════
# 1. PIPELINES — Fastest Way to Use Models
# ════════════════════════════════════════════
print("1. PIPELINES")

# Sentiment Analysis
sentiment = pipeline("sentiment-analysis",
                     model="distilbert-base-uncased-finetuned-sst-2-english")
results = sentiment([
    "I love this product! It's amazing!",
    "Terrible experience, would not recommend.",
    "It's okay, nothing special.",
])
for r in results:
    print(f"  {r['label']:8s} ({r['score']:.4f})")
# POSITIVE (0.9999)
# NEGATIVE (0.9997)
# NEGATIVE (0.9174)

# Text Generation
generator = pipeline("text-generation", model="gpt2", max_new_tokens=50)
output = generator("Machine learning is", num_return_sequences=1)
print(f"\nGenerated: {output[0]['generated_text']}")

# Zero-shot Classification (no training needed!)
classifier = pipeline("zero-shot-classification",
                       model="facebook/bart-large-mnli")
result = classifier(
    "Apple just launched a new iPhone with AI features",
    candidate_labels=["technology", "sports", "politics", "health"]
)
print(f"\nZero-shot: {result['labels'][0]} ({result['scores'][0]:.4f})")
# technology (0.9842)

# Named Entity Recognition
ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
entities = ner("Elon Musk founded Tesla in California.")
for e in entities:
    print(f"  {e['word']:15} → {e['entity_group']}")
# Elon Musk       → PER
# Tesla           → ORG
# California      → LOC

# Question Answering
qa = pipeline("question-answering",
              model="deepset/roberta-base-squad2")
result = qa(
    question="Who founded Tesla?",
    context="Tesla was founded by Elon Musk, Martin Eberhard and Marc Tarpenning in 2003."
)
print(f"\nAnswer: {result['answer']} (score: {result['score']:.4f})")
# Answer: Elon Musk, Martin Eberhard and Marc Tarpenning

# Summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn",
                       min_length=30, max_length=100)
long_text = """
Machine learning is a branch of artificial intelligence (AI) and computer science which focuses
on the use of data and algorithms to imitate the way that humans learn, gradually improving its
accuracy. Machine learning is an important component of the growing field of data science.
Through the use of statistical methods, algorithms are trained to make classifications or
predictions, and to uncover key insights in data mining projects.
"""
summary = summarizer(long_text)
print(f"\nSummary: {summary[0]['summary_text']}")

# Translation
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
translated = translator("Hello, how are you?")
print(f"\nTranslated: {translated[0]['translation_text']}")

# ════════════════════════════════════════════
# 2. TOKENIZER
# ════════════════════════════════════════════
print("\n2. TOKENIZER")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Hello, my name is Alice and I love machine learning!"

# Basic tokenization
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
# ['hello', ',', 'my', 'name', 'is', 'alice', 'and', 'i', 'love', 'machine', 'learning', '!']

# Encode to IDs
ids = tokenizer.encode(text)
print(f"IDs: {ids}")
# [101, 7592, 1010, 2026, 2171, ...]  (101=CLS, 102=SEP)

# Full encoding with attention mask
encoded = tokenizer(
    text,
    max_length=64,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)
print(f"input_ids shape:      {encoded['input_ids'].shape}")       # (1, 64)
print(f"attention_mask shape: {encoded['attention_mask'].shape}")  # (1, 64)

# Batch encoding
texts = ["First sentence", "Second sentence", "Third one"]
batch = tokenizer(texts, padding=True, truncation=True,
                  max_length=32, return_tensors="pt")
print(f"Batch shape: {batch['input_ids'].shape}")   # (3, 32)

# Decode back to text
decoded = tokenizer.decode(ids, skip_special_tokens=True)
print(f"Decoded: {decoded}")

# ════════════════════════════════════════════
# 3. GET EMBEDDINGS (BERT)
# ════════════════════════════════════════════
print("\n3. BERT EMBEDDINGS")

tokenizer  = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")
bert_model.eval()

def get_embeddings(texts: list) -> np.ndarray:
    encoded = tokenizer(texts, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**encoded)

    # CLS token embedding = sentence representation
    cls_embeddings = outputs.last_hidden_state[:, 0, :]   # (batch, 768)
    # Mean pooling (often better)
    attention_mask = encoded["attention_mask"].unsqueeze(-1)
    mean_emb = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)

    return mean_emb.numpy()

texts = ["I love machine learning", "I enjoy deep learning", "Football is fun"]
embeddings = get_embeddings(texts)
print(f"Embeddings shape: {embeddings.shape}")  # (3, 768)

# Similarity
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(embeddings)
print(f"Similarity(ML, DL): {sim[0,1]:.4f}")   # high (both AI)
print(f"Similarity(ML, Football): {sim[0,2]:.4f}")  # low

# ════════════════════════════════════════════
# 4. SENTENCE TRANSFORMERS (Best for Similarity)
# ════════════════════════════════════════════
print("\n4. SENTENCE TRANSFORMERS")

from sentence_transformers import SentenceTransformer, util

smodel = SentenceTransformer("all-MiniLM-L6-v2")  # fast, small, good quality

sentences = [
    "Machine learning is a subset of AI",
    "AI includes machine learning techniques",
    "Football is played with a round ball",
    "Soccer is popular worldwide",
]

embeddings = smodel.encode(sentences, convert_to_tensor=True)
print(f"Shape: {embeddings.shape}")   # (4, 384)

# Pairwise similarity
cos_scores = util.cos_sim(embeddings, embeddings)
print(f"\nSimilarity Matrix:")
for i, s1 in enumerate(sentences):
    for j, s2 in enumerate(sentences):
        if i < j:
            print(f"  [{cos_scores[i][j]:.3f}] '{s1[:30]}' ↔ '{s2[:30]}'")

# Semantic search
query   = "deep learning neural networks"
q_emb   = smodel.encode(query, convert_to_tensor=True)
hits    = util.semantic_search(q_emb, embeddings, top_k=2)
for hit in hits[0]:
    print(f"  Score {hit['score']:.3f}: {sentences[hit['corpus_id']]}")

# ════════════════════════════════════════════
# 5. FINE-TUNING (Classification)
# ════════════════════════════════════════════
print("\n5. FINE-TUNING BERT FOR CLASSIFICATION")

model_name = "distilbert-base-uncased"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Prepare dataset
data = {
    "text":  ["I love this", "Terrible product", "Great quality", "Worst ever"],
    "label": [1, 0, 1, 0]
}
dataset = Dataset.from_dict(data)

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True,
                     max_length=128, padding="max_length")

tokenized_ds = dataset.map(tokenize_fn, batched=True)
tokenized_ds = tokenized_ds.train_test_split(test_size=0.25)

training_args = TrainingArguments(
    output_dir          = "./results",
    num_train_epochs    = 3,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 16,
    learning_rate       = 2e-5,
    weight_decay        = 0.01,
    evaluation_strategy = "epoch",
    save_strategy       = "epoch",
    load_best_model_at_end = True,
    logging_dir         = "./logs",
    report_to           = "none",
)

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="weighted")
    }

trainer = Trainer(
    model            = model,
    args             = training_args,
    train_dataset    = tokenized_ds["train"],
    eval_dataset     = tokenized_ds["test"],
    tokenizer        = tokenizer,
    compute_metrics  = compute_metrics,
)

# trainer.train()     # uncomment to actually train
print("Trainer ready — call trainer.train() to start")

# ════════════════════════════════════════════
# 6. INFERENCE PIPELINE
# ════════════════════════════════════════════
print("\n6. INFERENCE")

# After training, inference is simple
def predict(texts: list, model, tokenizer, device="cpu"):
    model.eval()
    encoded = tokenizer(texts, padding=True, truncation=True,
                        max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encoded)
    probs = torch.softmax(outputs.logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)
    return preds.tolist(), probs.tolist()

texts = ["This is great!", "Totally disappointed."]
preds, probs = predict(texts, model, tokenizer)
for t, p, prob in zip(texts, preds, probs):
    print(f"  {t:25s} → {'POS' if p==1 else 'NEG'} ({max(prob):.3f})")

print("\nAll done! ✓")
