"""
HuggingFace Transformers — Definitions + Code + Outputs
=========================================================
pip install transformers datasets accelerate sentence-transformers torch
"""

from transformers import (
    pipeline,
    AutoTokenizer, AutoModel,
    AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
)
try:
    from datasets import Dataset   # pip install datasets
except ImportError:
    Dataset = None
import torch
import numpy as np


# ══════════════════════════════════════════════════════
# 1. PIPELINES — Fastest Way to Use Pre-trained Models
# ══════════════════════════════════════════════════════
# WHAT IS HUGGINGFACE?
#   → A platform + library with thousands of pre-trained NLP models
#   → Pre-trained = already trained on huge datasets (Wikipedia, web, books)
#   → You can use them directly OR fine-tune on your own data
#
# WHAT IS A PIPELINE?
#   → The EASIEST way to use a HuggingFace model — one line of code
#   → Handles: tokenization → model inference → post-processing
#   → Available tasks: sentiment-analysis, text-generation, zero-shot-classification,
#     named-entity-recognition, question-answering, summarization, translation
#
# WHAT IS A PRE-TRAINED MODEL?
#   → A model already trained on billions of tokens
#   → Understands language structure, grammar, semantics
#   → You get all this knowledge "for free" — no training needed for basic use

print("=" * 55)
print("1. PIPELINES")
print("=" * 55)

# SENTIMENT ANALYSIS — positive or negative?
sentiment = pipeline("sentiment-analysis",
                     model="distilbert-base-uncased-finetuned-sst-2-english")
results = sentiment([
    "I love this product! It's amazing!",
    "Terrible experience, would not recommend.",
    "It's okay, nothing special.",
])
for r in results:
    print(f"  {r['label']:8s} (confidence: {r['score']:.4f})")
# POSITIVE (0.9999)
# NEGATIVE (0.9997)
# NEGATIVE (0.9174)

# TEXT GENERATION — complete a sentence
generator = pipeline("text-generation", model="gpt2", max_new_tokens=30)
output = generator("Machine learning is", num_return_sequences=1)
print(f"\nGenerated: {output[0]['generated_text']}")

# ZERO-SHOT CLASSIFICATION — classify into ANY labels, no training needed!
# "Zero-shot" = model has never seen these exact labels during training
# Uses Natural Language Inference (NLI) to check if text "entails" a label
classifier = pipeline("zero-shot-classification",
                       model="facebook/bart-large-mnli")
result = classifier(
    "Apple just launched a new iPhone with AI features",
    candidate_labels=["technology", "sports", "politics", "health"]
)
print(f"\nZero-shot top label: {result['labels'][0]} ({result['scores'][0]:.4f})")
# technology (0.9842)

# NAMED ENTITY RECOGNITION — extract people, orgs, locations
ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
entities = ner("Elon Musk founded Tesla in California.")
for e in entities:
    print(f"  {e['word']:15} → {e['entity_group']}")
# Elon Musk       → PER
# Tesla           → ORG
# California      → LOC

# QUESTION ANSWERING — extract answer from a context passage
qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
result = qa(
    question="Who founded Tesla?",
    context="Tesla was founded by Elon Musk, Martin Eberhard and Marc Tarpenning in 2003."
)
print(f"\nQA Answer: {result['answer']} (score: {result['score']:.4f})")
# Answer: Elon Musk, Martin Eberhard and Marc Tarpenning

# SUMMARIZATION — condense long text to short summary
summarizer = pipeline("summarization", model="facebook/bart-large-cnn",
                       min_length=20, max_length=80)
long_text = """Machine learning is a branch of artificial intelligence which focuses
on the use of data and algorithms to imitate the way humans learn, gradually improving
accuracy. Through statistical methods, algorithms are trained to make classifications
or predictions, and to uncover key insights in data mining projects."""
summary = summarizer(long_text)
print(f"\nSummary: {summary[0]['summary_text']}")

# TRANSLATION
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
translated = translator("Hello, how are you?")
print(f"\nTranslated to French: {translated[0]['translation_text']}")


# ══════════════════════════════════════════════════════
# 2. TOKENIZER
# ══════════════════════════════════════════════════════
# WHAT IS A TOKENIZER?
#   → Converts raw text into numbers (token IDs) the model can process
#   → NOT just splitting by spaces — handles subwords!
#   → "machine" → ["machine"] | "unbelievable" → ["un", "##believable"]
#   → Subword tokenization: rare words split into known pieces
#
# SPECIAL TOKENS:
#   → [CLS] (101) : Classification token — added at START. Its embedding
#                   represents the WHOLE sentence (used for classification)
#   → [SEP] (102) : Separator token — marks END of sentence or boundary
#   → [PAD] (0)   : Padding — makes all sequences the same length in a batch
#
# attention_mask:
#   → 1 = real token (attend to this)
#   → 0 = padding (ignore this)
#   → Tells model which positions are real vs padded

print("\n" + "=" * 55)
print("2. TOKENIZER")
print("=" * 55)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text      = "Hello, my name is Alice and I love machine learning!"

# Tokenize — split into subword tokens
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
# ['hello', ',', 'my', 'name', 'is', 'alice', 'and', 'i', 'love', 'machine', 'learning', '!']

# Encode — convert tokens to integer IDs
ids = tokenizer.encode(text)
print(f"IDs:    {ids}")
# [101, 7592, 1010, 2026, 2171, ...]  (101=[CLS], 102=[SEP])

# Full encoding with padding + attention mask (ready for model input)
encoded = tokenizer(
    text,
    max_length=64,
    padding="max_length",   # pad shorter sequences to max_length
    truncation=True,         # cut longer sequences to max_length
    return_tensors="pt"      # return PyTorch tensors
)
print(f"input_ids shape:      {encoded['input_ids'].shape}")       # (1, 64)
print(f"attention_mask shape: {encoded['attention_mask'].shape}")  # (1, 64)

# Batch encoding — process multiple texts at once
texts = ["First sentence", "Second sentence here", "Third"]
batch = tokenizer(texts, padding=True, truncation=True,
                  max_length=32, return_tensors="pt")
print(f"Batch shape: {batch['input_ids'].shape}")   # (3, 32)

# Decode — convert IDs back to text
decoded = tokenizer.decode(ids, skip_special_tokens=True)
print(f"Decoded: {decoded}")   # Hello, my name is Alice and I love machine learning!


# ══════════════════════════════════════════════════════
# 3. BERT EMBEDDINGS
# ══════════════════════════════════════════════════════
# WHAT ARE EMBEDDINGS?
#   → Dense numerical vectors that represent text in a high-dimensional space
#   → Words/sentences with similar meaning → vectors that are CLOSE together
#   → BERT produces 768-dimensional vectors for each token and the whole sentence
#
# CLS EMBEDDING vs MEAN POOLING:
#   → CLS embedding : take the [CLS] token's output vector → sentence representation
#   → Mean pooling  : average ALL token vectors (often more accurate)
#                     (weighted by attention_mask to ignore padding)
#
# WHY USE EMBEDDINGS?
#   → Semantic similarity: "happy" and "joyful" → close vectors
#   → Clustering, search, classification, recommendation systems

print("\n" + "=" * 55)
print("3. BERT EMBEDDINGS")
print("=" * 55)

bert_model = AutoModel.from_pretrained("bert-base-uncased")
bert_model.eval()   # set to evaluation mode (disables dropout)

def get_embeddings(texts: list) -> np.ndarray:
    encoded = tokenizer(texts, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
    with torch.no_grad():   # no_grad = don't compute gradients (faster, less memory)
        outputs = bert_model(**encoded)

    # Mean pooling — average token embeddings weighted by attention mask
    attention_mask = encoded["attention_mask"].unsqueeze(-1)
    mean_emb = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
    return mean_emb.numpy()

texts      = ["I love machine learning", "I enjoy deep learning", "Football is fun"]
embeddings = get_embeddings(texts)
print(f"Embeddings shape: {embeddings.shape}")   # (3, 768)

from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(embeddings)
print(f"Similarity(ML, DL):       {sim[0,1]:.4f}")   # high — both about AI
print(f"Similarity(ML, Football): {sim[0,2]:.4f}")   # low  — different topics


# ══════════════════════════════════════════════════════
# 4. SENTENCE TRANSFORMERS — Best for Semantic Similarity
# ══════════════════════════════════════════════════════
# WHAT ARE SENTENCE TRANSFORMERS?
#   → Models specifically fine-tuned to produce SENTENCE-LEVEL embeddings
#   → Better than plain BERT for similarity tasks (BERT wasn't trained for this)
#   → all-MiniLM-L6-v2 : small (22M params), fast, 384-dim embeddings, very good
#   → all-mpnet-base-v2: larger, slower, better quality
#
# COSINE SIMILARITY:
#   → Measures angle between two vectors (1 = identical direction, 0 = orthogonal)
#   → Range: -1 to 1 (for normalized vectors: 0 to 1)
#   → Does NOT care about vector length — only direction
#
# USE CASES:
#   → Semantic search (find most similar documents)
#   → Duplicate detection
#   → Clustering similar texts
#   → RAG (Retrieval Augmented Generation) — embed + store + retrieve

print("\n" + "=" * 55)
print("4. SENTENCE TRANSFORMERS")
print("=" * 55)

from sentence_transformers import SentenceTransformer, util

smodel    = SentenceTransformer("all-MiniLM-L6-v2")
sentences = [
    "Machine learning is a subset of AI",
    "AI includes machine learning techniques",
    "Football is played with a round ball",
    "Soccer is popular worldwide",
]

embeddings  = smodel.encode(sentences, convert_to_tensor=True)
print(f"Embeddings shape: {embeddings.shape}")   # (4, 384)

cos_scores = util.cos_sim(embeddings, embeddings)
print("\nPairwise similarities:")
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        print(f"  [{cos_scores[i][j]:.3f}] '{sentences[i][:35]}' ↔ '{sentences[j][:35]}'")

# Semantic search — find most similar sentences to a query
query   = "deep learning neural networks"
q_emb   = smodel.encode(query, convert_to_tensor=True)
hits    = util.semantic_search(q_emb, embeddings, top_k=2)
print(f"\nTop matches for: '{query}'")
for hit in hits[0]:
    print(f"  [{hit['score']:.3f}] {sentences[hit['corpus_id']]}")


# ══════════════════════════════════════════════════════
# 5. FINE-TUNING — Train Model on Your Own Data
# ══════════════════════════════════════════════════════
# WHAT IS FINE-TUNING?
#   → Take a pre-trained model and CONTINUE training on your specific task/data
#   → Much faster and cheaper than training from scratch
#   → The pre-trained weights are a great starting point (transfer learning)
#
# WHY FINE-TUNE?
#   → Pre-trained model: general language understanding
#   → After fine-tuning: specialized for YOUR domain (medical, legal, finance)
#   → Example: BERT → fine-tune on customer reviews → better sentiment for your product
#
# PROCESS:
#   1. Load pre-trained model + tokenizer (from HuggingFace Hub)
#   2. Prepare your dataset (tokenize, split train/test)
#   3. Set TrainingArguments (learning rate, epochs, batch size)
#   4. Create Trainer object
#   5. Call trainer.train()
#
# KEY TRAINING ARGS:
#   → num_train_epochs        : how many full passes over data
#   → learning_rate           : 2e-5 to 5e-5 works well for fine-tuning
#   → per_device_train_batch_size: how many samples per GPU per step
#   → evaluation_strategy     : when to evaluate ("epoch" or "steps")
#   → load_best_model_at_end  : keep the best checkpoint

print("\n" + "=" * 55)
print("5. FINE-TUNING")
print("=" * 55)

model_name = "distilbert-base-uncased"
ft_tokenizer = AutoTokenizer.from_pretrained(model_name)
ft_model     = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Prepare dataset
data    = {
    "text":  ["I love this", "Terrible product", "Great quality", "Worst ever",
              "Absolutely fantastic", "Very disappointing"],
    "label": [1, 0, 1, 0, 1, 0]
}
dataset = Dataset.from_dict(data)

def tokenize_fn(examples):
    return ft_tokenizer(examples["text"], truncation=True,
                        max_length=128, padding="max_length")

tokenized_ds = dataset.map(tokenize_fn, batched=True)
tokenized_ds = tokenized_ds.train_test_split(test_size=0.33)

training_args = TrainingArguments(
    output_dir               = "./results",
    num_train_epochs         = 3,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size  = 8,
    learning_rate            = 2e-5,
    weight_decay             = 0.01,
    eval_strategy            = "epoch",   # evaluate after each epoch
    save_strategy            = "epoch",
    load_best_model_at_end   = True,
    report_to                = "none",
)

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds),
            "f1":       f1_score(labels, preds, average="weighted")}

trainer = Trainer(
    model           = ft_model,
    args            = training_args,
    train_dataset   = tokenized_ds["train"],
    eval_dataset    = tokenized_ds["test"],
    tokenizer       = ft_tokenizer,
    compute_metrics = compute_metrics,
)

# trainer.train()   # uncomment to actually run
print("Trainer configured — call trainer.train() to start fine-tuning")
print("After training: trainer.save_model('./my-model')")


# ══════════════════════════════════════════════════════
# 6. INFERENCE — Use Trained Model for Predictions
# ══════════════════════════════════════════════════════
# WHAT IS INFERENCE?
#   → Using a trained model to make predictions on NEW data
#   → model.eval()   → turn off dropout/batch norm (inference mode)
#   → torch.no_grad(): → don't compute gradients (faster, less memory)
#   → softmax        → convert raw logits to probabilities (sum to 1)
#   → argmax         → pick class with highest probability

print("\n" + "=" * 55)
print("6. INFERENCE")
print("=" * 55)

def predict(texts: list, model, tokenizer, device="cpu"):
    model.eval()
    encoded = tokenizer(texts, padding=True, truncation=True,
                        max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encoded)
    probs = torch.softmax(outputs.logits, dim=-1)   # convert logits → probabilities
    preds = torch.argmax(probs, dim=-1)              # class with highest probability
    return preds.tolist(), probs.tolist()

test_texts  = ["This is great!", "Totally disappointed."]
preds, probs = predict(test_texts, ft_model, ft_tokenizer)
for t, p, prob in zip(test_texts, preds, probs):
    label = "POS" if p == 1 else "NEG"
    print(f"  {t:25s} → {label} ({max(prob):.3f} confidence)")

print("\nAll done! ✓")
