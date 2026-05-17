# =====================================================================
# PYTHON FOR GenAI ENGINEERS: SCIKIT-LEARN MASTERCLASS
# =====================================================================
# This script covers classical machine learning baselines, embedding
# clustering, PCA projection, and evaluation metric utilities 
# that are crucial for high-performance RAG and GenAI pipelines.
# 
# Run: python practical/13_python_libraries/python_masterclass_sklearn.py
# =====================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, f1_score

def heading(title):
    print("\n" + "=" * 65)
    print(f"[{title}]")
    print("=" * 65)

# ---------------------------------------------------------------------
# 1. TEXT REPRESENTATION: TF-IDF VECTORIZER (SPARSE RETRIEVAL)
# ---------------------------------------------------------------------
# THE PROBLEM:
#    In RAG systems, relying solely on dense neural embeddings can sometimes
#    fail to capture specific, exact keyword matches (e.g., product IDs, SKU numbers).
#    Hybrid search combines dense vectors with sparse vector representations (like TF-IDF or BM25).
#
# THE SOLUTION:
#    Convert text corpora into highly optimized sparse frequency-inverse document frequency matrices.
# ---------------------------------------------------------------------
heading("1. TF-IDF & Sparse Vectorization for Classical Retrieval")

documents = [
    "LLaMA is an open-source large language model created by Meta.",
    "RAG combining retrieval mechanisms with generative LLMs provides grounded facts.",
    "Retrieval-Augmented Generation helps reduce model hallucinations.",
    "Fine-tuning adjusts model weights on custom instruction datasets.",
    "FastAPI is an async-native Python framework used for deploying high-performance APIs."
]

# Instantiate Vectorizer
# ngram_range=(1,2) extracts both single words and two-word combinations (bigrams)
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words="english")

# Learn the vocabulary and transform document corpus into feature vectors
tfidf_matrix = vectorizer.fit_transform(documents)

print("TF-IDF Matrix Shape:", tfidf_matrix.shape) # (5 docs, unique tokens features)
print("Extracted Feature Names (First 10):\n", vectorizer.get_feature_names_out()[:10])

# Perform simple vector similarity search (simulating standard sparse search index)
query = "retrieval augmented generation"
query_vec = vectorizer.transform([query])
scores = (tfidf_matrix @ query_vec.T).toarray().flatten()

print(f"\nSparse Search Scores for query '{query}':")
for idx, doc in enumerate(documents):
    print(f"  [{scores[idx]:.3f}] {doc}")


# ---------------------------------------------------------------------
# 2. CLASSICAL MACHINE LEARNING BASELINES FOR CLASSIFICATION
# ---------------------------------------------------------------------
# THE PROBLEM:
#    Hiring managers hate when engineers immediately deploy a heavy, expensive
#    GPU-based LLM for simple text routing or classification without trying a
#    classical model. You MUST show that you can build quick baselines.
#
# THE SOLUTION:
#    Use Scikit-learn's LogisticRegression + TF-IDF vectors as an instant baseline.
# ---------------------------------------------------------------------
heading("2. Building Low-Latency Text Classification Baselines")

# Mock dataset for sentiment routing (0 = negative/complaint, 1 = positive/billing)
corpus = [
    "My model training keeps crashing due to cuda out of memory errors",
    "How do I upgrade my billing tier and add my corporate credit card?",
    "The api server has a high latency and slow time-to-first-token responses",
    "Where can I find the invoices for my monthly subscription?",
    "The output of the instruction model is completely hallucinated and garbage",
    "I need to cancel my paid premium account immediately"
]
labels = [0, 1, 0, 1, 0, 1]  # 0 = Technical support, 1 = Billing/Account

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    corpus, labels, test_size=0.33, random_state=42, stratify=labels
)

# Extract TF-IDF features
baseline_vectorizer = TfidfVectorizer()
X_train_vec = baseline_vectorizer.fit_transform(X_train)
X_val_vec = baseline_vectorizer.transform(X_val)

# Fit Logistic Regression model (runs in milliseconds!)
clf = LogisticRegression(C=1.0, max_iter=1000)
clf.fit(X_train_vec, y_train)

# Evaluate
predictions = clf.predict(X_val_vec)
val_accuracy = accuracy_score(y_val, predictions)
print(f"Baseline Validation Accuracy: {val_accuracy * 100:.1f}%")
print("Target Validation Labels  :", y_val)
print("Predicted Validation Labels:", list(predictions))


# ---------------------------------------------------------------------
# 3. K-MEANS EMBEDDING CLUSTERING ( gap analysis )
# ---------------------------------------------------------------------
# THE PROBLEM:
#    In production RAG systems, users ask all sorts of queries. You need to
#    discover what topics users query about or identify cluster centroids
#    to build hierarchical indexes (like the RAPTOR approach).
#
# THE SOLUTION:
#    Cluster high-dimensional dense embeddings using KMeans.
# ---------------------------------------------------------------------
heading("3. K-Means Clustering on Dense Sentence Embeddings")

# Generate 6 mock 16-dimensional dense embeddings (representing text embeddings)
np.random.seed(42)
embeddings = np.concatenate([
    np.random.normal(loc=0.5, scale=0.1, size=(3, 16)),  # Cluster A (e.g. coding topics)
    np.random.normal(loc=-0.5, scale=0.1, size=(3, 16))  # Cluster B (e.g. billing topics)
])

# Fit KMeans clustering to find 2 distinct clusters
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(embeddings)

print("KMeans Assigned Cluster Labels:", cluster_labels)
print("Cluster Centers Shape         :", kmeans.cluster_centers_.shape)


# ---------------------------------------------------------------------
# 4. DIMENSIONALITY REDUCTION: PCA
# ---------------------------------------------------------------------
# THE PROBLEM:
#    High-dimensional vectors (e.g. 768-dim) are impossible to visualize.
#    To debug data embeddings or spot overlap, we must reduce them to 2D.
#
# THE SOLUTION:
#    Use Principal Component Analysis (PCA) to project vectors to their
#    top 2 principal directions of variation.
# ---------------------------------------------------------------------
heading("4. Dimensionality Reduction (PCA) for Embedding Debugging")

# Embeddings array of shape (6, 16)
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

print("Original Embedding Shape:", embeddings.shape)
print("PCA Projected Shape     :", reduced_embeddings.shape)
print("PCA Explained Variance Ratio:", pca.explained_variance_ratio_)
print("\nPCA Projected 2D coordinates for visualization:\n", np.round(reduced_embeddings, 4))


# ---------------------------------------------------------------------
# 5. PRODUCTION EVALUATION METRICS
# ---------------------------------------------------------------------
# THE PROBLEM:
#    Accuracy is misleading when classes are imbalanced (e.g., 99% non-spam, 1% spam).
#    If the model classifies everything as non-spam, accuracy is 99% but the spam system is broken.
#    You must evaluate precision, recall, and F1 scores.
# ---------------------------------------------------------------------
heading("5. Production Evaluation Metrics (Precision, Recall, F1)")

# Actual prompt routing outcomes
y_true_production = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  # Imbalanced test sample
y_pred_production = [0, 0, 1, 1, 1, 1, 1, 1, 0, 1]  # Predicted output

# Detailed classification metrics
print("Accuracy Score:", accuracy_score(y_true_production, y_pred_production))
print("Weighted F1    :", f1_score(y_true_production, y_pred_production, average="weighted"))

print("\nFull Classification Report:")
print(classification_report(y_true_production, y_pred_production, target_names=["Tech Support", "Billing"]))


# =====================================================================
# INTERVIEW PRACTICE QUESTIONS (SCIKIT-LEARN FOR GenAI)
# =====================================================================
print("\n" + "=" * 65)
print("[PRO-LEVEL INTERVIEW DRILLS (SCIKIT-LEARN FOR GenAI)]")
print("=" * 65)
print("""
Q1: Why is an F1-score preferred over standard Accuracy for evaluating LLM classification pipelines?
    
    -> Answer: Standard accuracy is deceptive when dealing with imbalanced datasets. 
       F1-score represents the harmonic mean of Precision (minimizing false positives) 
       and Recall (minimizing false negatives). It gives a robust measure of overall 
       model performance across both minority and majority categories.

Q2: How does the RAPTOR algorithm use clustering to perform advanced, multi-hop RAG?
    
    -> Answer: RAPTOR recursively clusters dense document embeddings using GMMs 
       or K-Means. It summarizes the text chunks in each cluster using an LLM, 
       then indexes those summaries. This creates a multi-layered hierarchical tree, 
       allowing the system to retrieve both high-level thematic summaries and 
       fine-grained content chunks during user queries.

Q3: Why would you use a TF-IDF vectorizer alongside dense embeddings in a production RAG system?
    
    -> Answer: Dense embeddings struggle with precise keyword matching, serial numbers, 
       and exact terminology because they map words to a semantic space. TF-IDF/BM25 
       scores are based on exact term frequencies. Combining them (hybrid search) 
       gives the best of both worlds: robust keyword matches and rich conceptual understanding.
""")
print("=" * 65 + "\n")
