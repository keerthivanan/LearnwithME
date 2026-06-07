"""
NLTK + spaCy — Everything
===========================
pip install nltk spacy
python -m spacy download en_core_web_sm
"""

import nltk
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("maxent_ne_chunker", quiet=True)
nltk.download("words", quiet=True)
nltk.download("vader_lexicon", quiet=True)

# ════════════════════════════════════════════
# 1. NLTK BASICS
# ════════════════════════════════════════════
print("1. NLTK BASICS")

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk

text = "Apple is looking at buying U.K. startup for $1 billion. Tim Cook announced this on Monday."

# Sentence tokenization
sentences = sent_tokenize(text)
print("Sentences:", sentences)
# ['Apple is looking at buying U.K. startup for $1 billion.', 'Tim Cook announced this on Monday.']

# Word tokenization
words = word_tokenize(text)
print("Words:", words[:10])
# ['Apple', 'is', 'looking', 'at', 'buying', 'U.K.', 'startup', 'for', '$', '1']

# Stopwords
stop_words = set(stopwords.words("english"))
filtered   = [w for w in words if w.lower() not in stop_words and w.isalpha()]
print("Filtered:", filtered)
# ['Apple', 'looking', 'buying', 'startup', 'billion', 'Tim', 'Cook', 'announced', 'Monday']

# Stemming (cuts off suffix — fast but rough)
stemmer = PorterStemmer()
stems   = [stemmer.stem(w) for w in filtered]
print("Stems:", stems)
# ['appl', 'look', 'bui', 'startup', 'billion', 'tim', 'cook', 'announc', 'monday']

# Lemmatization (proper base form — slower but accurate)
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(w.lower(), pos='v') for w in filtered]
print("Lemmas:", lemmas)
# ['apple', 'look', 'buy', 'startup', 'billion', 'tim', 'cook', 'announce', 'monday']

# POS Tagging
tagged = pos_tag(words[:8])
print("POS Tags:", tagged)
# [('Apple','NNP'),('is','VBZ'),('looking','VBG'),('at','IN'),('buying','VBG'),...]

# Named Entity Recognition
from nltk import ne_chunk, pos_tag, word_tokenize
ner_tree = ne_chunk(pos_tag(word_tokenize(text)))
for entity in ner_tree:
    if hasattr(entity, "label"):
        print(f"Entity: {' '.join(c[0] for c in entity)} → {entity.label()}")
# Entity: Apple → GPE
# Entity: Tim Cook → PERSON

# ════════════════════════════════════════════
# 2. TEXT PREPROCESSING PIPELINE (NLTK)
# ════════════════════════════════════════════
print("\n2. TEXT PREPROCESSING PIPELINE")

import re

def preprocess(text: str) -> str:
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove emails
    text = re.sub(r"\S+@\S+", "", text)
    # Remove punctuation and numbers
    text = re.sub(r"[^a-z\s]", "", text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords + short words
    stop = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop and len(t) > 2]
    # Lemmatize
    lem = WordNetLemmatizer()
    tokens = [lem.lemmatize(t) for t in tokens]
    return " ".join(tokens)

review = "I absolutely LOVED this product! It's amazing. Visit https://shop.com for more."
cleaned = preprocess(review)
print(f"Original: {review}")
print(f"Cleaned:  {cleaned}")
# Cleaned:  absolutely love product amazing visit

# ════════════════════════════════════════════
# 3. SENTIMENT ANALYSIS (VADER)
# ════════════════════════════════════════════
print("\n3. SENTIMENT ANALYSIS — VADER")

from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

texts = [
    "I absolutely love this product! It's amazing!",
    "This is the worst product I've ever bought.",
    "The product is okay, nothing special.",
    "Great quality but expensive price.",
]

for t in texts:
    score = sia.polarity_scores(t)
    sentiment = "POSITIVE" if score["compound"] >= 0.05 else \
                "NEGATIVE" if score["compound"] <= -0.05 else "NEUTRAL"
    print(f"{sentiment:8s} ({score['compound']:+.3f}) → {t[:50]}")

# POSITIVE (+0.743) → I absolutely love this product! It's amazing!
# NEGATIVE (-0.796) → This is the worst product I've ever bought.
# NEUTRAL  (+0.000) → The product is okay, nothing special.
# POSITIVE (+0.296) → Great quality but expensive price.

# ════════════════════════════════════════════
# 4. TF-IDF WITH SKLEARN
# ════════════════════════════════════════════
print("\n4. TF-IDF")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Sample corpus
docs = [
    "I love machine learning and Python",
    "Deep learning with neural networks is amazing",
    "Python is great for data science",
    "I hate bugs in my code",
    "Debugging is frustrating and annoying",
    "The error messages are confusing",
]
labels = [1, 1, 1, 0, 0, 0]

vectorizer = TfidfVectorizer(
    max_features=100,
    ngram_range=(1, 2),     # unigrams + bigrams
    stop_words="english",
    min_df=1
)
X = vectorizer.fit_transform(docs)
print(f"TF-IDF matrix shape: {X.shape}")  # (6, N_features)
print(f"Top features: {vectorizer.get_feature_names_out()[:10]}")

# ════════════════════════════════════════════
# 5. SPACY — Industrial NLP
# ════════════════════════════════════════════
print("\n5. SPACY")

import spacy
nlp = spacy.load("en_core_web_sm")

text = "Apple was founded by Steve Jobs and Steve Wozniak in Cupertino, California in 1976."
doc  = nlp(text)

# Tokens
print("Tokens:")
for token in doc:
    print(f"  {token.text:15} | POS: {token.pos_:6} | Dep: {token.dep_:10} | Lemma: {token.lemma_}")

# Named Entities
print("\nEntities:")
for ent in doc.ents:
    print(f"  {ent.text:20} → {ent.label_:10} ({spacy.explain(ent.label_)})")
# Apple              → ORG        (Companies, agencies)
# Steve Jobs         → PERSON     (People)
# Steve Wozniak      → PERSON
# Cupertino          → GPE        (Countries, cities, states)
# California         → GPE
# 1976               → DATE

# Noun chunks
print("\nNoun chunks:")
for chunk in doc.noun_chunks:
    print(f"  '{chunk.text}' → root: {chunk.root.text}")

# Dependency parsing
print("\nDependencies:")
for token in doc[:5]:
    print(f"  {token.text} --[{token.dep_}]--> {token.head.text}")

# ════════════════════════════════════════════
# 6. SPACY PIPELINE
# ════════════════════════════════════════════
print("\n6. SPACY PIPELINE")

def extract_info(text: str) -> dict:
    doc = nlp(text)
    return {
        "entities":    [(e.text, e.label_) for e in doc.ents],
        "noun_chunks": [c.text for c in doc.noun_chunks],
        "sentences":   [s.text for s in doc.sents],
        "tokens":      len(doc),
    }

sample = "Elon Musk founded Tesla in 2003 and SpaceX in 2002, both based in California."
info   = extract_info(sample)
print(f"Entities: {info['entities']}")
print(f"Chunks:   {info['noun_chunks']}")
print(f"Tokens:   {info['tokens']}")

# ════════════════════════════════════════════
# 7. SPACY CUSTOM RULES
# ════════════════════════════════════════════
print("\n7. SPACY RULE-BASED MATCHING")

from spacy.matcher import Matcher

matcher = Matcher(nlp.vocab)

# Rule: find "machine learning" or "deep learning"
pattern1 = [{"LOWER": "machine"}, {"LOWER": "learning"}]
pattern2 = [{"LOWER": "deep"}, {"LOWER": "learning"}]
matcher.add("AI_TERM", [pattern1, pattern2])

doc2 = nlp("I study machine learning and deep learning every day.")
matches = matcher(doc2)
for match_id, start, end in matches:
    span = doc2[start:end]
    print(f"  Matched: '{span.text}' at [{start}:{end}]")
# Matched: 'machine learning' at [2:4]
# Matched: 'deep learning' at [5:7]

# ════════════════════════════════════════════
# 8. WORD FREQUENCY & N-GRAMS
# ════════════════════════════════════════════
print("\n8. WORD FREQUENCY & N-GRAMS")

from collections import Counter
from nltk.util import ngrams

corpus = """machine learning is a subset of artificial intelligence.
deep learning uses neural networks. neural networks are inspired by the brain."""

tokens = [t for t in word_tokenize(corpus.lower())
          if t.isalpha() and t not in stopwords.words("english")]

# Word frequency
freq = Counter(tokens)
print("Top 5 words:", freq.most_common(5))
# [('learning', 2), ('neural', 2), ('networks', 2), ...]

# Bigrams
bigram_list = list(ngrams(tokens, 2))
bigram_freq = Counter(bigram_list)
print("Top 3 bigrams:", bigram_freq.most_common(3))

print("\nAll done! ✓")
