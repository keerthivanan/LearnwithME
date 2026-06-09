"""
NLTK + spaCy — Definitions + Code + Outputs
=============================================
pip install nltk spacy
python -m spacy download en_core_web_sm
"""

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("maxent_ne_chunker", quiet=True)
nltk.download("maxent_ne_chunker_tab", quiet=True)
nltk.download("words", quiet=True)
nltk.download("vader_lexicon", quiet=True)


# ══════════════════════════════════════════════════════
# 1. NLTK BASICS — Tokenization, Stopwords, Stemming
# ══════════════════════════════════════════════════════
# WHAT IS NLTK?
#   → Natural Language Toolkit — classic Python NLP library
#   → Great for: tokenization, POS tagging, NER, sentiment analysis
#   → Used in academia and for learning NLP fundamentals
#
# TOKENIZATION:
#   → Split text into smaller units (tokens)
#   → sent_tokenize() → split into SENTENCES
#   → word_tokenize() → split into WORDS (handles punctuation properly)
#   → WHY: most NLP models need text as a list of tokens, not a raw string
#
# STOPWORDS:
#   → Common words that carry little meaning: "the", "is", "at", "which"
#   → Remove them to reduce noise and focus on meaningful words
#
# STEMMING vs LEMMATIZATION:
#   → Stemming    : chops off suffix  → "running" → "run" (fast, rough)
#                   "studies" → "studi" (might not be a real word!)
#   → Lemmatization: returns proper base form using vocabulary
#                   "studies" → "study", "better" → "good"  (slower, accurate)

print("=" * 55)
print("1. NLTK BASICS")
print("=" * 55)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk

text = "Apple is looking at buying U.K. startup for $1 billion. Tim Cook announced this on Monday."

# Sentence tokenization
sentences = sent_tokenize(text)
print("Sentences:", sentences)
# ['Apple is looking at buying U.K. startup for $1 billion.', 'Tim Cook announced this on Monday.']

# Word tokenization — smarter than split() (handles contractions, punctuation)
words = word_tokenize(text)
print("Words:", words[:10])
# ['Apple', 'is', 'looking', 'at', 'buying', 'U.K.', 'startup', 'for', '$', '1']

# Remove stopwords
stop_words = set(stopwords.words("english"))
filtered   = [w for w in words if w.lower() not in stop_words and w.isalpha()]
print("After stopword removal:", filtered)
# ['Apple', 'looking', 'buying', 'startup', 'billion', 'Tim', 'Cook', 'announced', 'Monday']

# Stemming — fast but rough (may produce non-words)
stemmer = PorterStemmer()
stems   = [stemmer.stem(w) for w in filtered]
print("Stems:  ", stems)
# ['appl', 'look', 'bui', 'startup', 'billion', 'tim', 'cook', 'announc', 'monday']

# Lemmatization — slow but accurate (returns real words)
lemmatizer = WordNetLemmatizer()
lemmas     = [lemmatizer.lemmatize(w.lower(), pos='v') for w in filtered]
print("Lemmas: ", lemmas)
# ['apple', 'look', 'buy', 'startup', 'billion', 'tim', 'cook', 'announce', 'monday']

# POS Tagging — label each word with its grammatical role
# NNP=proper noun, VBZ=verb present, VBG=verb gerund, IN=preposition
tagged = pos_tag(words[:8])
print("POS Tags:", tagged)
# [('Apple','NNP'),('is','VBZ'),('looking','VBG'),('at','IN'),('buying','VBG'),...]

# Named Entity Recognition — identify PEOPLE, ORGs, PLACES, DATES, MONEY
ner_tree = ne_chunk(pos_tag(word_tokenize(text)))
for entity in ner_tree:
    if hasattr(entity, "label"):
        print(f"  NER: {' '.join(c[0] for c in entity):20} → {entity.label()}")
# NER: Apple               → GPE
# NER: Tim Cook            → PERSON


# ══════════════════════════════════════════════════════
# 2. TEXT PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════
# WHAT IS TEXT PREPROCESSING?
#   → Cleaning raw text before feeding it to an NLP model
#   → Raw text has noise: URLs, emails, numbers, HTML, extra spaces
#   → Standard pipeline:
#     1. Lowercase          → "Apple" and "apple" = same word
#     2. Remove URLs/emails → not meaningful for most tasks
#     3. Remove punctuation/numbers → usually noise
#     4. Tokenize           → split into words
#     5. Remove stopwords   → remove "the", "is", "at" etc.
#     6. Lemmatize          → reduce to base form

print("\n" + "=" * 55)
print("2. TEXT PREPROCESSING PIPELINE")
print("=" * 55)

import re

def preprocess(text: str) -> str:
    text   = text.lower()                               # 1. lowercase
    text   = re.sub(r"http\S+|www\S+", "", text)       # 2. remove URLs
    text   = re.sub(r"\S+@\S+", "", text)              # 3. remove emails
    text   = re.sub(r"[^a-z\s]", "", text)             # 4. keep only letters+spaces
    tokens = word_tokenize(text)                         # 5. tokenize
    stop   = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop and len(t) > 2]  # 6. remove stopwords
    lem    = WordNetLemmatizer()
    tokens = [lem.lemmatize(t) for t in tokens]         # 7. lemmatize
    return " ".join(tokens)

review  = "I absolutely LOVED this product! It's amazing. Visit https://shop.com for more."
cleaned = preprocess(review)
print(f"Original: {review}")
print(f"Cleaned:  {cleaned}")
# Cleaned: absolutely love product amazing visit


# ══════════════════════════════════════════════════════
# 3. SENTIMENT ANALYSIS — VADER
# ══════════════════════════════════════════════════════
# WHAT IS SENTIMENT ANALYSIS?
#   → Determine the EMOTIONAL TONE of text: positive, negative, neutral
#
# WHAT IS VADER?
#   → Valence Aware Dictionary and sEntiment Reasoner
#   → Rule-based: works on a pre-built lexicon of words with sentiment scores
#   → Great for SOCIAL MEDIA text (handles CAPS, !, emojis, slang)
#   → No training required — works out of the box
#
# VADER SCORES:
#   → pos, neg, neu : proportions of text that are positive/negative/neutral
#   → compound     : overall score from -1 (most negative) to +1 (most positive)
#   → Threshold: compound >= 0.05 = POSITIVE, <= -0.05 = NEGATIVE, else NEUTRAL

print("\n" + "=" * 55)
print("3. SENTIMENT ANALYSIS — VADER")
print("=" * 55)

from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

texts = [
    "I absolutely love this product! It's amazing!",
    "This is the worst product I've ever bought.",
    "The product is okay, nothing special.",
    "Great quality but expensive price.",
]
for t in texts:
    score     = sia.polarity_scores(t)
    sentiment = ("POSITIVE" if score["compound"] >= 0.05 else
                 "NEGATIVE" if score["compound"] <= -0.05 else "NEUTRAL")
    print(f"  {sentiment:8s} ({score['compound']:+.3f}) → {t}")
# POSITIVE (+0.743) → I absolutely love this product!...
# NEGATIVE (-0.796) → This is the worst product...
# NEUTRAL  (+0.000) → The product is okay...
# POSITIVE (+0.296) → Great quality but expensive...


# ══════════════════════════════════════════════════════
# 4. TF-IDF — Text to Numbers for ML
# ══════════════════════════════════════════════════════
# WHAT IS TF-IDF?
#   → Term Frequency–Inverse Document Frequency
#   → Converts text into numerical vectors that ML models can use
#
# TF (Term Frequency):
#   → How often a word appears in THIS document
#   → High TF = word is important for this specific document
#
# IDF (Inverse Document Frequency):
#   → How RARE the word is across ALL documents
#   → High IDF = word appears in few documents = more distinctive
#   → "the" appears in every doc → low IDF → less important
#   → "neural" appears in few docs → high IDF → more important
#
# TF-IDF = TF × IDF → high score = word is frequent HERE and rare elsewhere
#
# ngram_range=(1,2): use single words AND word pairs ("machine learning" as one feature)
# max_features     : keep only top N most frequent terms

print("\n" + "=" * 55)
print("4. TF-IDF")
print("=" * 55)

from sklearn.feature_extraction.text import TfidfVectorizer

docs   = [
    "I love machine learning and Python",
    "Deep learning with neural networks is amazing",
    "Python is great for data science",
    "I hate bugs in my code",
    "Debugging is frustrating and annoying",
    "The error messages are confusing",
]
labels = [1, 1, 1, 0, 0, 0]   # 1=positive, 0=negative

vectorizer = TfidfVectorizer(
    max_features=100,
    ngram_range=(1, 2),     # unigrams + bigrams ("machine learning" as one feature)
    stop_words="english",
    min_df=1
)
X = vectorizer.fit_transform(docs)
print(f"TF-IDF matrix shape: {X.shape}")     # (6, N_features)
print(f"Features: {vectorizer.get_feature_names_out()[:8]}")


# ══════════════════════════════════════════════════════
# 5. SPACY — Industrial-Strength NLP
# ══════════════════════════════════════════════════════
# WHAT IS SPACY?
#   → Modern, fast NLP library optimized for PRODUCTION use
#   → Pre-trained models for 60+ languages
#   → Pipeline-based: text → tokenize → POS tag → NER → dependency parse
#
# SPACY vs NLTK:
#   → NLTK:  educational, modular, good for learning and research
#   → spaCy: production-ready, faster, better pre-trained models
#   → Use spaCy for real projects, NLTK for learning concepts
#
# WHAT IS A DOC OBJECT?
#   → nlp(text) returns a Doc — the central object in spaCy
#   → Doc is a sequence of Token objects
#   → Each token has: .text, .pos_, .dep_, .lemma_, .ent_type_
#
# NER LABELS (common):
#   → PERSON  : people (Steve Jobs)
#   → ORG     : companies/institutions (Apple, NASA)
#   → GPE     : countries, cities, states (California, India)
#   → DATE    : dates and periods
#   → MONEY   : monetary values
#   → LOC     : locations (mountains, rivers)

print("\n" + "=" * 55)
print("5. SPACY")
print("=" * 55)

import spacy
nlp  = spacy.load("en_core_web_sm")
text = "Apple was founded by Steve Jobs and Steve Wozniak in Cupertino, California in 1976."
doc  = nlp(text)

# Tokens — each word/punctuation is a Token object
print("Tokens:")
for token in doc[:5]:
    print(f"  {token.text:12} | POS: {token.pos_:6} | Dep: {token.dep_:10} | Lemma: {token.lemma_}")

# Named Entity Recognition
print("\nEntities:")
for ent in doc.ents:
    print(f"  {ent.text:20} → {ent.label_:8} ({spacy.explain(ent.label_)})")
# Apple              → ORG      (Companies, agencies, institutions)
# Steve Jobs         → PERSON   (People, including fictional)
# Steve Wozniak      → PERSON
# Cupertino          → GPE      (Countries, cities, states)
# California         → GPE
# 1976               → DATE     (Absolute or relative dates)

# Noun chunks — meaningful noun phrases
print("\nNoun chunks:")
for chunk in doc.noun_chunks:
    print(f"  '{chunk.text}' → root: '{chunk.root.text}'")


# ══════════════════════════════════════════════════════
# 6. SPACY EXTRACTION PIPELINE
# ══════════════════════════════════════════════════════
# WHAT IS A SPACY PIPELINE?
#   → When you call nlp(text), spaCy runs a series of components in order:
#     tokenizer → tagger (POS) → parser (dependencies) → NER → ...
#   → You can add/remove/customize components
#
# TYPICAL USE: build an info-extraction function that pulls
#   structured data (entities, noun phrases, sentences) from raw text

print("\n" + "=" * 55)
print("6. SPACY EXTRACTION PIPELINE")
print("=" * 55)

def extract_info(text: str) -> dict:
    doc = nlp(text)
    return {
        "entities":    [(e.text, e.label_) for e in doc.ents],
        "noun_chunks": [c.text for c in doc.noun_chunks],
        "sentences":   [s.text.strip() for s in doc.sents],
        "token_count": len(doc),
    }

sample = "Elon Musk founded Tesla in 2003 and SpaceX in 2002, both based in California."
info   = extract_info(sample)
print(f"Entities:    {info['entities']}")
print(f"Noun chunks: {info['noun_chunks']}")
print(f"Sentences:   {info['sentences']}")
print(f"Token count: {info['token_count']}")


# ══════════════════════════════════════════════════════
# 7. SPACY RULE-BASED MATCHING
# ══════════════════════════════════════════════════════
# WHAT IS RULE-BASED MATCHING?
#   → Find patterns in text using RULES instead of (or in addition to) ML
#   → spaCy Matcher: define patterns as lists of token attribute dicts
#   → LOWER: match lowercase form, POS: match part of speech, etc.
#   → Use for: finding specific phrases, extracting structured data,
#     boosting NER with domain-specific rules (medical, legal, finance)

print("\n" + "=" * 55)
print("7. SPACY RULE-BASED MATCHING")
print("=" * 55)

from spacy.matcher import Matcher

matcher = Matcher(nlp.vocab)

# Rule: find exactly "machine learning" or "deep learning"
pattern1 = [{"LOWER": "machine"}, {"LOWER": "learning"}]
pattern2 = [{"LOWER": "deep"},    {"LOWER": "learning"}]
matcher.add("AI_TERM", [pattern1, pattern2])

doc2    = nlp("I study machine learning and deep learning every day.")
matches = matcher(doc2)
for match_id, start, end in matches:
    span = doc2[start:end]
    print(f"  Matched: '{span.text}' at positions [{start}:{end}]")
# Matched: 'machine learning' at positions [2:4]
# Matched: 'deep learning' at positions [5:7]


# ══════════════════════════════════════════════════════
# 8. WORD FREQUENCY & N-GRAMS
# ══════════════════════════════════════════════════════
# WHAT ARE N-GRAMS?
#   → Contiguous sequences of N words from a text
#   → Unigram (n=1): each word individually  → ["machine", "learning"]
#   → Bigram  (n=2): pairs of consecutive words → ["machine learning", "deep learning"]
#   → Trigram (n=3): triples → ["deep learning model"]
#
# WHY USE N-GRAMS?
#   → Capture CONTEXT that single words miss
#   → "New York" as a bigram is better than "New" + "York" separately
#   → Used in language models, text classification, spam detection
#
# WORD FREQUENCY:
#   → Counter({word: count}) → shows most common words in a corpus
#   → After removing stopwords → reveals TOPIC of the text

print("\n" + "=" * 55)
print("8. WORD FREQUENCY & N-GRAMS")
print("=" * 55)

from collections import Counter
from nltk.util import ngrams

corpus = """machine learning is a subset of artificial intelligence.
deep learning uses neural networks. neural networks are inspired by the brain."""

tokens = [t for t in word_tokenize(corpus.lower())
          if t.isalpha() and t not in stopwords.words("english")]

freq = Counter(tokens)
print("Top 5 words:", freq.most_common(5))
# [('learning', 2), ('neural', 2), ('networks', 2), ('machine', 1), ...]

bigrams      = list(ngrams(tokens, 2))
bigram_freq  = Counter(bigrams)
print("Top 3 bigrams:", bigram_freq.most_common(3))
# [('neural', 'networks'), ('machine', 'learning'), ('deep', 'learning')]

print("\nAll done! ✓")
