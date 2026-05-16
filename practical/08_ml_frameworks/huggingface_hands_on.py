"""
WORKSHOP 01 — Hugging Face + LLMs Basics
==========================================
WHAT YOU LEARN:
  - Load a real LLM (GPT-2, no account needed)
  - Tokenize text (see how words become numbers)
  - Generate text (see the model actually think)
  - Control generation (temperature, top-p)
  - Get embeddings (turn text into vectors)

WHY THIS MATTERS FOR THE JOB:
  Every LLM task — fine-tuning, RAG, deployment —
  starts with exactly this code. This is your foundation.

SETUP:
  pip install transformers torch sentencepiece
"""

# ============================================================
# PART 1: TOKENIZATION — How text becomes numbers
# ============================================================

from transformers import AutoTokenizer

print("=" * 60)
print("PART 1: TOKENIZATION")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "Hello, I am learning about large language models!"

# Tokenize
tokens = tokenizer.tokenize(text)
print(f"\nOriginal text: {text}")
print(f"Tokens:        {tokens}")
# Output: ['Hello', ',', 'ĠI', 'Ġam', 'Ġlearning', ...]
# Notice: 'Ġ' means there's a space before the word

# Encode — convert tokens to IDs
input_ids = tokenizer.encode(text)
print(f"Token IDs:     {input_ids}")
# Output: [15496, 11, 314, 716, 4673, ...]

# Decode — convert IDs back to text
decoded = tokenizer.decode(input_ids)
print(f"Decoded:       {decoded}")

# Full encode with attention mask (what models actually use)
inputs = tokenizer(text, return_tensors="pt")
print(f"\nModel inputs shape: {inputs['input_ids'].shape}")
print(f"Attention mask:     {inputs['attention_mask']}")
# attention_mask: 1 = real token, 0 = padding (none here)

# INTERVIEW POINT: Tokens ≠ words. "tokenization" → ["token", "ization"]
# ~1.3 tokens per word in English on average
print(f"\nToken count: {len(tokens)} tokens for {len(text.split())} words")


# ============================================================
# PART 2: TEXT GENERATION — Make the model generate
# ============================================================

from transformers import AutoModelForCausalLM
import torch

print("\n" + "=" * 60)
print("PART 2: TEXT GENERATION")
print("=" * 60)

model_name = "gpt2"   # Small, fast, no login needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()   # eval mode = disable dropout for inference

prompt = "Artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

print(f"\nPrompt: '{prompt}'")

# --- GREEDY DECODING (always picks most likely token) ---
with torch.no_grad():
    greedy_output = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,      # greedy
    )
greedy_text = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
print(f"\nGreedy output:\n  {greedy_text}")

# --- SAMPLING WITH TEMPERATURE ---
with torch.no_grad():
    sampled_output = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.7,      # < 1 = focused, > 1 = creative
        top_p=0.9,            # nucleus sampling
        top_k=50,             # only consider top 50 tokens
    )
sampled_text = tokenizer.decode(sampled_output[0], skip_special_tokens=True)
print(f"\nSampled output (temp=0.7):\n  {sampled_text}")

# Run twice — greedy always same, sampled changes
print("\n[Run again to see sampled output is different every time]")


# ============================================================
# PART 3: WHAT THE MODEL ACTUALLY OUTPUTS (LOGITS)
# ============================================================

print("\n" + "=" * 60)
print("PART 3: INSIDE THE MODEL — LOGITS")
print("=" * 60)

"""
CONCEPT: At each step, the model outputs a score (logit)
for EVERY word in its vocabulary. Softmax turns these
into probabilities. We sample the next token from that.
"""

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits    # shape: [batch, seq_len, vocab_size]
print(f"\nLogits shape: {logits.shape}")
# (1, 7, 50257) = 1 batch, 7 tokens, 50257 vocab words

# Get probabilities for the LAST position (next token prediction)
import torch.nn.functional as F
last_logits = logits[0, -1, :]          # shape: [50257]
probs = F.softmax(last_logits, dim=-1)

# Top 5 predictions
top5_probs, top5_ids = torch.topk(probs, 5)
print(f"\nPrompt: '{prompt}'")
print("Top 5 next tokens:")
for prob, token_id in zip(top5_probs, top5_ids):
    token = tokenizer.decode([token_id.item()])
    print(f"  '{token}' → probability: {prob.item():.4f}")
# Output:  ' Paris' → 0.2341, ' the' → 0.0890, ...

# INTERVIEW POINT: This is exactly how temperature works!
# temp=0.1: probs become [0.99, 0.001, ...] → always picks Paris
# temp=2.0: probs become [0.25, 0.24, ...] → randomly picks any


# ============================================================
# PART 4: USING A PIPELINE (Production Way)
# ============================================================

from transformers import pipeline

print("\n" + "=" * 60)
print("PART 4: PIPELINES (PRODUCTION SHORTCUT)")
print("=" * 60)

# Text generation pipeline
generator = pipeline("text-generation", model="gpt2")

result = generator(
    "Machine learning engineers should know",
    max_new_tokens=40,
    num_return_sequences=2,    # generate 2 different completions
    temperature=0.8,
    do_sample=True,
)

print("\nGenerated 2 completions:")
for i, r in enumerate(result):
    print(f"\n[{i+1}] {r['generated_text']}")


# ============================================================
# PART 5: EMBEDDINGS — Text to Vector (Used in RAG)
# ============================================================

from transformers import AutoModel
import numpy as np

print("\n" + "=" * 60)
print("PART 5: EMBEDDINGS — Text to Vector")
print("=" * 60)

"""
CONCEPT: An embedding is a fixed-size vector that captures
the MEANING of a sentence. Similar sentences have similar vectors.
This is the foundation of RAG (semantic search).
"""

# Use a dedicated embedding model
from transformers import AutoTokenizer, AutoModel
import torch

embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
embed_model = AutoModel.from_pretrained(embed_model_name)

def get_embedding(text):
    """Convert text to a 384-dimensional vector."""
    inputs = embed_tokenizer(text, return_tensors="pt",
                              truncation=True, max_length=512,
                              padding=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    # Mean pooling over token embeddings
    token_embeddings = outputs.last_hidden_state   # [1, seq_len, 384]
    attention_mask = inputs['attention_mask']
    mask_expanded = attention_mask.unsqueeze(-1).float()
    embedding = (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1)
    return embedding.squeeze().numpy()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Embed sentences
sentences = [
    "How do I reset my password?",
    "I forgot my login credentials and need to recover them.",
    "What is the capital of France?",
    "Tell me about the Eiffel Tower.",
]

embeddings = [get_embedding(s) for s in sentences]
print(f"\nEmbedding shape: {embeddings[0].shape}")   # (384,)

query = "I can't log in to my account"
query_embedding = get_embedding(query)

print(f"\nQuery: '{query}'")
print("\nSimilarity scores:")
for i, (sent, emb) in enumerate(zip(sentences, embeddings)):
    score = cosine_similarity(query_embedding, emb)
    print(f"  [{score:.4f}] {sent}")

# Output:
# [0.8901] How do I reset my password?          ← MOST SIMILAR
# [0.8234] I forgot my login credentials...
# [0.1234] What is the capital of France?        ← NOT RELATED
# [0.0987] Tell me about the Eiffel Tower.

# INTERVIEW POINT: This is EXACTLY how RAG retrieval works.
# The query becomes a vector, we find the closest document vectors.


# ============================================================
# EXERCISE — Do This Yourself
# ============================================================

print("\n" + "=" * 60)
print("EXERCISE — TRY THIS YOURSELF")
print("=" * 60)

print("""
1. EASY: Change the prompt in Part 2 to something about Python coding.
   Run with temperature=0.2 and temperature=1.5.
   See how the outputs differ.

2. MEDIUM: In Part 3, change the prompt to:
   "The best programming language for AI is"
   Print the top 10 predictions instead of 5.
   Are the predictions sensible?

3. HARD: In Part 5, create 5 sentences about food.
   Create 1 query sentence about food.
   Check that the most similar sentence is food-related.
   This is a tiny RAG retrieval system you built yourself!

4. CHALLENGE: Generate the same prompt 5 times with temperature=1.0
   and 5 times with temperature=0.1.
   Print all 10 and see the difference in diversity.
""")

print("=" * 60)
print("WHAT TO SAY IN YOUR INTERVIEW:")
print("=" * 60)
print("""
"In practice I work with the HuggingFace Transformers library.
For generation, I load a CausalLM model — GPT, LLaMA, Mistral.
The tokenizer converts text to token IDs, the model outputs
logits over the vocabulary at each position, and we sample
the next token using temperature and top-p.

For embeddings and RAG, I use a bi-encoder like BGE-large —
it converts chunks and queries into 1024-dimensional vectors,
and we use cosine similarity to find the most relevant chunks."
""")
