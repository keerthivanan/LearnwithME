# ==============================================================
# SESSION 1: Python ML Patterns — Production Level
# ==============================================================
# These are the Python patterns that appear in EVERY ML codebase.
# If you don't know these, you can't read PyTorch or HuggingFace code.
# Run this file: python lesson.py
# ==============================================================


# --------------------------------------------------------------
# CONCEPT 1: List vs Generator
# --------------------------------------------------------------
# THE PROBLEM:
#   You have 1 million text samples. If you load them ALL into a
#   list, you use gigabytes of RAM. Generators fix this.
#
# THEORY:
#   - List comprehension  → evaluates EVERYTHING immediately → stored in RAM
#   - Generator expression → evaluates ONE item at a time → tiny RAM footprint
#   - This is how PyTorch DataLoader works internally!
#
# INTERVIEW ANSWER:
#   "Generators are lazy iterators — they yield one item at a time
#    instead of materializing the entire dataset in memory. In ML,
#    we use them for batch processing large datasets. PyTorch's
#    DataLoader is built on this principle."
# --------------------------------------------------------------

print("=" * 55)
print("CONCEPT 1: List vs Generator")
print("=" * 55)

texts = ["hello", "world", "foo", "bar", "baz"]

# LIST — all 5 items loaded at once
texts_list = [t.upper() for t in texts]
print("List (all in memory):", texts_list)
print("Type:", type(texts_list))       # <class 'list'>

# GENERATOR — produces ONE at a time, never stores all
texts_gen = (t.upper() for t in texts)
print("\nGenerator object:", texts_gen) # object reference, not data
print("next() call 1:", next(texts_gen)) # HELLO
print("next() call 2:", next(texts_gen)) # WORLD
# The other 3 haven't been computed yet!

# REAL ML USE — batch generator (this is what DataLoader does)
def batch_generator(dataset, batch_size=2):
    """
    Yields mini-batches one at a time.
    If dataset has 1M items, only batch_size items are in RAM at once.
    """
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]

training_data = list(range(10))  # simulating 10 samples
print("\nBatch Generator output:")
for batch in batch_generator(training_data, batch_size=3):
    print("  batch:", batch)
# Mimics exactly how PyTorch DataLoader yields batches


# --------------------------------------------------------------
# CONCEPT 2: *args and **kwargs
# --------------------------------------------------------------
# THE PROBLEM:
#   model(**inputs) — you see this EVERYWHERE in HuggingFace.
#   What does ** do? Why not just pass arguments normally?
#
# THEORY:
#   - *args  → packs/unpacks POSITIONAL arguments as a tuple
#   - **kwargs → packs/unpacks KEYWORD arguments as a dict
#   - The tokenizer returns a dict. ** unpacks it as named params.
#
# INTERVIEW ANSWER:
#   "**kwargs unpacks a dictionary as keyword arguments. In HuggingFace,
#    tokenizer() returns {'input_ids': ..., 'attention_mask': ...}.
#    We call model(**inputs) which is equivalent to
#    model(input_ids=..., attention_mask=...). This pattern makes
#    code flexible and decoupled from specific argument names."
# --------------------------------------------------------------

print("\n" + "=" * 55)
print("CONCEPT 2: *args and **kwargs")
print("=" * 55)

# Basic **kwargs unpacking
def add(a, b):
    print(f"  a={a}, b={b}, sum={a+b}")

params = {"a": 3, "b": 5}
add(**params)           # same as add(a=3, b=5)

# *args — positional packing
def show_args(*args):
    for i, val in enumerate(args):
        print(f"  arg[{i}] = {val}")

show_args("GPT-4", "LLaMA", "Mistral")  # all packed into tuple

# REAL ML USE — exactly how HuggingFace model call works
def fake_transformer_model(input_ids, attention_mask, token_type_ids=None):
    """Simulates a HuggingFace model forward pass."""
    print(f"  input_ids shape     : {len(input_ids)} tokens")
    print(f"  attention_mask      : {attention_mask}")
    print(f"  token_type_ids      : {token_type_ids}")
    return {"logits": [0.1, 0.9]}  # fake output

# Tokenizer returns this dict — you DON'T unpack it manually
tokenizer_output = {
    "input_ids":      [101, 7592, 2088, 102],  # [CLS] hello world [SEP]
    "attention_mask": [1, 1, 1, 1],             # all real tokens (no padding)
}

print("\nCalling model(**tokenizer_output):")
result = fake_transformer_model(**tokenizer_output)
print("  Output:", result)
# This is EXACTLY model(**inputs) in real code


# --------------------------------------------------------------
# CONCEPT 3: Decorators
# --------------------------------------------------------------
# THE PROBLEM:
#   @torch.no_grad() — what is this? Why put it before functions?
#   @app.post("/predict") — how does FastAPI routing work?
#
# THEORY:
#   A decorator is a function that WRAPS another function,
#   adding behavior before/after without changing the original code.
#   Syntax sugar: @decorator is same as func = decorator(func)
#
# INTERVIEW ANSWER:
#   "Decorators wrap functions to add behavior. @torch.no_grad()
#    disables gradient tracking during inference — saving memory
#    and compute since we don't need backprop. @app.post() in
#    FastAPI registers the function as an HTTP route handler."
# --------------------------------------------------------------

print("\n" + "=" * 55)
print("CONCEPT 3: Decorators")
print("=" * 55)

import time

# Step 1: Build a decorator from scratch
def timer_decorator(func):
    """Adds timing to any function."""
    def wrapper(*args, **kwargs):        # accepts any arguments
        start = time.time()
        result = func(*args, **kwargs)   # call the original function
        elapsed = time.time() - start
        print(f"  [TIME] {func.__name__}() took {elapsed:.4f}s")
        return result
    return wrapper

@timer_decorator
def slow_embedding(text):
    """Simulates a slow embedding call."""
    time.sleep(0.01)
    return [0.1, 0.2, 0.3]  # fake embedding

vec = slow_embedding("hello world")
print("  embedding:", vec)

# Step 2: Simulate @torch.no_grad() — the most common ML decorator
def no_grad(func):
    """
    Simulates @torch.no_grad()
    In real PyTorch: disables gradient computation graph.
    Saves memory + compute during inference (not training).
    """
    def wrapper(*args, **kwargs):
        print("  >> Gradient tracking: OFF (inference mode)")
        result = func(*args, **kwargs)
        print("  >> Gradient tracking: ON (restored)")
        return result
    return wrapper

@no_grad
def predict(model_input):
    """Inference function — gradients never needed here."""
    fake_logits = [0.1, 0.85, 0.05]  # raw model output
    return fake_logits

print("\nRunning inference with @no_grad:")
logits = predict("user question about AI")
print("  Logits:", logits)

# Step 3: Decorator with arguments (like @app.post("/route"))
def route(path):
    """Simulates FastAPI's @app.post("/predict")"""
    def decorator(func):
        print(f"  Route registered: POST {path} -> {func.__name__}()")
        return func
    return decorator

print("\nFastAPI-style route registration:")

@route("/v1/generate")
def generate_endpoint(prompt: str):
    return {"text": "AI generated response"}

@route("/v1/embed")
def embed_endpoint(text: str):
    return {"embedding": [0.1, 0.2, 0.3]}


# --------------------------------------------------------------
# CONCEPT 4: Context Managers (with statement)
# --------------------------------------------------------------
# THE PROBLEM:
#   with torch.no_grad(): — seen in EVERY inference code block.
#   with torch.cuda.amp.autocast(): — used in mixed precision training.
#   Why "with"? What does it actually do?
#
# THEORY:
#   Context managers guarantee setup + teardown, even if an error occurs.
#   They implement __enter__ and __exit__ methods.
#   The "with" keyword calls __enter__ at start, __exit__ at end.
#
# INTERVIEW ANSWER:
#   "Context managers handle resource management — setup and cleanup
#    are guaranteed even if exceptions occur. In ML:
#    torch.no_grad() disables autograd graph (saves memory for inference),
#    torch.cuda.amp.autocast() enables mixed precision (FP16) for faster training."
# --------------------------------------------------------------

print("\n" + "=" * 55)
print("CONCEPT 4: Context Managers")
print("=" * 55)

# Build a context manager from scratch
class GradientContext:
    """
    Simulates torch.no_grad() context manager.
    Real PyTorch does: torch._C.set_grad_enabled(False)
    """
    def __enter__(self):
        print("  >> __enter__: gradient tracking DISABLED")
        return self   # this is what 'as' captures

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("  >> __exit__: gradient tracking RESTORED")
        # return False → don't suppress exceptions

# Using our custom context manager
print("Inference block:")
with GradientContext():
    print("  Running model forward pass...")
    output = [0.2, 0.7, 0.1]  # fake logits
    print("  Output:", output)
# After the with block: gradients restored automatically

# Using contextlib — the Pythonic way to build context managers
from contextlib import contextmanager

@contextmanager
def mixed_precision():
    """Simulates torch.cuda.amp.autocast() for BF16/FP16 training."""
    print("  >> Switched to FP16 precision (faster, less memory)")
    yield   # code inside 'with' block runs here
    print("  >> Switched back to FP32 precision")

print("\nTraining step with mixed precision:")
with mixed_precision():
    print("  Forward pass in FP16...")
    print("  Loss computed in FP32...")

# Real usage pattern you must know:
print("\nReal PyTorch inference pattern (memorize this):")
print("""
  with torch.no_grad():                  # disable gradient graph
      with torch.cuda.amp.autocast():    # use FP16 (optional, for speed)
          outputs = model(**inputs)
          logits = outputs.logits
""")


# --------------------------------------------------------------
# CONCEPT 5: Type Hints — Production Code Standard
# --------------------------------------------------------------
# THE PROBLEM:
#   In production ML code, functions handle tensors, arrays, strings.
#   Without type hints: impossible to know what goes in / comes out.
#   With type hints: self-documenting, IDE autocomplete, fewer bugs.
#
# THEORY:
#   Python 3.5+ supports type hints via the `typing` module.
#   They don't enforce types at runtime — but tools like mypy do.
#   FastAPI uses them to auto-generate API docs and validate inputs.
#
# INTERVIEW ANSWER:
#   "Type hints are essential in production ML code. FastAPI uses them
#    for request validation and auto-generated API docs. They also
#    enable static analysis tools like mypy to catch bugs before runtime."
# --------------------------------------------------------------

print("\n" + "=" * 55)
print("CONCEPT 5: Type Hints")
print("=" * 55)

from typing import List, Dict, Optional, Tuple
import numpy as np

# Production-level typed function signatures
def embed_texts(
    texts: List[str],
    model_name: str = "BAAI/bge-large-en-v1.5",
    normalize: bool = True
) -> np.ndarray:
    """
    Embed a list of texts to dense vectors.

    Args:
        texts:      List of input strings
        model_name: HuggingFace model to use for embedding
        normalize:  Whether to L2-normalize output vectors (needed for cosine sim)

    Returns:
        np.ndarray of shape (len(texts), embedding_dim)
    """
    # Simulated embeddings (in real code: model.encode(texts))
    dim = 4
    embeddings = np.random.randn(len(texts), dim)
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
    return embeddings

def retrieve_top_k(
    query: str,
    documents: List[str],
    k: int = 3
) -> List[Tuple[str, float]]:
    """
    Simple RAG retrieval: returns top-k (document, score) pairs.

    Returns:
        List of (document_text, similarity_score) tuples
    """
    query_vec = embed_texts([query])[0]
    doc_vecs  = embed_texts(documents)
    scores    = doc_vecs @ query_vec          # cosine sim (normalized)
    top_k_idx = np.argsort(scores)[::-1][:k] # sort descending
    return [(documents[i], float(scores[i])) for i in top_k_idx]

docs = [
    "LLaMA is an open-source large language model by Meta.",
    "RAG combines retrieval with generation for grounded answers.",
    "The Eiffel Tower is in Paris, France.",
    "LoRA fine-tunes models with low-rank adapter matrices.",
]

results = retrieve_top_k("how do you fine-tune an LLM cheaply?", docs, k=2)
print("RAG retrieval results:")
for doc, score in results:
    print(f"  [{score:.3f}] {doc}")


# --------------------------------------------------------------
# CONCEPT 6: Async / Await — Why LLM APIs Must Be Async
# --------------------------------------------------------------
# THE PROBLEM:
#   An LLM call takes 2-10 seconds. If your server handles one
#   request at a time (sync), 100 users = 100-1000 second wait.
#   Async: while waiting for LLM, handle other requests.
#
# THEORY:
#   - Synchronous: do task A, wait, then task B (sequential)
#   - Asynchronous: start task A, while waiting start task B
#   - async def → coroutine function
#   - await → "pause here and let other tasks run"
#   - asyncio.gather() → run multiple coroutines in parallel
#
# INTERVIEW ANSWER:
#   "LLM API calls are I/O bound — the model compute is remote.
#    Async endpoints handle many concurrent requests on a single thread.
#    FastAPI is async-native. We use async def endpoints + await for
#    any I/O: LLM calls, database queries, embedding API calls."
# --------------------------------------------------------------

print("\n" + "=" * 55)
print("CONCEPT 6: Async / Await")
print("=" * 55)

import asyncio

# Simulated async LLM call
async def call_llm(prompt: str, delay: float = 0.1) -> str:
    """Simulates an async LLM API call (network I/O)."""
    await asyncio.sleep(delay)   # non-blocking wait (simulates network)
    return f"Response to: '{prompt[:20]}...'"

async def handle_batch(prompts: List[str]) -> List[str]:
    """
    Process multiple prompts CONCURRENTLY.
    asyncio.gather() runs all coroutines at the same time.
    Total time ≈ max(individual times), not sum.
    """
    tasks = [call_llm(p, delay=0.05) for p in prompts]
    results = await asyncio.gather(*tasks)  # all run simultaneously!
    return list(results)

async def demo():
    prompts = [
        "Explain transformer attention",
        "What is LoRA?",
        "How does RAG work?",
    ]

    import time
    start = time.time()
    results = await handle_batch(prompts)
    elapsed = time.time() - start

    print(f"Processed {len(prompts)} prompts in {elapsed:.3f}s (concurrently!)")
    for r in results:
        print(f"  {r}")

asyncio.run(demo())

# FastAPI pattern — memorize this
print("\nFastAPI async endpoint pattern (memorize):")
print("""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7

@app.post("/v1/generate")
async def generate(request: GenerateRequest):
    # Non-blocking: server handles other requests while waiting
    result = await llm_client.generate(
        prompt=request.prompt,
        max_tokens=request.max_tokens,
    )
    return {"text": result, "tokens_used": len(result.split())}
""")


# ==============================================================
# INTERVIEW CHEAT SHEET — Python ML Patterns
# ==============================================================
print("=" * 55)
print("INTERVIEW CHEAT SHEET")
print("=" * 55)
print("""
Generator      -> lazy eval, 1 item at a time, memory efficient
               -> DataLoader, batch processing

**kwargs       -> unpack dict as keyword args
               -> model(**inputs), trainer(**config)

Decorator      -> wrap function, add behavior (timing, logging)
               -> @torch.no_grad(), @app.post(), @timer

Context Mgr    -> guaranteed setup+teardown
               -> with torch.no_grad(): (inference)
               -> with torch.cuda.amp.autocast(): (mixed precision)

Type Hints     -> self-documenting, FastAPI validation
               -> List[str], np.ndarray, Optional[int]

Async/Await    -> concurrent I/O, handle many users at once
               -> async def endpoint, await llm_call()
               -> asyncio.gather() for parallel requests
""")
