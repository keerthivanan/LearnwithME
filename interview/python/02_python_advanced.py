"""
Python Advanced — Definitions + Code + Outputs
Run: python 02_python_advanced.py
Every section starts with WHAT IS IT? so you know the concept before seeing code.
"""

import asyncio
import threading
import concurrent.futures
from dataclasses import dataclass, field
from typing import List, Optional
import itertools
import heapq
import time
import sys


# ══════════════════════════════════════════════════════
# 1. DATACLASSES
# ══════════════════════════════════════════════════════
# WHAT IS A DATACLASS?
#   → A class that AUTO-GENERATES boilerplate methods for you:
#       __init__  (constructor)
#       __repr__  (readable string for print/debug)
#       __eq__    (== comparison compares all fields)
#   → You just declare the fields with type hints and defaults
#   → Much less code than a regular class for the same result
#
# KEY PARTS:
#   → @dataclass               : the decorator that does the magic
#   → field(default_factory=X) : for mutable defaults (list, dict)
#                                DON'T use = [] directly — shared across instances!
#   → __post_init__            : runs AFTER __init__ for validation/extra setup
#   → frozen=True              : makes the dataclass immutable (like a named tuple)
#
# WHEN TO USE:
#   → Config objects (model config, training config)
#   → Data containers (results, records, API responses)
#   → Anywhere you're writing a class just to hold data

print("=" * 55)
print("DATACLASSES")
print("=" * 55)

@dataclass
class MLModel:
    name:        str
    version:     str
    accuracy:    float
    tags:        List[str] = field(default_factory=list)   # mutable default → use field()
    is_deployed: bool      = False                         # immutable default → ok directly

    def __post_init__(self):     # runs right after __init__ — great for validation
        if not (0 <= self.accuracy <= 1):
            raise ValueError(f"accuracy must be 0–1, got {self.accuracy}")

    def deploy(self):
        self.is_deployed = True
        return f"{self.name} v{self.version} deployed!"


model = MLModel("ResNet50", "1.2", 0.95, tags=["vision", "classification"])
print(model)
# MLModel(name='ResNet50', version='1.2', accuracy=0.95, tags=['vision', 'classification'], is_deployed=False)
print(model.deploy())        # ResNet50 v1.2 deployed!
print(model.is_deployed)     # True

# __eq__ is auto-generated — compare by field values
m1 = MLModel("BERT", "2.0", 0.88)
m2 = MLModel("BERT", "2.0", 0.88)
print(m1 == m2)    # True  ← dataclass auto-generates __eq__

# FROZEN dataclass — immutable (fields can't be changed after creation)
@dataclass(frozen=True)
class Point:
    x: float
    y: float

p = Point(3.0, 4.0)
print(p)        # Point(x=3.0, y=4.0)
# p.x = 5.0   ← FrozenInstanceError! can't modify frozen dataclass


# ══════════════════════════════════════════════════════
# 2. ASYNC / AWAIT
# ══════════════════════════════════════════════════════
# WHAT IS ASYNC/AWAIT?
#   → A way to write CONCURRENT code that handles waiting efficiently
#   → Instead of blocking (sitting idle), async code YIELDS CONTROL
#     while waiting, letting other tasks run in the meantime
#
# THE PROBLEM IT SOLVES:
#   → Normal code: fetch URL1 (wait 1s) → fetch URL2 (wait 1s) → total: 2s
#   → Async code:  start URL1 + URL2 together → both finish in ~1s  ← FASTER
#
# KEY CONCEPTS:
#   → async def   : marks a function as a "coroutine" (async-capable function)
#   → await       : pauses THIS coroutine and lets others run while waiting
#   → asyncio.run() : runs the async event loop (entry point)
#   → asyncio.gather() : run MULTIPLE coroutines concurrently, wait for ALL
#   → asyncio.create_task() : schedule coroutine to run, continue immediately
#
# ASYNC vs THREADING vs MULTIPROCESSING:
#   → Async         : I/O-bound, single thread, cooperative (yields voluntarily)
#   → Threading     : I/O-bound, multiple threads, preemptive (OS switches)
#   → Multiprocessing: CPU-bound, multiple processes, true parallelism
#
# WHEN TO USE ASYNC:
#   → Many I/O operations: HTTP requests, DB queries, file reads
#   → Chat servers, API servers, web scrapers
#   → FastAPI, aiohttp — both built on async

print("\n" + "=" * 55)
print("ASYNC / AWAIT")
print("=" * 55)

async def fetch_data(url: str, delay: float = 1.0) -> dict:
    print(f"  Started fetching: {url}")
    await asyncio.sleep(delay)    # PAUSE here — let other coroutines run
    print(f"  Done fetching:    {url}")
    return {"url": url, "data": f"result from {url}"}

async def fetch_all_sequential(urls: List[str]) -> List[dict]:
    results = []
    for url in urls:
        result = await fetch_data(url)   # waits for EACH one — total: N seconds
        results.append(result)
    return results

async def fetch_all_concurrent(urls: List[str]) -> List[dict]:
    tasks   = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)   # ALL run at once! — total: ~1s not N*1s
    return results

# Compare sequential vs concurrent
urls = ["api/users", "api/orders", "api/products"]

print("\n  [Sequential — waits for each]")
start = time.perf_counter()
results = asyncio.run(fetch_all_sequential(urls))
print(f"  Sequential time: {time.perf_counter()-start:.2f}s")
# Started fetching: api/users
# Done fetching: api/users
# Started fetching: api/orders ... (one at a time)
# Sequential time: ~3.0s

print("\n  [Concurrent — all at once]")
start = time.perf_counter()
results = asyncio.run(fetch_all_concurrent(urls))
print(f"  Concurrent time: {time.perf_counter()-start:.2f}s")
# Started fetching: api/users
# Started fetching: api/orders   ← all start immediately
# Started fetching: api/products
# Done fetching: api/users (all finish ~same time)
# Concurrent time: ~1.0s  ← 3x faster!

# asyncio.create_task — fire and forget, run in background
async def background_demo():
    async def worker(name, delay):
        await asyncio.sleep(delay)
        return f"{name} done"

    task1 = asyncio.create_task(worker("A", 0.2))  # starts immediately
    task2 = asyncio.create_task(worker("B", 0.1))  # starts immediately
    # do other work here while tasks run...
    result2 = await task2   # wait for task2
    result1 = await task1   # wait for task1
    print(f"  task1: {result1}, task2: {result2}")

asyncio.run(background_demo())
# task1: A done, task2: B done


# ══════════════════════════════════════════════════════
# 3. THREADING
# ══════════════════════════════════════════════════════
# WHAT IS THREADING?
#   → Running multiple threads (lightweight sub-processes) WITHIN one program
#   → Threads share the SAME memory — can access same variables
#   → Good for I/O-bound tasks (waiting on network, disk, DB)
#
# PYTHON'S GIL (Global Interpreter Lock):
#   → Python only lets ONE thread run Python code at a time
#   → So threads can't speed up CPU-heavy work (use multiprocessing for that)
#   → BUT threads CAN overlap on I/O — while one waits, another runs
#   → This is why threading works well for: HTTP requests, file I/O, DB calls
#
# ThreadPoolExecutor — easiest way to use threads:
#   → executor.submit(fn, arg)  : submit ONE task
#   → executor.map(fn, items)   : submit MANY tasks (like map() but threaded)
#   → as_completed(futures)     : process results as they finish (not in order)
#
# WHEN TO USE THREADING vs ASYNC:
#   → Threading: existing blocking libraries (requests, psycopg2, boto3)
#   → Async:     libraries built for async (aiohttp, asyncpg)

print("\n" + "=" * 55)
print("THREADING")
print("=" * 55)

def download(url: str) -> str:
    print(f"  Downloading {url}...")
    time.sleep(0.2)                  # simulate I/O wait (network call)
    return f"data from {url}"

# WITHOUT threading — sequential, each waits for previous
start = time.perf_counter()
results_seq = [download(url) for url in ["url1", "url2", "url3"]]
print(f"  Sequential: {time.perf_counter()-start:.2f}s")   # ~0.6s

# WITH ThreadPoolExecutor — all run in parallel threads
start = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    # submit() returns a Future — result available later
    futures = [executor.submit(download, url) for url in ["url1", "url2", "url3"]]
    # as_completed() yields futures as they FINISH (not submission order)
    results_threaded = [f.result() for f in concurrent.futures.as_completed(futures)]
print(f"  Threaded:   {time.perf_counter()-start:.2f}s")   # ~0.2s  ← 3x faster

# executor.map() — simpler: applies function to each item, returns in ORDER
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results_map = list(executor.map(download, ["url1", "url2", "url3"]))
print(results_map)   # ['data from url1', 'data from url2', 'data from url3']

# Manual Thread — when you need more control
def worker(name, result_list):
    time.sleep(0.1)
    result_list.append(f"{name} done")

results_manual = []
threads = [threading.Thread(target=worker, args=(f"T{i}", results_manual)) for i in range(3)]
for t in threads: t.start()
for t in threads: t.join()    # wait for ALL threads to finish
print(results_manual)         # ['T0 done', 'T1 done', 'T2 done'] (order may vary)


# ══════════════════════════════════════════════════════
# 4. MULTIPROCESSING
# ══════════════════════════════════════════════════════
# WHAT IS MULTIPROCESSING?
#   → Runs multiple PROCESSES (separate Python interpreters) in parallel
#   → Each process has its OWN memory and its OWN GIL
#   → TRUE parallelism — can use multiple CPU cores at the same time
#
# MULTIPROCESSING vs THREADING:
#   → Threading: same memory, limited by GIL, good for I/O-bound
#   → Multiprocessing: separate memory, no GIL, good for CPU-bound
#   → CPU-bound = heavy computation (training models, image processing, math)
#
# Pool — easiest way to use multiprocessing:
#   → pool.map(fn, items)   : like map() but across multiple processes
#   → pool.starmap(fn, args): like map() but unpacks tuples as args
#   → pool.apply_async()    : non-blocking submit
#
# ProcessPoolExecutor — modern alternative (same as ThreadPoolExecutor API):
#   → Part of concurrent.futures (consistent API with threading)
#
# CAUTION on Windows:
#   → Multiprocessing code MUST be inside if __name__ == "__main__":
#   → Otherwise it spawns infinite child processes

print("\n" + "=" * 55)
print("MULTIPROCESSING")
print("=" * 55)

def cpu_heavy(n: int) -> int:
    return sum(i**2 for i in range(n))   # lots of CPU math

# WITH Pool — distributes work across 4 CPU cores
# (Must be inside if __name__ == "__main__" on Windows)
#
# from multiprocessing import Pool
# with Pool(processes=4) as pool:
#     results = pool.map(cpu_heavy, [100000, 200000, 300000, 400000])
#     print(results)   # [333328333350000, ...] — computed in parallel

# WITH ProcessPoolExecutor — same API as ThreadPoolExecutor (consistent interface)
#
# from concurrent.futures import ProcessPoolExecutor
# with ProcessPoolExecutor(max_workers=4) as executor:
#     results = list(executor.map(cpu_heavy, [10000, 20000, 30000]))
#     print(results)

# QUICK RULE TO REMEMBER:
print("""
  WHICH ONE TO USE?
  ─────────────────────────────────────────
  I/O-bound  (HTTP, DB, files)  → Threading or Async
  CPU-bound  (math, ML, image)  → Multiprocessing
  Simple async I/O              → asyncio (fastest, least overhead)
  ─────────────────────────────────────────
""")


# ══════════════════════════════════════════════════════
# 5. ITERTOOLS
# ══════════════════════════════════════════════════════
# WHAT IS ITERTOOLS?
#   → A built-in module of functions for EFFICIENT looping and combinatorics
#   → All return ITERATORS (lazy — compute on demand, memory efficient)
#   → Faster than writing equivalent loops manually
#
# KEY FUNCTIONS:
#   → chain(*iterables)       : combine multiple iterables into one stream
#   → chain.from_iterable()   : like chain() but takes one nested iterable
#   → combinations(it, r)     : all unique r-length groups (order doesn't matter)
#   → permutations(it, r)     : all ordered r-length arrangements
#   → product(*its, repeat=n) : cartesian product (like nested for loops)
#   → groupby(it, key)        : group consecutive items by key (MUST sort first!)
#   → islice(it, start, stop) : slice an iterator (like list slicing)
#   → accumulate(it, fn)      : running total/product/etc.
#   → cycle(it)               : repeat an iterable forever
#   → repeat(x, n)            : repeat a value n times

print("\n" + "=" * 55)
print("ITERTOOLS")
print("=" * 55)

# CHAIN — combine multiple iterables into one flat sequence
chained = list(itertools.chain([1, 2], [3, 4], [5, 6]))
print(chained)    # [1, 2, 3, 4, 5, 6]

nested_list = [[1, 2], [3, 4], [5, 6]]
flat = list(itertools.chain.from_iterable(nested_list))
print(flat)       # [1, 2, 3, 4, 5, 6]  ← flattens one level

# COMBINATIONS — unique groups, ORDER DOESN'T MATTER
# (1,2) and (2,1) are the SAME combination → only (1,2) included
combs = list(itertools.combinations([1, 2, 3, 4], 2))
print(combs)      # [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]

# PERMUTATIONS — ordered arrangements, ORDER MATTERS
# (1,2) and (2,1) are DIFFERENT permutations → both included
perms = list(itertools.permutations([1, 2, 3], 2))
print(perms)      # [(1,2),(1,3),(2,1),(2,3),(3,1),(3,2)]

# PRODUCT — cartesian product (like nested for loops)
grid = list(itertools.product([0, 1], repeat=3))
print(grid)
# [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]

# GROUPBY — group consecutive elements by a key (MUST sort first!)
# If not sorted, same values in different positions = separate groups
data = [
    {"name": "Alice", "dept": "Eng"},
    {"name": "Bob",   "dept": "Eng"},
    {"name": "Carol", "dept": "HR"},
    {"name": "Dave",  "dept": "HR"},
]
data.sort(key=lambda x: x["dept"])   # MUST sort by the groupby key first!

for dept, group in itertools.groupby(data, key=lambda x: x["dept"]):
    members = [p["name"] for p in group]
    print(f"  {dept}: {members}")
# Eng: ['Alice', 'Bob']
# HR:  ['Carol', 'Dave']

# ISLICE — slice an iterator (like list[start:stop] but for iterators)
first_5 = list(itertools.islice(range(100), 5))
print(first_5)    # [0, 1, 2, 3, 4]

skip_2_take_5 = list(itertools.islice(range(100), 2, 7))
print(skip_2_take_5)   # [2, 3, 4, 5, 6]

# ACCUMULATE — running totals
import operator
running_sum     = list(itertools.accumulate([1, 2, 3, 4, 5]))
running_product = list(itertools.accumulate([1, 2, 3, 4, 5], operator.mul))
print(running_sum)      # [1, 3, 6, 10, 15]
print(running_product)  # [1, 2, 6, 24, 120]

# BATCHED — split iterable into chunks (manual, since batched is Python 3.12)
def batched(iterable, n):
    it = iter(iterable)
    while batch := list(itertools.islice(
        it, n)):
        yield batch

for batch in batched(range(10), 3):
    print(f"  batch: {batch}")
# batch: [0, 1, 2]
# batch: [3, 4, 5]
# batch: [6, 7, 8]
# batch: [9]


# ══════════════════════════════════════════════════════
# 6. HEAPQ (Priority Queue)
# ══════════════════════════════════════════════════════
# WHAT IS A HEAP?
#   → A data structure where the SMALLEST item is always at the front
#   → Python's heapq is a MIN-HEAP (smallest = index 0)
#   → Efficient: push/pop is O(log n), peek smallest is O(1)
#   → Used for: priority queues, finding top-K items, scheduling
#
# KEY FUNCTIONS:
#   → heapq.heapify(list)         : convert list into heap IN PLACE  O(n)
#   → heapq.heappush(heap, item)  : add item, maintains heap order   O(log n)
#   → heapq.heappop(heap)         : remove + return SMALLEST item    O(log n)
#   → heapq.nlargest(k, iterable) : top K largest items              O(n log k)
#   → heapq.nsmallest(k, iterable): top K smallest items             O(n log k)
#
# MAX-HEAP trick: negate values (push -x, pop and negate result)
#
# WHEN TO USE:
#   → Find K-th largest/smallest element
#   → Merge K sorted lists
#   → Dijkstra's shortest path algorithm
#   → Task scheduling by priority

print("\n" + "=" * 55)
print("HEAPQ (Priority Queue)")
print("=" * 55)

# Build a heap from a list
heap = [5, 3, 1, 4, 2]
heapq.heapify(heap)          # rearranges IN PLACE into heap order
print(heap)                  # [1, 2, 3, 4, 5]  ← smallest at index 0 (not fully sorted!)
print(heap[0])               # 1  ← peek smallest (O(1))

heapq.heappush(heap, 0)      # insert 0 — heap auto-adjusts
print(heap[0])               # 0  ← new smallest

smallest = heapq.heappop(heap)   # removes and returns smallest
print(smallest)              # 0
print(heap[0])               # 1  ← next smallest is back at front

# TOP K elements
data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
top_3    = heapq.nlargest(3, data)    # [9, 6, 5]
bottom_3 = heapq.nsmallest(3, data)  # [1, 1, 2]
print(top_3)     # [9, 6, 5]
print(bottom_3)  # [1, 1, 2]

# Top K with custom key — e.g., top 3 students by score
students = [
    {"name": "Alice",   "score": 95},
    {"name": "Bob",     "score": 82},
    {"name": "Charlie", "score": 91},
    {"name": "Diana",   "score": 88},
]
top_2 = heapq.nlargest(2, students, key=lambda s: s["score"])
print([s["name"] for s in top_2])   # ['Alice', 'Charlie']

# PRIORITY QUEUE — push (priority, item), lower number = higher priority
task_queue = []
heapq.heappush(task_queue, (3, "low priority task"))
heapq.heappush(task_queue, (1, "urgent task"))
heapq.heappush(task_queue, (2, "medium task"))

while task_queue:
    priority, task = heapq.heappop(task_queue)
    print(f"  [{priority}] {task}")
# [1] urgent task
# [2] medium task
# [3] low priority task


# ══════════════════════════════════════════════════════
# 7. DESIGN PATTERNS
# ══════════════════════════════════════════════════════
# WHAT ARE DESIGN PATTERNS?
#   → Proven, reusable solutions to COMMON software design problems
#   → Not code — they are TEMPLATES you adapt to your situation
#
# SINGLETON — "only ONE instance of this class should ever exist"
#   → Overrides __new__ to return the same instance every time
#   → Use for: config managers, DB connection pools, loggers
#
# FACTORY — "create objects WITHOUT specifying their exact class"
#   → A method/class that decides WHICH object to create
#   → Decouples object creation from usage
#   → Use for: ML model registry, plugin systems, parsers
#
# OBSERVER / EVENT BUS — "notify all subscribers when something happens"
#   → Publishers emit events, subscribers react
#   → Decouples the producer from the consumers
#   → Use for: model training callbacks, UI updates, logging

print("\n" + "=" * 55)
print("DESIGN PATTERNS")
print("=" * 55)

# ── SINGLETON ──────────────────────────────────────────
# __new__ is called BEFORE __init__ — it creates the object
# We intercept it to return the same object every time
print("  [Singleton]")

class Config:
    _instance = None   # class variable holds the one instance

    def __new__(cls):
        if cls._instance is None:                    # first call → create
            cls._instance = super().__new__(cls)
            cls._instance.settings = {}
        return cls._instance                         # all calls → return same object

    def set(self, key, value): self.settings[key] = value
    def get(self, key, default=None): return self.settings.get(key, default)


c1 = Config()
c2 = Config()
c1.set("lr", 0.001)
print(f"  c2.get('lr') = {c2.get('lr')}")   # 0.001 ← same object!
print(f"  c1 is c2: {c1 is c2}")            # True  ← literally the same object

# ── FACTORY ────────────────────────────────────────────
# Registry: map names → classes
# register() decorator: annotate a class to auto-add it to registry
# create() method: look up name, instantiate with kwargs
print("\n  [Factory Pattern]")

class ModelFactory:
    _registry = {}

    @classmethod
    def register(cls, name):          # use as @ModelFactory.register("linear")
        def decorator(model_class):
            cls._registry[name] = model_class
            return model_class
        return decorator

    @classmethod
    def create(cls, name, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Unknown model: {name}. Available: {list(cls._registry)}")
        return cls._registry[name](**kwargs)

    @classmethod
    def list_models(cls):
        return list(cls._registry.keys())


@ModelFactory.register("linear")       # registers LinearModel under name "linear"
class LinearModel:
    def __init__(self, lr=0.01): self.lr = lr
    def __repr__(self): return f"LinearModel(lr={self.lr})"

@ModelFactory.register("forest")
class ForestModel:
    def __init__(self, n_trees=100): self.n_trees = n_trees
    def __repr__(self): return f"ForestModel(n_trees={self.n_trees})"

model = ModelFactory.create("linear", lr=0.001)
print(f"  Created: {model}")              # LinearModel(lr=0.001)
print(f"  Available: {ModelFactory.list_models()}")  # ['linear', 'forest']

# ── OBSERVER / EVENT BUS ───────────────────────────────
# subscribe(event, handler): register a callback for an event
# publish(event, **data):    call ALL registered handlers for that event
print("\n  [Observer / Event Bus]")

class EventBus:
    def __init__(self):
        self._subscribers = {}   # event_name → [list of handler functions]

    def subscribe(self, event: str, handler):
        self._subscribers.setdefault(event, []).append(handler)

    def publish(self, event: str, **data):
        handlers = self._subscribers.get(event, [])
        for handler in handlers:
            handler(**data)   # call each subscriber with the event data


bus = EventBus()

# Register subscribers (completely decoupled from each other)
bus.subscribe("model_trained", lambda **d: print(f"  Logger:  accuracy={d['accuracy']}"))
bus.subscribe("model_trained", lambda **d: print(f"  Alerter: saving to {d.get('path','default')}"))
bus.subscribe("model_trained", lambda **d: print(f"  Slack:   model done! acc={d['accuracy']}"))

# Publish event — all 3 handlers run automatically
bus.publish("model_trained", accuracy=0.95, path="models/v1.pkl")
# Logger:  accuracy=0.95
# Alerter: saving to models/v1.pkl
# Slack:   model done! acc=0.95


# ══════════════════════════════════════════════════════
# 8. PROTOCOL & DUCK TYPING
# ══════════════════════════════════════════════════════
# WHAT IS DUCK TYPING?
#   → "If it walks like a duck and quacks like a duck, it IS a duck"
#   → Python doesn't care about the TYPE — only that the object
#     has the METHODS/ATTRIBUTES you need
#   → No need for classes to inherit from a base class
#     as long as they have the right methods
#
# WHAT IS A PROTOCOL?
#   → A formal way to define "structural subtyping" in Python
#   → Protocol says: "any class that has THESE methods qualifies"
#   → Unlike ABC (Abstract Base Class), no inheritance required
#   → @runtime_checkable: makes isinstance() checks work at runtime
#
# PROTOCOL vs ABC:
#   → ABC (inherit from it): class Dog(Animal): — explicit opt-in
#   → Protocol (structural): if Dog has .speak() and .eat() → it qualifies
#   → Protocol is more flexible — works with existing classes

print("\n" + "=" * 55)
print("PROTOCOL & DUCK TYPING")
print("=" * 55)

from typing import Protocol, runtime_checkable

# Define WHAT METHODS something must have (not HOW they're implemented)
@runtime_checkable
class Trainable(Protocol):
    def fit(self, X: list, y: list) -> None: ...    # "..." means not implemented here
    def predict(self, X: list) -> list: ...

# These classes DON'T inherit from Trainable — they just implement the methods
class SklearnModel:
    def fit(self, X, y):
        print(f"  SklearnModel fitting on {len(X)} samples")

    def predict(self, X):
        return [0] * len(X)

class MyCustomModel:
    def fit(self, X, y):
        print(f"  MyCustomModel fitting on {len(X)} samples")

    def predict(self, X):
        return [1] * len(X)

# Function accepts ANY object that satisfies the Protocol
def train_and_evaluate(model: Trainable, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"  Predictions: {predictions}")
    return predictions

X_train = [[1], [2], [3]]
y_train = [0, 1, 0]
X_test  = [[4], [5]]

train_and_evaluate(SklearnModel(), X_train, y_train, X_test)
# SklearnModel fitting on 3 samples
# Predictions: [0, 0]

train_and_evaluate(MyCustomModel(), X_train, y_train, X_test)
# MyCustomModel fitting on 3 samples
# Predictions: [1, 1]

# isinstance() works because of @runtime_checkable
print(isinstance(SklearnModel(), Trainable))    # True  ← has fit() and predict()
print(isinstance("hello", Trainable))           # False ← str has no fit/predict


# ══════════════════════════════════════════════════════
# 9. MEMORY PROFILING
# ══════════════════════════════════════════════════════
# WHAT IS MEMORY PROFILING?
#   → Understanding how much RAM your data structures use
#   → Helps choose the right data structure for large-scale work
#
# KEY INSIGHT — Python object sizes:
#   → int           :  28 bytes  (yes, even a simple number has overhead!)
#   → list of 1000  : ~8,056 bytes
#   → tuple of 1000 : ~8,040 bytes  (slightly smaller than list)
#   → generator     :  ~200 bytes   (doesn't store data at all!)
#   → dict          :  ~200+ bytes  (grows with entries)
#
# sys.getsizeof(obj) — returns size in bytes of the object ITSELF
#   → Doesn't include size of objects IT CONTAINS (shallow)
#   → For true deep size, need a recursive approach
#
# MEMORY OPTIMIZATION TRICKS:
#   → Use generators instead of lists for large sequences
#   → Use tuples instead of lists for immutable data (slightly smaller)
#   → Use __slots__ in classes to avoid per-instance __dict__
#   → Use numpy arrays instead of Python lists for numbers (10-50x smaller)

print("\n" + "=" * 55)
print("MEMORY PROFILING")
print("=" * 55)

# Compare sizes of different structures
small_list  = [1] * 100
small_tuple = (1,) * 100
list_large  = [x**2 for x in range(10000)]
gen_large   = (x**2 for x in range(10000))

print(f"  list  100 items: {sys.getsizeof(small_list):,} bytes")   # 856 bytes
print(f"  tuple 100 items: {sys.getsizeof(small_tuple):,} bytes")  # 840 bytes
print(f"  list 10000 sq:   {sys.getsizeof(list_large):,} bytes")   # ~85,000 bytes
print(f"  generator 10000: {sys.getsizeof(gen_large):,} bytes")    # ~200 bytes!

# __slots__ — class memory optimization
# Normal class: each instance has a __dict__ (~200 bytes overhead)
# __slots__:   no __dict__ — stores attrs directly (much smaller)

class NormalPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class SlottedPoint:
    __slots__ = ["x", "y"]   # declare allowed attributes — no __dict__ created
    def __init__(self, x, y):
        self.x = x
        self.y = y

np_obj = NormalPoint(1, 2)
sp_obj = SlottedPoint(1, 2)
print(f"  NormalPoint  size: {sys.getsizeof(np_obj)} bytes")   # 48 bytes + __dict__
print(f"  SlottedPoint size: {sys.getsizeof(sp_obj)} bytes")   # 32 bytes  ← smaller

# With 1 million objects this difference is massive:
# NormalPoint  × 1M = ~256 MB
# SlottedPoint × 1M = ~56 MB   ← 4x less memory!


print("\n" + "=" * 55)
print("All done! ✓")
print("=" * 55)
