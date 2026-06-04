"""
Python Advanced — Concurrency, Async, Design Patterns, Dataclasses
"""

import asyncio
import threading
import concurrent.futures
from dataclasses import dataclass, field
from typing import List, Optional
import itertools
import heapq

# ── DATACLASSES ───────────────────────────────────────
@dataclass
class MLModel:
    name: str
    version: str
    accuracy: float
    tags: List[str] = field(default_factory=list)
    is_deployed: bool = False

    def __post_init__(self):
        if self.accuracy < 0 or self.accuracy > 1:
            raise ValueError("accuracy must be between 0 and 1")

    def deploy(self):
        self.is_deployed = True
        return f"{self.name} v{self.version} deployed!"


model = MLModel("ResNet50", "1.2", 0.95, tags=["vision", "classification"])
print(model)
print(model.deploy())

# ── ASYNC / AWAIT ─────────────────────────────────────
async def fetch_data(url: str) -> dict:
    await asyncio.sleep(1)  # simulate network call
    return {"url": url, "data": "some data"}

async def fetch_all(urls: List[str]) -> List[dict]:
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)  # run concurrently!
    return results

# Run
urls = ["http://api1.com", "http://api2.com", "http://api3.com"]
# results = asyncio.run(fetch_all(urls))   # all 3 run in ~1s, not 3s

# Real async HTTP (with aiohttp)
async def real_fetch(session, url):
    async with session.get(url) as response:
        return await response.json()

# ── THREADING ─────────────────────────────────────────
import time

def download(url):
    print(f"Downloading {url}")
    time.sleep(2)  # simulate I/O
    print(f"Done: {url}")
    return f"data from {url}"

# Thread pool for I/O-bound tasks
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    urls = ["url1", "url2", "url3", "url4"]
    futures = [executor.submit(download, url) for url in urls]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]

# ── MULTIPROCESSING ───────────────────────────────────
from multiprocessing import Pool

def cpu_heavy(n):
    return sum(i**2 for i in range(n))

# Process pool for CPU-bound tasks
with Pool(processes=4) as pool:
    results = pool.map(cpu_heavy, [100000, 200000, 300000, 400000])

# ── ITERTOOLS ─────────────────────────────────────────
data = [1, 2, 3, 4, 5]

# Chain multiple iterables
chained = list(itertools.chain([1, 2], [3, 4], [5, 6]))

# Combinations and permutations
combs = list(itertools.combinations([1, 2, 3], 2))  # (1,2),(1,3),(2,3)
perms = list(itertools.permutations([1, 2, 3], 2))

# GroupBy
data = [{"name": "Alice", "dept": "Eng"}, {"name": "Bob", "dept": "Eng"},
        {"name": "Charlie", "dept": "HR"}]
data.sort(key=lambda x: x["dept"])

for dept, group in itertools.groupby(data, key=lambda x: x["dept"]):
    print(dept, list(group))

# Batched (Python 3.12) or manual
def batched(iterable, n):
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch

for batch in batched(range(10), 3):
    print(batch)

# ── HEAPQ ─────────────────────────────────────────────
heap = [5, 3, 1, 4, 2]
heapq.heapify(heap)

heapq.heappush(heap, 0)
smallest = heapq.heappop(heap)  # 0

# Top K elements
data = [3, 1, 4, 1, 5, 9, 2, 6]
top_3    = heapq.nlargest(3, data)   # [9, 6, 5]
bottom_3 = heapq.nsmallest(3, data)  # [1, 1, 2]

# ── DESIGN PATTERNS ───────────────────────────────────

# Singleton
class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.settings = {}
        return cls._instance

    def set(self, key, value):
        self.settings[key] = value

    def get(self, key, default=None):
        return self.settings.get(key, default)


c1 = Config()
c2 = Config()
c1.set("lr", 0.01)
print(c2.get("lr"))  # 0.01 — same instance!

# Factory Pattern
class ModelFactory:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(model_class):
            cls._registry[name] = model_class
            return model_class
        return decorator

    @classmethod
    def create(cls, name, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Unknown model: {name}")
        return cls._registry[name](**kwargs)


@ModelFactory.register("linear")
class LinearModel:
    def __init__(self, lr=0.01):
        self.lr = lr
    def fit(self, X, y): pass

@ModelFactory.register("forest")
class ForestModel:
    def __init__(self, n_trees=100):
        self.n_trees = n_trees
    def fit(self, X, y): pass

model = ModelFactory.create("linear", lr=0.001)

# Observer Pattern (Event System)
class EventBus:
    def __init__(self):
        self._subscribers = {}

    def subscribe(self, event, handler):
        self._subscribers.setdefault(event, []).append(handler)

    def publish(self, event, **data):
        for handler in self._subscribers.get(event, []):
            handler(**data)


bus = EventBus()
bus.subscribe("model_trained", lambda **d: print(f"Model trained! Accuracy: {d['accuracy']}"))
bus.subscribe("model_trained", lambda **d: print(f"Saving model to {d.get('path', 'default')}"))

bus.publish("model_trained", accuracy=0.95, path="models/v1.pkl")

# ── PROTOCOL / DUCK TYPING ────────────────────────────
from typing import Protocol, runtime_checkable

@runtime_checkable
class Trainable(Protocol):
    def fit(self, X, y) -> None: ...
    def predict(self, X) -> list: ...

class MyModel:
    def fit(self, X, y): pass
    def predict(self, X): return []

def train_and_predict(model: Trainable, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    return model.predict(X_test)

# ── PROFILING & MEMORY ────────────────────────────────
import sys

# Memory size of objects
print(sys.getsizeof([1] * 1000))      # list of 1000 ints
print(sys.getsizeof((1,) * 1000))     # tuple (smaller)

# Generator vs list memory
import sys
list_gen  = [x**2 for x in range(10000)]
gen_obj   = (x**2 for x in range(10000))
print(sys.getsizeof(list_gen))   # ~80K bytes
print(sys.getsizeof(gen_obj))    # ~200 bytes (generator object)
