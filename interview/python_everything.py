"""
Python Everything — With Outputs Shown
Run this file: python python_everything.py
"""

# ════════════════════════════════════════════════════
# 1. LISTS
# ════════════════════════════════════════════════════
print("=" * 50)
print("LISTS")
print("=" * 50)

fruits = ["apple", "banana", "cherry", "mango"]

print(fruits[0])          # apple
print(fruits[-1])         # mango      ← last item
print(fruits[1:3])        # ['banana', 'cherry']

fruits.append("grape")
print(fruits)             # ['apple', 'banana', 'cherry', 'mango', 'grape']

fruits.insert(1, "kiwi")
print(fruits)             # ['apple', 'kiwi', 'banana', 'cherry', 'mango', 'grape']

fruits.remove("kiwi")
popped = fruits.pop()     # removes + returns last item
print(popped)             # grape
print(fruits)             # ['apple', 'banana', 'cherry', 'mango']

nums = [3, 1, 4, 1, 5, 9, 2, 6]
print(sorted(nums))           # [1, 1, 2, 3, 4, 5, 6, 9]
print(sorted(nums, reverse=True))  # [9, 6, 5, 4, 3, 2, 1, 1]
print(len(nums))              # 8
print(nums.count(1))          # 2
print(max(nums), min(nums), sum(nums))  # 9 1 31

# List comprehension
squares = [x**2 for x in range(6)]
print(squares)              # [0, 1, 4, 9, 16, 25]

evens = [x for x in range(10) if x % 2 == 0]
print(evens)                # [0, 2, 4, 6, 8]

pairs = [(x, y) for x in [1, 2] for y in [3, 4]]
print(pairs)                # [(1,3),(1,4),(2,3),(2,4)]

# Unpack
first, *middle, last = [1, 2, 3, 4, 5]
print(first, middle, last)  # 1 [2, 3, 4] 5

# Flatten nested list
nested = [[1, 2], [3, 4], [5, 6]]
flat = [x for row in nested for x in row]
print(flat)                 # [1, 2, 3, 4, 5, 6]


# ════════════════════════════════════════════════════
# 2. DICTIONARIES
# ════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("DICTIONARIES")
print("=" * 50)

person = {"name": "Alice", "age": 25, "city": "Mumbai"}

print(person["name"])               # Alice
print(person.get("salary", 0))      # 0  ← default if key missing

person["email"] = "alice@gmail.com"
person.update({"age": 26, "country": "India"})
print(person)
# {'name':'Alice','age':26,'city':'Mumbai','email':'alice@gmail.com','country':'India'}

del person["country"]
removed = person.pop("city")
print(removed)                      # Mumbai

# Iterate
for key, value in person.items():
    print(f"  {key}: {value}")

# Dict comprehension
word_len = {word: len(word) for word in ["hello", "world", "python"]}
print(word_len)   # {'hello': 5, 'world': 5, 'python': 6}

squares = {x: x**2 for x in range(6)}
print(squares)    # {0:0, 1:1, 2:4, 3:9, 4:16, 5:25}

# Merge
defaults = {"color": "blue", "size": 10}
custom   = {"color": "red"}
merged   = defaults | custom         # Python 3.9+
print(merged)                        # {'color': 'red', 'size': 10}

# Nested
students = {
    "alice": {"grade": "A", "score": 95},
    "bob":   {"grade": "B", "score": 82},
}
print(students["alice"]["score"])   # 95

# Sort by value
sorted_by_score = sorted(students.items(), key=lambda x: x[1]["score"], reverse=True)
print(sorted_by_score)
# [('alice', {'grade': 'A', 'score': 95}), ('bob', {'grade': 'B', 'score': 82})]

# fromkeys
keys = ["x", "y", "z"]
zeroed = dict.fromkeys(keys, 0)
print(zeroed)   # {'x': 0, 'y': 0, 'z': 0}


# ════════════════════════════════════════════════════
# 3. SETS
# ════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("SETS")
print("=" * 50)

a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7}

print(a & b)    # {3, 4, 5}     intersection
print(a | b)    # {1,2,3,4,5,6,7}  union
print(a - b)    # {1, 2}        difference
print(a ^ b)    # {1, 2, 6, 7}  symmetric difference

a.add(10)
a.discard(99)   # no error if not present
print(a)        # {1, 2, 3, 4, 5, 10}

# Remove duplicates
data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
unique = list(dict.fromkeys(data))  # preserves order
print(unique)   # [3, 1, 4, 5, 9, 2, 6]


# ════════════════════════════════════════════════════
# 4. TUPLES
# ════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("TUPLES")
print("=" * 50)

point = (3, 4)
x, y = point
print(x, y)     # 3 4

rgb = (255, 128, 0)
r, g, b = rgb
print(r, g, b)  # 255 128 0

# Named tuple
from collections import namedtuple
Person = namedtuple("Person", ["name", "age", "city"])
alice  = Person("Alice", 25, "Mumbai")
print(alice.name, alice.age, alice.city)   # Alice 25 Mumbai
print(alice[0])                            # Alice  ← also index access


# ════════════════════════════════════════════════════
# 5. STRINGS
# ════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("STRINGS")
print("=" * 50)

s = "  Hello, World!  "
print(s.strip())              # "Hello, World!"
print(s.strip().lower())      # "hello, world!"
print(s.strip().upper())      # "HELLO, WORLD!"
print(s.strip().replace("World", "Python"))   # Hello, Python!
print(s.strip().split(", "))  # ['Hello', 'World!']
print("-".join(["a", "b", "c"]))  # a-b-c
print(s.strip().startswith("Hello"))  # True
print("Hello" in s)           # True
print(s.strip().count("l"))   # 3

# f-strings
name, score = "Alice", 95.678
print(f"Name: {name}, Score: {score:.2f}")   # Name: Alice, Score: 95.68
print(f"{'Left':<10}|{'Right':>10}")         # Left      |     Right
print(f"{12345:,}")                           # 12,345

# Regex
import re
text = "My phone: 9876543210 and email: alice@gmail.com"
phones = re.findall(r'\d{10}', text)
print(phones)    # ['9876543210']

emails = re.findall(r'[\w.]+@[\w.]+\.\w+', text)
print(emails)    # ['alice@gmail.com']

clean = re.sub(r'\s+', ' ', "too   many   spaces").strip()
print(clean)     # too many spaces


# ════════════════════════════════════════════════════
# 6. FUNCTIONS
# ════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("FUNCTIONS")
print("=" * 50)

def add(a, b):
    return a + b

print(add(3, 4))    # 7

# Default arguments
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

print(greet("Alice"))               # Hello, Alice!
print(greet("Bob", "Hi"))           # Hi, Bob!

# *args
def total(*args):
    return sum(args)

print(total(1, 2, 3, 4, 5))        # 15

# **kwargs
def display(**kwargs):
    return ", ".join(f"{k}={v}" for k, v in kwargs.items())

print(display(name="Alice", age=25))   # name=Alice, age=25

# Combined
def create_model(name, *layers, **params):
    print(f"Model: {name}, Layers: {layers}, Params: {params}")

create_model("ResNet", 64, 128, 256, lr=0.001, dropout=0.3)
# Model: ResNet, Layers: (64,128,256), Params: {'lr':0.001,'dropout':0.3}

# Lambda
square  = lambda x: x ** 2
is_even = lambda x: x % 2 == 0
print(square(5))        # 25
print(is_even(4))       # True

# Map / Filter / Reduce
from functools import reduce
nums    = [1, 2, 3, 4, 5]
doubled = list(map(lambda x: x * 2, nums))
evens   = list(filter(lambda x: x % 2 == 0, nums))
product = reduce(lambda x, y: x * y, nums)
print(doubled)   # [2, 4, 6, 8, 10]
print(evens)     # [2, 4]
print(product)   # 120

# Closure
def make_multiplier(factor):
    def multiply(x):
        return x * factor
    return multiply

double = make_multiplier(2)
triple = make_multiplier(3)
print(double(7))    # 14
print(triple(7))    # 21


# ════════════════════════════════════════════════════
# 7. DECORATORS
# ════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("DECORATORS")
print("=" * 50)

import time
import functools

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start  = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"  {func.__name__} took {time.perf_counter()-start:.4f}s")
        return result
    return wrapper

def logger(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"  Calling {func.__name__}({args}, {kwargs})")
        result = func(*args, **kwargs)
        print(f"  Returned: {result}")
        return result
    return wrapper

@timer
def slow_add(a, b):
    time.sleep(0.05)
    return a + b

result = slow_add(3, 4)
print(f"Result: {result}")
# slow_add took 0.0502s
# Result: 7

@logger
def multiply(a, b):
    return a * b

multiply(3, 4)
# Calling multiply((3, 4), {})
# Returned: 12

# Decorator with arguments
def repeat(times):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(times=3)
def say_hello():
    print("  Hello!")

say_hello()
# Hello!
# Hello!
# Hello!

# LRU Cache — memoization
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2: return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(30))    # 832040  (very fast with cache)
print(fibonacci.cache_info())
# CacheInfo(hits=28, misses=31, maxsize=128, currsize=31)


# ════════════════════════════════════════════════════
# 8. GENERATORS
# ════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("GENERATORS")
print("=" * 50)

def count_up(start, end):
    while start <= end:
        yield start
        start += 1

for n in count_up(1, 5):
    print(n, end=" ")   # 1 2 3 4 5
print()

def fibonacci_gen():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

fib = fibonacci_gen()
first_8 = [next(fib) for _ in range(8)]
print(first_8)   # [0, 1, 1, 2, 3, 5, 8, 13]

# Generator expression (memory efficient!)
import sys
list_obj = [x**2 for x in range(10000)]
gen_obj  = (x**2 for x in range(10000))
print(f"List: {sys.getsizeof(list_obj):,} bytes")   # List: 85,176 bytes
print(f"Generator: {sys.getsizeof(gen_obj)} bytes")  # Generator: 200 bytes

# Pipeline
def read_numbers():
    for i in range(10):
        yield i

def square_them(nums):
    for n in nums:
        yield n ** 2

def only_big(nums, threshold=20):
    for n in nums:
        if n > threshold:
            yield n

pipeline = only_big(square_them(read_numbers()))
print(list(pipeline))   # [25, 36, 49, 64, 81]


# ════════════════════════════════════════════════════
# 9. CLASSES & OOP
# ════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("CLASSES & OOP")
print("=" * 50)

class Animal:
    kingdom = "Animalia"    # class variable
    _count  = 0

    def __init__(self, name, sound):
        self.name  = name
        self.sound = sound
        Animal._count += 1

    def speak(self):
        return f"{self.name} says {self.sound}"

    @classmethod
    def get_count(cls):
        return cls._count

    @staticmethod
    def is_valid_name(name):
        return bool(name and name.isalpha())

    def __repr__(self):
        return f"Animal({self.name!r})"

    def __str__(self):
        return self.name

    def __len__(self):
        return len(self.name)

    def __eq__(self, other):
        return isinstance(other, Animal) and self.name == other.name


class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, sound="Woof")
        self.breed   = breed
        self._tricks = []

    def speak(self):
        return f"{self.name} ({self.breed}): Woof!"

    def learn(self, trick):
        self._tricks.append(trick)

    @property
    def tricks(self):
        return self._tricks.copy()

    @tricks.setter
    def tricks(self, new_tricks):
        if not isinstance(new_tricks, list):
            raise TypeError("Must be a list")
        self._tricks = new_tricks


cat = Animal("Whiskers", "Meow")
dog = Dog("Rex", "Labrador")

print(cat.speak())              # Whiskers says Meow
print(dog.speak())              # Rex (Labrador): Woof!
print(repr(cat))                # Animal('Whiskers')
print(str(cat))                 # Whiskers
print(len(cat))                 # 8
print(cat == Animal("Whiskers", "Purr"))   # True (same name)
print(Animal.get_count())       # 2
print(Animal.is_valid_name("Rex"))    # True
print(Animal.is_valid_name("Rex123")) # False

dog.learn("sit")
dog.learn("shake")
print(dog.tricks)               # ['sit', 'shake']
dog.tricks = ["roll over"]
print(dog.tricks)               # ['roll over']

# isinstance & issubclass
print(isinstance(dog, Animal))  # True
print(isinstance(dog, Dog))     # True
print(issubclass(Dog, Animal))  # True


# ════════════════════════════════════════════════════
# 10. DATACLASSES
# ════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("DATACLASSES")
print("=" * 50)

from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    name:          str
    learning_rate: float = 0.001
    epochs:        int   = 10
    layers:        List[int] = field(default_factory=lambda: [64, 128, 64])

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

    @property
    def depth(self):
        return len(self.layers)


cfg = Config("ResNet", learning_rate=0.01, epochs=50)
print(cfg)
# Config(name='ResNet', learning_rate=0.01, epochs=50, layers=[64, 128, 64])
print(cfg.depth)    # 3

cfg2 = Config("ResNet", learning_rate=0.01, epochs=50)
print(cfg == cfg2)  # True  ← auto __eq__


# ════════════════════════════════════════════════════
# 11. ERROR HANDLING
# ════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("ERROR HANDLING")
print("=" * 50)

class AppError(Exception):
    def __init__(self, code, message):
        self.code = code
        super().__init__(f"[{code}] {message}")

def divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        raise AppError("DIV_ZERO", "Cannot divide by zero")
    except TypeError as e:
        raise AppError("TYPE_ERR", str(e))
    else:
        print("  Division successful")
        return result
    finally:
        print("  divide() always runs this")

try:
    print(divide(10, 2))    # 5.0
    print(divide(10, 0))
except AppError as e:
    print(f"Error code: {e.code}, message: {e}")


# ════════════════════════════════════════════════════
# 12. CONTEXT MANAGERS
# ════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("CONTEXT MANAGERS")
print("=" * 50)

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        print("  Timer started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start
        print(f"  Timer stopped: {self.elapsed:.4f}s")
        return False

with Timer() as t:
    total = sum(range(1_000_000))

print(f"  Sum = {total:,}")
# Timer started
# Timer stopped: 0.0234s
# Sum = 499,999,500,000

# contextmanager shorthand
from contextlib import contextmanager

@contextmanager
def managed_db(name):
    print(f"  Connecting to {name}")
    try:
        yield {"db": name, "connected": True}
    finally:
        print(f"  Closing connection to {name}")

with managed_db("PostgreSQL") as conn:
    print(f"  Using: {conn}")
# Connecting to PostgreSQL
# Using: {'db': 'PostgreSQL', 'connected': True}
# Closing connection to PostgreSQL


# ════════════════════════════════════════════════════
# 13. COLLECTIONS MODULE
# ════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("COLLECTIONS")
print("=" * 50)

from collections import Counter, defaultdict, deque

# Counter
words   = "the quick brown fox jumps over the lazy dog the".split()
counter = Counter(words)
print(counter.most_common(3))    # [('the', 3), ('quick', 1), ('brown', 1)]
print(counter["the"])            # 3

# defaultdict
positions = defaultdict(list)
for i, word in enumerate(words):
    positions[word].append(i)
print(dict(positions["the"]))    # Error — it's a list, not dict
print(positions["the"])          # [0, 6, 9]

# deque (efficient queue with max length)
window = deque(maxlen=3)
for i in range(6):
    window.append(i)
    print(list(window))
# [0]
# [0, 1]
# [0, 1, 2]
# [1, 2, 3]  ← 0 dropped
# [2, 3, 4]
# [3, 4, 5]


# ════════════════════════════════════════════════════
# 14. SORTING TRICKS
# ════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("SORTING")
print("=" * 50)

people = [
    {"name": "Charlie", "age": 30, "score": 92},
    {"name": "Alice",   "age": 25, "score": 95},
    {"name": "Bob",     "age": 30, "score": 87},
]

by_score = sorted(people, key=lambda x: x["score"], reverse=True)
print([p["name"] for p in by_score])   # ['Alice', 'Charlie', 'Bob']

# Multi-key sort: age asc, score desc
multi = sorted(people, key=lambda x: (x["age"], -x["score"]))
print([p["name"] for p in multi])      # ['Alice', 'Charlie', 'Bob']

# Sort strings by length then alphabetical
words = ["banana", "apple", "cherry", "fig", "date"]
print(sorted(words, key=lambda x: (len(x), x)))
# ['fig', 'date', 'apple', 'banana', 'cherry']


# ════════════════════════════════════════════════════
# 15. ITERTOOLS
# ════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("ITERTOOLS")
print("=" * 50)

import itertools

print(list(itertools.chain([1,2], [3,4], [5,6])))
# [1, 2, 3, 4, 5, 6]

print(list(itertools.combinations([1,2,3,4], 2)))
# [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]

print(list(itertools.permutations([1,2,3], 2)))
# [(1,2),(1,3),(2,1),(2,3),(3,1),(3,2)]

print(list(itertools.product([0,1], repeat=3)))
# [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]

# accumulate
import operator
print(list(itertools.accumulate([1,2,3,4,5])))                    # [1,3,6,10,15]
print(list(itertools.accumulate([1,2,3,4,5], operator.mul)))      # [1,2,6,24,120]


print("\nAll done! ✓")
