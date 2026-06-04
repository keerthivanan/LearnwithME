"""
Python Basics — Interview Code Examples
"""

# ── LIST COMPREHENSIONS ───────────────────────────────
squares = [x**2 for x in range(10)]
evens   = [x for x in range(20) if x % 2 == 0]
matrix  = [[i * j for j in range(5)] for i in range(5)]

# ── DICTIONARIES ──────────────────────────────────────
student = {"name": "Alice", "age": 25, "grade": "A"}

# dict comprehension
word_lengths = {word: len(word) for word in ["hello", "world", "python"]}

# merge dicts (Python 3.9+)
defaults = {"color": "blue", "size": 10}
custom   = {"color": "red"}
merged   = defaults | custom  # {"color": "red", "size": 10}

# ── SETS ──────────────────────────────────────────────
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}
print(a & b)  # intersection: {3, 4}
print(a | b)  # union:        {1,2,3,4,5,6}
print(a - b)  # difference:   {1, 2}

# ── FUNCTIONS ─────────────────────────────────────────
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# args and kwargs
def log(*args, **kwargs):
    print(args)           # tuple of positional args
    print(kwargs)         # dict of keyword args

log("error", "warning", level="critical", timestamp="2024")

# Lambda
add = lambda x, y: x + y
double = lambda x: x * 2

# Map, filter, reduce
numbers = [1, 2, 3, 4, 5]
doubled  = list(map(lambda x: x * 2, numbers))
filtered = list(filter(lambda x: x > 2, numbers))

from functools import reduce
total = reduce(lambda x, y: x + y, numbers)  # 15

# ── DECORATORS ────────────────────────────────────────
import time
import functools

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(0.1)
    return "done"

slow_function()

# ── GENERATORS ────────────────────────────────────────
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

gen = fibonacci()
first_10 = [next(gen) for _ in range(10)]
print(first_10)  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# Generator expression (memory efficient)
sum_squares = sum(x**2 for x in range(1000000))  # doesn't build full list

# ── CLASSES ──────────────────────────────────────────
class Animal:
    species_count = 0  # class variable

    def __init__(self, name, sound):
        self.name = name
        self.sound = sound
        Animal.species_count += 1

    def speak(self):
        return f"{self.name} says {self.sound}"

    def __repr__(self):
        return f"Animal(name={self.name!r})"

    def __str__(self):
        return self.name

    @classmethod
    def get_count(cls):
        return cls.species_count

    @staticmethod
    def is_animal(obj):
        return isinstance(obj, Animal)


class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, sound="Woof")
        self.breed = breed

    def fetch(self, item):
        return f"{self.name} fetches the {item}!"


dog = Dog("Rex", "Labrador")
print(dog.speak())      # Rex says Woof
print(dog.fetch("ball"))

# ── CONTEXT MANAGERS ──────────────────────────────────
class ManagedFile:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        return False  # don't suppress exceptions

# with ManagedFile("test.txt", "w") as f:
#     f.write("hello")

# ── ERROR HANDLING ────────────────────────────────────
def safe_divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("Cannot divide by zero")
        return None
    except TypeError as e:
        print(f"Type error: {e}")
        return None
    else:
        print("Division successful")
        return result
    finally:
        print("Always runs")


# Custom exception
class ValidationError(Exception):
    def __init__(self, field, message):
        self.field = field
        super().__init__(f"{field}: {message}")


def validate_age(age):
    if not isinstance(age, int):
        raise ValidationError("age", "must be an integer")
    if age < 0 or age > 150:
        raise ValidationError("age", "must be between 0 and 150")
    return age

# ── USEFUL BUILT-INS ──────────────────────────────────
nums = [3, 1, 4, 1, 5, 9, 2, 6]

print(sorted(nums))                          # [1, 1, 2, 3, 4, 5, 6, 9]
print(sorted(nums, reverse=True))            # descending
print(sorted(["banana","apple","cherry"], key=len))  # sort by length

# enumerate
for i, val in enumerate(["a", "b", "c"]):
    print(i, val)

# zip
names  = ["Alice", "Bob", "Charlie"]
scores = [90, 85, 92]
for name, score in zip(names, scores):
    print(f"{name}: {score}")

combined = dict(zip(names, scores))

# any, all
data = [True, False, True]
print(any(data))   # True
print(all(data))   # False

# ── COLLECTIONS ───────────────────────────────────────
from collections import Counter, defaultdict, deque, namedtuple

# Counter
words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
count = Counter(words)
print(count.most_common(2))   # [('apple', 3), ('banana', 2)]

# defaultdict
graph = defaultdict(list)
graph["a"].append("b")
graph["a"].append("c")  # no KeyError!

# deque (efficient queue)
queue = deque(maxlen=3)
queue.append(1); queue.append(2); queue.append(3); queue.append(4)
print(queue)  # deque([2, 3, 4]) — 1 was dropped

# namedtuple
Point = namedtuple("Point", ["x", "y"])
p = Point(3, 4)
print(p.x, p.y)

# ── PATHLIB ───────────────────────────────────────────
from pathlib import Path

data_dir = Path("data")
csv_files = list(data_dir.glob("**/*.csv"))  # all CSVs recursively

# ── TYPE HINTS ────────────────────────────────────────
from typing import List, Dict, Optional, Tuple, Union

def process_data(
    data: List[Dict[str, float]],
    threshold: float = 0.5,
    output: Optional[str] = None
) -> Tuple[List[float], int]:
    values = [d["value"] for d in data if d["value"] > threshold]
    return values, len(values)
