"""
Python Basics — Definitions + Code + Outputs
Run: python 01_python_basics.py
Every section starts with WHAT IS IT? so you know the concept before seeing code.
"""

# ══════════════════════════════════════════════════════
# 1. LIST COMPREHENSIONS
# ══════════════════════════════════════════════════════
# WHAT IS A LIST COMPREHENSION?
#   → A SHORT, READABLE way to build a new list from an existing iterable
#   → Replaces 3-4 lines of for-loop with ONE line
#   → Syntax: [expression  for item in iterable  if condition]
#              ↑ what to   ↑ loop variable       ↑ optional filter
#                keep/do
#
# WHEN TO USE:
#   → Use it when you want to transform or filter a list
#   → Avoid it if the logic is complex — a for loop is clearer then

print("=" * 55)
print("LIST COMPREHENSIONS")
print("=" * 55)

squares = [x**2 for x in range(10)]
print(squares)
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

evens = [x for x in range(20) if x % 2 == 0]   # only keep even numbers
print(evens)
# [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

# NESTED comprehension — 2D matrix (list of lists)
# Inner loop runs first: for each row i, build a row [i*0, i*1, i*2, i*3, i*4]
matrix = [[i * j for j in range(5)] for i in range(4)]
for row in matrix:
    print(row)
# [0, 0, 0, 0, 0]
# [0, 1, 2, 3, 4]
# [0, 2, 4, 6, 8]
# [0, 3, 6, 9, 12]

# Conditional expression (if/else INSIDE) — transforms, doesn't filter
labels = ["even" if x % 2 == 0 else "odd" for x in range(6)]
print(labels)
# ['even', 'odd', 'even', 'odd', 'even', 'odd']

# Dict comprehension — same idea, produces a dict
word_lengths = {word: len(word) for word in ["hello", "world", "python"]}
print(word_lengths)
# {'hello': 5, 'world': 5, 'python': 6}

# Set comprehension — produces a set (no duplicates)
unique_lengths = {len(word) for word in ["hi", "hello", "hey", "world"]}
print(unique_lengths)
# {2, 5}  ← only unique lengths


# ══════════════════════════════════════════════════════
# 2. DICTIONARIES
# ══════════════════════════════════════════════════════
# WHAT IS A DICTIONARY?
#   → Stores data as KEY → VALUE pairs
#   → Keys must be UNIQUE and IMMUTABLE (string, int, tuple)
#   → Values can be ANYTHING — list, dict, function, object
#   → ORDERED (Python 3.7+) — remembers insertion order
#   → O(1) lookup speed — finding a value by key is INSTANT
#
# COMMON METHODS:
#   → d["key"]          → get value (KeyError if missing)
#   → d.get("key", X)   → get value, return X if missing (SAFE)
#   → d.update({...})   → add/update multiple keys at once
#   → d.pop("key")      → remove + return value
#   → d.keys()          → all keys
#   → d.values()        → all values
#   → d.items()         → all (key, value) pairs ← most used in loops

print("\n" + "=" * 55)
print("DICTIONARIES")
print("=" * 55)

student = {"name": "Alice", "age": 25, "grade": "A"}

print(student["name"])            # Alice
print(student.get("score", 0))   # 0   ← safe: returns default instead of error

student["city"] = "Mumbai"                           # add new key
student.update({"age": 26, "score": 95})             # update multiple at once
print(student)
# {'name':'Alice','age':26,'grade':'A','city':'Mumbai','score':95}

# Iterating — .items() gives you both key and value
for key, value in student.items():
    print(f"  {key} → {value}")

# Dict comprehension — build dict with transformation
word_lengths = {word: len(word) for word in ["hello", "world", "python"]}
print(word_lengths)    # {'hello': 5, 'world': 5, 'python': 6}

# MERGE two dicts — Python 3.9+ syntax using |
# If same key exists → RIGHT side WINS
defaults = {"color": "blue", "size": 10}
custom   = {"color": "red"}
merged   = defaults | custom
print(merged)          # {'color': 'red', 'size': 10}  ← 'red' overwrote 'blue'

# Nested dict — dict inside a dict
students = {
    "alice": {"grade": "A", "score": 95},
    "bob":   {"grade": "B", "score": 82},
}
print(students["alice"]["score"])   # 95

# Sort dict by value (score descending)
ranked = sorted(students.items(), key=lambda x: x[1]["score"], reverse=True)
print(ranked[0])    # ('alice', {'grade': 'A', 'score': 95})


# ══════════════════════════════════════════════════════
# 3. SETS
# ══════════════════════════════════════════════════════
# WHAT IS A SET?
#   → UNORDERED collection that has NO DUPLICATES
#   → Adding the same item twice? It's stored only once.
#   → Super fast membership check: "is X in set?" is O(1)
#   → Supports math operations like Venn diagrams
#
# SET OPERATIONS (think Venn diagram):
#   → a & b  → INTERSECTION  — items in BOTH sets
#   → a | b  → UNION         — items in EITHER set
#   → a - b  → DIFFERENCE    — items in a but NOT in b
#   → a ^ b  → SYMMETRIC DIFF— items in one but NOT both
#
# WHEN TO USE:
#   → Remove duplicates from a list
#   → Fast "is this item in the collection?" checks
#   → Set math: finding common/unique elements between groups

print("\n" + "=" * 55)
print("SETS")
print("=" * 55)

a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

print(a & b)   # {3, 4}           INTERSECTION — only what's in both
print(a | b)   # {1, 2, 3, 4, 5, 6}  UNION — everything
print(a - b)   # {1, 2}           DIFFERENCE — in a, not in b
print(a ^ b)   # {1, 2, 5, 6}    SYMMETRIC DIFFERENCE — not in both

# Add / remove
a.add(10)
a.discard(99)   # discard = safe remove (no error if not found)
print(a)        # {1, 2, 3, 4, 10}

# REMOVE DUPLICATES from a list using set
# Note: set() doesn't preserve order — use dict.fromkeys() to preserve order
names_with_dupes = ["Alice", "Bob", "Alice", "Charlie", "Bob"]
unique_names = list(set(names_with_dupes))
print(unique_names)    # ['Alice', 'Bob', 'Charlie'] (order not guaranteed)

order_preserved = list(dict.fromkeys(names_with_dupes))
print(order_preserved)  # ['Alice', 'Bob', 'Charlie'] (order preserved)

# Membership check — O(1), much faster than list for large data
big_set = set(range(1000000))
print(999999 in big_set)   # True  ← instant lookup


# ══════════════════════════════════════════════════════
# 4. FUNCTIONS — args, kwargs, lambda, map/filter/reduce
# ══════════════════════════════════════════════════════
# WHAT IS A FUNCTION?
#   → A named, reusable block of code
#   → Takes input (parameters), does work, returns output
#   → def defines it, return sends back the result
#
# *args  (positional arguments):
#   → Let a function accept ANY NUMBER of positional arguments
#   → Inside the function, args is a TUPLE
#   → Use when you don't know how many values will be passed
#
# **kwargs  (keyword arguments):
#   → Let a function accept ANY NUMBER of keyword=value arguments
#   → Inside the function, kwargs is a DICT
#   → Use when you want named optional parameters
#
# LAMBDA:
#   → Anonymous (nameless) one-line function
#   → Syntax: lambda args: expression
#   → Good for short throwaway functions (sorting keys, map/filter)
#
# MAP: apply a function to EVERY item → iterator of results
# FILTER: keep only items where function returns True
# REDUCE: combine all items into ONE value using a rolling operation

print("\n" + "=" * 55)
print("FUNCTIONS")
print("=" * 55)

def greet(name, greeting="Hello"):   # greeting has a default value
    return f"{greeting}, {name}!"

print(greet("Alice"))           # Hello, Alice!   ← used default
print(greet("Bob", "Hi"))       # Hi, Bob!        ← overrode default

# *args — any number of positional args → collected as TUPLE
def log(*args, **kwargs):
    print(f"  args   = {args}")     # tuple
    print(f"  kwargs = {kwargs}")   # dict

log("error", "warning", level="critical", timestamp="2024")
# args   = ('error', 'warning')
# kwargs = {'level': 'critical', 'timestamp': '2024'}

# LAMBDA — short anonymous function
add    = lambda x, y: x + y
double = lambda x: x * 2
print(add(3, 4))     # 7
print(double(5))     # 10

# MAP — apply function to EVERY item (returns iterator, use list() to see)
numbers = [1, 2, 3, 4, 5]
doubled = list(map(lambda x: x * 2, numbers))
print(doubled)       # [2, 4, 6, 8, 10]

# FILTER — keep only items where function returns True
big_nums = list(filter(lambda x: x > 2, numbers))
print(big_nums)      # [3, 4, 5]

# REDUCE — rolls through list, combining with a function
# reduce(f, [a,b,c,d]) → f(f(f(a,b), c), d)
from functools import reduce
total_sum     = reduce(lambda x, y: x + y, numbers)    # 1+2+3+4+5
total_product = reduce(lambda x, y: x * y, numbers)    # 1*2*3*4*5
print(total_sum)      # 15
print(total_product)  # 120

# SORTED with key — very common interview pattern
people = [{"name": "Charlie", "age": 30}, {"name": "Alice", "age": 25}]
by_age = sorted(people, key=lambda p: p["age"])
print([p["name"] for p in by_age])   # ['Alice', 'Charlie']


# ══════════════════════════════════════════════════════
# 5. DECORATORS
# ══════════════════════════════════════════════════════
# WHAT IS A DECORATOR?
#   → A function that WRAPS another function to add extra behavior
#   → WITHOUT changing the original function's code
#   → @decorator_name is shorthand for: func = decorator(func)
#
# HOW IT WORKS (step by step):
#   1. You write @timer above a function
#   2. Python runs: my_func = timer(my_func)
#   3. timer() returns a NEW function (wrapper) that:
#        a. Does the extra work (start timer)
#        b. Calls the REAL function
#        c. Does more work (stop timer, print)
#   4. Now calling my_func() actually calls wrapper()
#
# @functools.wraps(func) — IMPORTANT: keeps original function's name/docs
#   Without it: func.__name__ would say "wrapper" not the real name
#
# COMMON USE CASES:
#   → @timer      — measure execution time
#   → @logger     — log function calls
#   → @retry      — retry on failure
#   → @require_auth — check if user is logged in
#   → @lru_cache  — cache results (memoization)

print("\n" + "=" * 55)
print("DECORATORS")
print("=" * 55)

import time
import functools

def timer(func):
    @functools.wraps(func)          # keeps original name (important!)
    def wrapper(*args, **kwargs):
        start  = time.time()
        result = func(*args, **kwargs)   # call the REAL function
        elapsed = time.time() - start
        print(f"  {func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper    # return wrapped version

def logger(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"  CALL: {func.__name__}({args}, {kwargs})")
        result = func(*args, **kwargs)
        print(f"  RETURNED: {result}")
        return result
    return wrapper

@timer               # slow_function = timer(slow_function)
def slow_function():
    time.sleep(0.1)
    return "done"

slow_function()      # slow_function took 0.1002s

@logger
def add(a, b):
    return a + b

add(3, 4)
# CALL: add((3, 4), {})
# RETURNED: 7

# DECORATOR WITH ARGUMENTS — extra wrapper layer
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
def say_hi():
    print("  Hi!")

say_hi()
# Hi!
# Hi!
# Hi!

# LRU CACHE — built-in memoization decorator
# Stores results of function calls → returns cached result instead of recomputing
from functools import lru_cache

@lru_cache(maxsize=128)
def fib(n):
    if n < 2: return n
    return fib(n-1) + fib(n-2)

print(fib(35))           # 9227465  (super fast with cache)
print(fib.cache_info())  # CacheInfo(hits=33, misses=36, maxsize=128, currsize=36)


# ══════════════════════════════════════════════════════
# 6. GENERATORS
# ══════════════════════════════════════════════════════
# WHAT IS A GENERATOR?
#   → A special function that YIELDS values ONE AT A TIME
#   → Uses "yield" instead of "return"
#   → It PAUSES at each yield and RESUMES when next() is called
#   → DOES NOT store all values in memory — computes on demand
#
# GENERATOR vs REGULAR FUNCTION:
#   → Regular function: runs all code, returns ONE result, done
#   → Generator:        runs to first yield, PAUSES, resumes later
#
# GENERATOR vs LIST:
#   → List:      [x**2 for x in range(1M)]  → stores 1 MILLION numbers in RAM
#   → Generator: (x**2 for x in range(1M))  → stores NOTHING, computes on request
#
# WHEN TO USE:
#   → Large datasets that don't fit in memory
#   → Infinite sequences (fibonacci, counters)
#   → Data pipelines (read → process → filter in steps)
#
# KEY RULE: You can only iterate a generator ONCE
#   → After it's exhausted, next() raises StopIteration

print("\n" + "=" * 55)
print("GENERATORS")
print("=" * 55)

# Basic generator — yield pauses and resumes
def countdown(n):
    while n > 0:
        yield n       # PAUSE here, send n to caller
        n -= 1        # resume here on next call

gen = countdown(5)
print(next(gen))   # 5   ← resume, get next value
print(next(gen))   # 4
print(next(gen))   # 3

for val in countdown(3):
    print(val, end=" ")   # 3 2 1
print()

# INFINITE generator — runs forever, take what you need
def fibonacci():
    a, b = 0, 1
    while True:     # never stops — but doesn't crash because it's lazy
        yield a
        a, b = b, a + b

gen = fibonacci()
first_10 = [next(gen) for _ in range(10)]
print(first_10)   # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# GENERATOR EXPRESSION — like list comprehension but with ()
import sys
list_version = [x**2 for x in range(100000)]    # builds full list in memory
gen_version  = (x**2 for x in range(100000))    # computes lazily, no memory

print(f"List:      {sys.getsizeof(list_version):,} bytes")  # ~800,000 bytes
print(f"Generator: {sys.getsizeof(gen_version)} bytes")     # ~200 bytes!

# Generator is great for sum/max/min — no need to store the list
sum_squares = sum(x**2 for x in range(1000000))   # doesn't build full list!
print(f"Sum of squares: {sum_squares:,}")

# GENERATOR PIPELINE — chain generators like Unix pipes
def read_data():
    for i in range(10):
        yield i

def square(nums):
    for n in nums:
        yield n ** 2

def filter_big(nums, threshold=30):
    for n in nums:
        if n > threshold:
            yield n

# Each generator pulls from the previous one — memory efficient!
result = list(filter_big(square(read_data())))
print(result)   # [36, 49, 64, 81]  (6²=36, 7²=49, ...)


# ══════════════════════════════════════════════════════
# 7. CLASSES & OOP
# ══════════════════════════════════════════════════════
# WHAT IS OOP (Object-Oriented Programming)?
#   → Organize code as OBJECTS that combine data + behavior
#   → A CLASS is a BLUEPRINT/TEMPLATE
#   → An OBJECT is an INSTANCE (a real thing made from the blueprint)
#   → class Dog: → blueprint    /    rex = Dog("Rex") → instance
#
# 4 PILLARS OF OOP:
#   1. ENCAPSULATION  — keep related data + methods together in one class
#                       hide internal details (use _ prefix for "private")
#   2. INHERITANCE    — child class gets all parent's code for free
#                       Dog inherits Animal → Dog has speak(), get_count() etc.
#   3. POLYMORPHISM   — same method name, different behavior per class
#                       cat.speak() → "Meow" / dog.speak() → "Woof"
#   4. ABSTRACTION    — expose only what's needed, hide complexity
#
# SPECIAL METHODS (dunder/magic methods):
#   → __init__    : called when object is created (constructor)
#   → __repr__    : "developer" representation  → used in console/debugging
#   → __str__     : "user-friendly" string      → used by print()
#   → __len__     : makes len(obj) work
#   → __eq__      : defines what obj1 == obj2 means
#   → __add__     : defines what obj1 + obj2 means
#
# TYPES OF METHODS:
#   → def method(self)      : instance method — needs an object to call
#   → @classmethod          : class method — called on the CLASS, gets cls
#   → @staticmethod         : utility — no self or cls, just a function inside class
#   → @property             : call a method as if it's an attribute (no parentheses)

print("\n" + "=" * 55)
print("CLASSES & OOP")
print("=" * 55)

class Animal:
    species_count = 0   # CLASS VARIABLE — shared by ALL instances

    def __init__(self, name, sound):
        self.name  = name      # INSTANCE VARIABLE — unique per object
        self.sound = sound
        Animal.species_count += 1

    def speak(self):           # instance method — needs self
        return f"{self.name} says {self.sound}"

    def __repr__(self):        # developer view: Animal(name='Rex')
        return f"Animal(name={self.name!r})"

    def __str__(self):         # user view: Rex
        return self.name

    def __len__(self):         # len(animal) → length of name
        return len(self.name)

    def __eq__(self, other):   # animal1 == animal2 → compare names
        return isinstance(other, Animal) and self.name == other.name

    @classmethod               # called on CLASS: Animal.get_count()
    def get_count(cls):        # cls = the class itself
        return cls.species_count

    @staticmethod              # utility: Animal.is_animal(x)
    def is_animal(obj):        # no self or cls needed
        return isinstance(obj, Animal)


class Dog(Animal):             # INHERITANCE — Dog gets everything from Animal
    def __init__(self, name, breed):
        super().__init__(name, sound="Woof")   # call Animal's __init__
        self.breed   = breed
        self._tricks = []      # _ prefix = "private by convention"

    def speak(self):           # OVERRIDES Animal.speak (polymorphism)
        return f"{self.name} ({self.breed}): Woof!"

    def fetch(self, item):
        return f"{self.name} fetches the {item}!"

    @property                  # access as dog.tricks (no parentheses)
    def tricks(self):
        return self._tricks.copy()

    def learn(self, trick):
        self._tricks.append(trick)


cat = Animal("Whiskers", "Meow")
dog = Dog("Rex", "Labrador")

print(cat.speak())              # Whiskers says Meow
print(dog.speak())              # Rex (Labrador): Woof!  ← overridden
print(repr(cat))                # Animal(name='Whiskers')
print(str(cat))                 # Whiskers
print(len(cat))                 # 8
print(cat == Animal("Whiskers", "Purr"))  # True  ← __eq__ compares names
print(Animal.get_count())       # 2  ← class variable shared
print(Animal.is_animal(dog))    # True

dog.learn("sit")
dog.learn("shake")
print(dog.fetch("ball"))        # Rex fetches the ball!
print(dog.tricks)               # ['sit', 'shake']

print(isinstance(dog, Animal))  # True  ← Dog is a subclass of Animal
print(issubclass(Dog, Animal))  # True


# ══════════════════════════════════════════════════════
# 8. CONTEXT MANAGERS
# ══════════════════════════════════════════════════════
# WHAT IS A CONTEXT MANAGER?
#   → Automatically handles SETUP and TEARDOWN (cleanup) using "with"
#   → Guarantees cleanup happens even if an EXCEPTION occurs
#
# HOW IT WORKS:
#   → __enter__ : runs at the start of "with" block (setup)
#   → __exit__  : runs when "with" block ends (cleanup — always runs)
#   → The "as f" part: f = whatever __enter__ returns
#
# MOST COMMON EXAMPLE:
#   with open("file.txt") as f:   ← file.close() is called automatically
#       data = f.read()
#
# CUSTOM CONTEXT MANAGER — two ways:
#   1. Class with __enter__ and __exit__
#   2. Function with @contextmanager decorator (simpler)
#
# WHEN TO USE:
#   → File operations    → auto-close file
#   → Database connections → auto-commit/rollback and close
#   → Timing blocks      → start/stop timer
#   → Locks/semaphores   → acquire/release automatically

print("\n" + "=" * 55)
print("CONTEXT MANAGERS")
print("=" * 55)

# METHOD 1: class-based
class ManagedTimer:
    def __enter__(self):
        self.start = time.perf_counter()
        print("  Timer started")
        return self           # this is what "as t" receives

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start
        print(f"  Timer stopped: {elapsed:.4f}s")
        return False          # False = don't suppress exceptions

with ManagedTimer() as t:
    total = sum(range(500000))
print(f"  Sum: {total:,}")
# Timer started
# Timer stopped: 0.0124s
# Sum: 124,999,750,000

# METHOD 2: @contextmanager decorator (simpler — no class needed)
# code BEFORE yield = __enter__
# code AFTER  yield = __exit__
from contextlib import contextmanager

@contextmanager
def db_connection(db_name):
    print(f"  Connecting to {db_name}...")     # setup
    conn = {"db": db_name, "status": "open"}
    try:
        yield conn                              # the "as conn" variable
    finally:
        conn["status"] = "closed"
        print(f"  Connection to {db_name} closed")   # cleanup

with db_connection("PostgreSQL") as conn:
    print(f"  Using connection: {conn}")
# Connecting to PostgreSQL...
# Using connection: {'db': 'PostgreSQL', 'status': 'open'}
# Connection to PostgreSQL closed


# ══════════════════════════════════════════════════════
# 9. ERROR HANDLING
# ══════════════════════════════════════════════════════
# WHAT IS ERROR HANDLING?
#   → Catch errors at runtime so your program doesn't CRASH
#   → Handle specific error types differently
#
# STRUCTURE:
#   try:      code that might fail
#   except:   what to do when a specific error happens
#   else:     runs ONLY if NO error occurred in try
#   finally:  ALWAYS runs — perfect for cleanup (close file, release lock)
#
# COMMON BUILT-IN EXCEPTIONS:
#   → ValueError       : right type, wrong value  → int("abc")
#   → TypeError        : wrong type               → 1 + "a"
#   → KeyError         : dict key missing          → d["missing"]
#   → IndexError       : list index out of range   → [1,2][5]
#   → AttributeError   : object has no such attr  → "str".foo
#   → ZeroDivisionError: divide by zero
#   → FileNotFoundError: file doesn't exist
#   → StopIteration    : generator exhausted
#
# CUSTOM EXCEPTIONS:
#   → Inherit from Exception to create your own error types
#   → Makes errors more descriptive and catchable specifically

print("\n" + "=" * 55)
print("ERROR HANDLING")
print("=" * 55)

# Basic try/except/else/finally
def safe_divide(a, b):
    try:
        result = a / b             # might fail
    except ZeroDivisionError:
        print("  Cannot divide by zero")
        return None
    except TypeError as e:
        print(f"  Type error: {e}")
        return None
    else:
        print("  Division successful")   # only if NO exception
        return result
    finally:
        print("  Always runs (cleanup here)")   # ALWAYS runs

print(safe_divide(10, 2))
# Division successful
# Always runs
# 5.0

print(safe_divide(10, 0))
# Cannot divide by zero
# Always runs
# None

# CUSTOM EXCEPTION — inherit from Exception
class ValidationError(Exception):
    def __init__(self, field, message):
        self.field = field
        super().__init__(f"{field}: {message}")

def validate_age(age):
    if not isinstance(age, int):
        raise ValidationError("age", "must be an integer")
    if not (0 <= age <= 150):
        raise ValidationError("age", "must be between 0 and 150")
    return age

try:
    validate_age("twenty")
except ValidationError as e:
    print(f"  Caught: {e}")       # age: must be an integer
    print(f"  Field: {e.field}")  # age

# Catch multiple exception types in one line
try:
    result = int("abc") + [1, 2]
except (ValueError, TypeError) as e:
    print(f"  Error: {e}")        # invalid literal for int() with base 10: 'abc'


# ══════════════════════════════════════════════════════
# 10. BUILT-IN FUNCTIONS — enumerate, zip, any, all
# ══════════════════════════════════════════════════════
# ENUMERATE — loop with index AND value together
#   → Returns (index, value) pairs
#   → Much cleaner than range(len(list))
#
# ZIP — combine two or more lists element by element
#   → Pairs up items: zip([1,2,3], ["a","b","c"]) → (1,"a"),(2,"b"),(3,"c")
#   → Stops at the SHORTER list
#   → Great for creating dicts from two parallel lists
#
# ANY — returns True if AT LEAST ONE item is truthy
# ALL — returns True if ALL items are truthy
#   → Both accept any iterable (list, generator expression)
#   → Generator expressions work perfectly: any(x > 5 for x in nums)

print("\n" + "=" * 55)
print("BUILT-IN FUNCTIONS")
print("=" * 55)

# ENUMERATE
fruits = ["apple", "banana", "cherry"]
for i, fruit in enumerate(fruits):
    print(f"  {i}: {fruit}")
# 0: apple
# 1: banana
# 2: cherry

for i, fruit in enumerate(fruits, start=1):    # start counting from 1
    print(f"  {i}. {fruit}")
# 1. apple    2. banana    3. cherry

# Use enumerate to build indexed dict
indexed = {i: fruit for i, fruit in enumerate(fruits)}
print(indexed)   # {0: 'apple', 1: 'banana', 2: 'cherry'}

# ZIP
names  = ["Alice", "Bob", "Charlie"]
scores = [90, 85, 92]

for name, score in zip(names, scores):
    print(f"  {name}: {score}")
# Alice: 90    Bob: 85    Charlie: 92

combined = dict(zip(names, scores))    # zip → dict (very common!)
print(combined)   # {'Alice': 90, 'Bob': 85, 'Charlie': 92}

# Zip three lists
ages = [25, 30, 28]
for name, score, age in zip(names, scores, ages):
    print(f"  {name}, age {age}, score {score}")

# UNZIP — reverse of zip using *
pairs = [(1, "a"), (2, "b"), (3, "c")]
nums_u, letters = zip(*pairs)
print(nums_u)    # (1, 2, 3)
print(letters)   # ('a', 'b', 'c')

# ANY and ALL
data = [True, False, True]
print(any(data))   # True   ← at least one True
print(all(data))   # False  ← not ALL are True

nums_list = [2, 4, 6, 8, 10]
print(all(x % 2 == 0 for x in nums_list))    # True  ← all are even
print(any(x > 100 for x in nums_list))        # False ← none exceed 100
print(any(x > 5 for x in nums_list))          # True  ← 6,8,10 are > 5


# ══════════════════════════════════════════════════════
# 11. COLLECTIONS MODULE
# ══════════════════════════════════════════════════════
# WHAT IS THE COLLECTIONS MODULE?
#   → Built-in module with specialized container types
#   → Better than plain dict/list for specific use cases
#
# COUNTER — count occurrences of items
#   → Counter("aabbc") → {'a':2, 'b':2, 'c':1}
#   → .most_common(n) → top n most frequent items
#   → Returns 0 for missing keys (not KeyError)
#
# DEFAULTDICT — dict that auto-creates missing keys
#   → defaultdict(list) → missing key creates []
#   → defaultdict(int)  → missing key creates 0
#   → Avoids "if key not in dict" checks
#
# DEQUE — double-ended queue
#   → appendleft/popleft is O(1) — list.insert(0,...) is O(n)
#   → maxlen: auto-drops oldest when full (sliding window)
#
# NAMEDTUPLE — tuple with field names
#   → Immutable like tuple, but access by name not just index
#   → Clearer than tuple(3, 4) — vs Point(x=3, y=4)

print("\n" + "=" * 55)
print("COLLECTIONS")
print("=" * 55)

from collections import Counter, defaultdict, deque, namedtuple

# COUNTER
words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
count = Counter(words)
print(count)                  # Counter({'apple': 3, 'banana': 2, 'cherry': 1})
print(count.most_common(2))   # [('apple', 3), ('banana', 2)]
print(count["apple"])         # 3
print(count["missing"])       # 0  ← no KeyError, returns 0

# Counter on a string
letter_count = Counter("mississippi")
print(letter_count.most_common(3))   # [('s', 4), ('i', 4), ('p', 2)]

# DEFAULTDICT — grouping without KeyError
graph = defaultdict(list)
for edge in [("a","b"), ("a","c"), ("b","c"), ("b","d")]:
    graph[edge[0]].append(edge[1])   # no KeyError even for new keys
print(dict(graph))   # {'a': ['b', 'c'], 'b': ['c', 'd']}

# Count with defaultdict
word_count = defaultdict(int)
for word in ["the", "quick", "the", "fox", "the"]:
    word_count[word] += 1   # no need to check if key exists
print(dict(word_count))     # {'the': 3, 'quick': 1, 'fox': 1}

# DEQUE — sliding window / efficient queue
queue = deque(maxlen=3)     # holds at most 3 items
for i in range(5):
    queue.append(i)
    print(f"  Added {i}: {list(queue)}")
# Added 0: [0]
# Added 1: [0, 1]
# Added 2: [0, 1, 2]
# Added 3: [1, 2, 3]  ← 0 dropped (oldest)
# Added 4: [2, 3, 4]  ← 1 dropped

# Deque as a queue (FIFO): appendleft + pop, or append + popleft
q = deque([1, 2, 3])
q.appendleft(0)    # [0, 1, 2, 3]
q.pop()            # remove from right → 3
q.popleft()        # remove from left  → 0
print(list(q))     # [1, 2]

# NAMEDTUPLE — readable tuple with field names
Point = namedtuple("Point", ["x", "y"])
p     = Point(3, 4)
print(p.x, p.y)    # 3 4        ← access by NAME
print(p[0], p[1])  # 3 4        ← also works by INDEX (it's still a tuple)
print(p)           # Point(x=3, y=4)

Person = namedtuple("Person", ["name", "age", "city"])
alice  = Person("Alice", 25, "Mumbai")
print(alice.name)  # Alice
print(alice._asdict())  # OrderedDict([('name', 'Alice'), ('age', 25), ('city', 'Mumbai')])


# ══════════════════════════════════════════════════════
# 12. PATHLIB
# ══════════════════════════════════════════════════════
# WHAT IS PATHLIB?
#   → Modern way to work with file paths (Python 3.4+)
#   → Better than os.path — uses / operator to join paths
#   → Path objects are smart: they know the OS, handle separators
#   → Methods: .exists(), .is_file(), .is_dir(), .suffix, .stem, .parent
#
# OLD WAY (os.path):  os.path.join("data", "file.csv")
# NEW WAY (pathlib):  Path("data") / "file.csv"      ← much cleaner

print("\n" + "=" * 55)
print("PATHLIB")
print("=" * 55)

from pathlib import Path

# Build paths with / operator
data_dir = Path("data")
file_path = data_dir / "results" / "output.csv"
print(file_path)         # data/results/output.csv

# Inspect path parts
p = Path("/home/user/project/data/file.csv")
print(p.name)     # file.csv        ← file name with extension
print(p.stem)     # file            ← file name WITHOUT extension
print(p.suffix)   # .csv            ← just the extension
print(p.parent)   # /home/user/project/data
print(p.parts)    # ('/', 'home', 'user', 'project', 'data', 'file.csv')

# Check if path exists
cwd = Path(".")                     # current directory
print(cwd.exists())                 # True
print(cwd.is_dir())                 # True

# Find files
py_files = list(cwd.glob("*.py"))   # all .py files in current dir
print(f"Python files: {[f.name for f in py_files]}")

# all .py files recursively (in all subdirectories)
# all_py = list(cwd.rglob("*.py"))


# ══════════════════════════════════════════════════════
# 13. TYPE HINTS
# ══════════════════════════════════════════════════════
# WHAT ARE TYPE HINTS?
#   → Annotations that say what TYPE a parameter/return value should be
#   → NOT enforced at runtime — Python won't error if wrong type is passed
#   → Benefits: IDE autocomplete, documentation, catches bugs with mypy
#   → Syntax: def func(param: type) -> return_type:
#
# COMMON TYPES:
#   → int, float, str, bool       : primitive types
#   → list, dict, set, tuple      : collection types (Python 3.9+)
#   → List[int], Dict[str, int]   : typed collections (older style)
#   → Optional[X]                 : X or None
#   → Union[X, Y]                 : X or Y
#   → Any                         : any type (escape hatch)
#   → Callable[[int], str]        : a function that takes int, returns str

print("\n" + "=" * 55)
print("TYPE HINTS")
print("=" * 55)

from typing import List, Dict, Optional, Tuple, Union

# Basic type hints
def add(a: int, b: int) -> int:
    return a + b

def greet_hint(name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"

# Optional — can be None
def find_user(user_id: int) -> Optional[str]:
    users = {1: "Alice", 2: "Bob"}
    return users.get(user_id)   # returns None if not found

# Complex types
def process_data(
    data:      List[Dict[str, float]],   # list of dicts
    threshold: float = 0.5,
) -> Tuple[List[float], int]:            # returns tuple of (list, int)
    values = [d["value"] for d in data if d["value"] > threshold]
    return values, len(values)

# Union — accepts multiple types
def stringify(value: Union[int, float, str]) -> str:
    return str(value)

# Python 3.10+ shorthand — use | instead of Union
# def stringify(value: int | float | str) -> str: ...

print(add(3, 4))             # 7
print(greet_hint("Alice"))   # Hello, Alice!
print(find_user(1))          # Alice
print(find_user(99))         # None
print(stringify(42))         # "42"
print(stringify(3.14))       # "3.14"

# TYPE HINTS DON'T ENFORCE — this won't raise an error at runtime:
result = add("3", "4")       # works! returns "34" (string concat)
print(result)                # 34  ← no error, just wrong behavior
# Use mypy (pip install mypy) to catch these: mypy 01_python_basics.py


print("\n" + "=" * 55)
print("All done! ✓")
print("=" * 55)
