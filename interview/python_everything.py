"""
Python Everything — Definitions + Code + Outputs
Run this file: python python_everything.py
"""

# ════════════════════════════════════════════════════
# 1. LISTS
# ════════════════════════════════════════════════════
# WHAT IS A LIST?
#   → Ordered, MUTABLE (changeable) collection that allows DUPLICATES
#   → Index starts at 0 from left, -1 from right (last item)
#   → Use when: you need a sequence that can grow/shrink/change
#   → Syntax: my_list = [item1, item2, item3]
print("=" * 50)
print("LISTS")
print("=" * 50)

fruits = ["apple", "banana", "cherry", "mango"]

# Indexing and slicing
print(fruits[0])          # apple      ← first item (index 0)
print(fruits[-1])         # mango      ← last item  (index -1)
print(fruits[1:3])        # ['banana', 'cherry']  ← slice [start:end] (end not included)

# Add items
fruits.append("grape")        # add to END
fruits.insert(1, "kiwi")      # add at position 1

# Remove items
fruits.remove("kiwi")         # remove by VALUE
popped = fruits.pop()         # remove + return LAST item
print(popped)             # grape
print(fruits)             # ['apple', 'banana', 'cherry', 'mango']

# Useful list operations
nums = [3, 1, 4, 1, 5, 9, 2, 6]
print(sorted(nums))                    # [1, 1, 2, 3, 4, 5, 6, 9]  ← returns NEW sorted list
print(sorted(nums, reverse=True))      # [9, 6, 5, 4, 3, 2, 1, 1]
print(len(nums))                       # 8    ← number of items
print(nums.count(1))                   # 2    ← how many times 1 appears
print(max(nums), min(nums), sum(nums)) # 9 1 31

# LIST COMPREHENSION — short way to build a list
# Syntax: [expression for item in iterable if condition]
squares = [x**2 for x in range(6)]
print(squares)   # [0, 1, 4, 9, 16, 25]

evens = [x for x in range(10) if x % 2 == 0]
print(evens)     # [0, 2, 4, 6, 8]

pairs = [(x, y) for x in [1, 2] for y in [3, 4]]
print(pairs)     # [(1,3),(1,4),(2,3),(2,4)]

# UNPACKING — pull values from a list into variables
first, *middle, last = [1, 2, 3, 4, 5]
print(first, middle, last)   # 1 [2, 3, 4] 5   ← *middle grabs everything in between

# Flatten nested list
nested = [[1, 2], [3, 4], [5, 6]]
flat = [x for row in nested for x in row]
print(flat)      # [1, 2, 3, 4, 5, 6]

# ENUMERATE — loop with BOTH index AND value at the same time
# Bad way:  for i in range(len(fruits)): print(i, fruits[i])  ← ugly
# Good way: for i, item in enumerate(fruits):                 ← clean
colors = ["red", "green", "blue"]
for i, color in enumerate(colors):
    print(f"  {i}: {color}")
# 0: red
# 1: green
# 2: blue

for i, color in enumerate(colors, start=1):   # start counting from 1 instead of 0
    print(f"  {i}. {color}")
# 1. red    2. green    3. blue

# ZIP — combine two (or more) lists element by element into pairs
# Stops at the SHORTER list if lengths differ
names  = ["Alice", "Bob", "Charlie"]
scores = [95, 82, 78]
for name, score in zip(names, scores):
    print(f"  {name}: {score}")
# Alice: 95
# Bob: 82
# Charlie: 78

# zip → dict (very common pattern)
name_score = dict(zip(names, scores))
print(name_score)   # {'Alice': 95, 'Bob': 82, 'Charlie': 78}

# zip two lists together → list of tuples
zipped = list(zip([1, 2, 3], ["a", "b", "c"]))
print(zipped)       # [(1, 'a'), (2, 'b'), (3, 'c')]

# Unzip — use * to reverse zip
pairs_data = [(1, "a"), (2, "b"), (3, "c")]
nums_u, letters = zip(*pairs_data)
print(nums_u)    # (1, 2, 3)
print(letters)   # ('a', 'b', 'c')

# ANY / ALL — check a condition across an entire collection
# any() → True if AT LEAST ONE item satisfies the condition
# all() → True if EVERY item satisfies the condition
num_check = [2, 4, 6, 7, 8]
print(any(x % 2 != 0 for x in num_check))   # True  ← at least one odd number (7)
print(all(x > 0 for x in num_check))         # True  ← all are positive
print(all(x % 2 == 0 for x in num_check))    # False ← 7 is odd, breaks "all even"
print(any(x > 100 for x in num_check))       # False ← none exceed 100


# ════════════════════════════════════════════════════
# 2. DICTIONARIES
# ════════════════════════════════════════════════════
# WHAT IS A DICTIONARY?
#   → KEY-VALUE pairs (like a real dictionary: word → definition)
#   → Keys must be UNIQUE and IMMUTABLE (strings, numbers, tuples)
#   → Values can be ANYTHING (list, dict, function, etc.)
#   → ORDERED (Python 3.7+) — maintains insertion order
#   → Syntax: my_dict = {"key": value, "key2": value2}
#   → Use when: you want to look up data by a meaningful name (not index)
print("\n" + "=" * 50)
print("DICTIONARIES")
print("=" * 50)

person = {"name": "Alice", "age": 25, "city": "Mumbai"}

# Access values
print(person["name"])             # Alice  ← direct access (KeyError if missing)
print(person.get("salary", 0))   # 0      ← safe access with DEFAULT if key missing

# Add / update keys
person["email"] = "alice@gmail.com"               # add new key
person.update({"age": 26, "country": "India"})    # update multiple at once
print(person)
# {'name':'Alice','age':26,'city':'Mumbai','email':'alice@gmail.com','country':'India'}

# Delete keys
del person["country"]           # delete by key name
removed = person.pop("city")    # delete + return value
print(removed)                  # Mumbai

# Iterate over dict
# .keys()   → only keys
# .values() → only values
# .items()  → (key, value) pairs — most common!
for key, value in person.items():
    print(f"  {key}: {value}")

# DICT COMPREHENSION — same idea as list comprehension
word_len = {word: len(word) for word in ["hello", "world", "python"]}
print(word_len)   # {'hello': 5, 'world': 5, 'python': 6}

squares_dict = {x: x**2 for x in range(6)}
print(squares_dict)  # {0:0, 1:1, 2:4, 3:9, 4:16, 5:25}

# MERGE two dicts (Python 3.9+)
# If same key exists — the RIGHT dict WINS (overwrites the left)
defaults = {"color": "blue", "size": 10}
custom   = {"color": "red"}
merged   = defaults | custom
print(merged)    # {'color': 'red', 'size': 10}  ← "red" overwrote "blue"

# NESTED dict — dict inside a dict
students = {
    "alice": {"grade": "A", "score": 95},
    "bob":   {"grade": "B", "score": 82},
}
print(students["alice"]["score"])   # 95

# Sort dict by value
sorted_by_score = sorted(students.items(), key=lambda x: x[1]["score"], reverse=True)
print(sorted_by_score)
# [('alice', {'grade':'A','score':95}), ('bob', {'grade':'B','score':82})]

# Create dict with all keys set to same value
keys   = ["x", "y", "z"]
zeroed = dict.fromkeys(keys, 0)
print(zeroed)   # {'x': 0, 'y': 0, 'z': 0}


# ════════════════════════════════════════════════════
# 3. SETS
# ════════════════════════════════════════════════════
# WHAT IS A SET?
#   → UNORDERED collection with NO DUPLICATES
#   → Fast membership check (O(1) lookup like dict keys)
#   → Supports math operations: union, intersection, difference
#   → Syntax: my_set = {1, 2, 3}  (NOT {} — that's an empty dict!)
#   → Use when: you want unique items, or to do set math
print("\n" + "=" * 50)
print("SETS")
print("=" * 50)

a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7}

# SET OPERATIONS (like Venn diagrams):
print(a & b)    # {3, 4, 5}       INTERSECTION  → items in BOTH
print(a | b)    # {1,2,3,4,5,6,7} UNION         → items in EITHER
print(a - b)    # {1, 2}          DIFFERENCE    → in a but NOT in b
print(a ^ b)    # {1, 2, 6, 7}   SYMMETRIC DIFF → in one but NOT both

a.add(10)
a.discard(99)    # discard = safe remove (no error if item doesn't exist)
print(a)         # {1, 2, 3, 4, 5, 10}

# Remove duplicates from a list (order-preserving trick)
data   = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
unique = list(dict.fromkeys(data))   # dict.fromkeys preserves order unlike set()
print(unique)    # [3, 1, 4, 5, 9, 2, 6]


# ════════════════════════════════════════════════════
# 4. TUPLES
# ════════════════════════════════════════════════════
# WHAT IS A TUPLE?
#   → Ordered, IMMUTABLE (cannot change after creation) collection
#   → Faster than lists, can be used as dict keys (lists cannot)
#   → Use when: data should NOT change (coordinates, RGB colors, config)
#   → Syntax: my_tuple = (item1, item2, item3)
print("\n" + "=" * 50)
print("TUPLES")
print("=" * 50)

point = (3, 4)
x, y  = point           # TUPLE UNPACKING — pull values into variables
print(x, y)             # 3 4

rgb = (255, 128, 0)
r, g, b = rgb
print(r, g, b)          # 255 128 0

# NAMED TUPLE — tuple with field names (like a lightweight class)
# Better than regular tuple because you can access by NAME not just index
from collections import namedtuple
Person = namedtuple("Person", ["name", "age", "city"])
alice  = Person("Alice", 25, "Mumbai")
print(alice.name, alice.age, alice.city)   # Alice 25 Mumbai   ← by name
print(alice[0])                            # Alice              ← by index (still works)


# ════════════════════════════════════════════════════
# 5. STRINGS
# ════════════════════════════════════════════════════
# WHAT IS A STRING?
#   → IMMUTABLE sequence of characters
#   → Enclosed in single ' or double " or triple ''' / """ quotes
#   → Supports indexing and slicing like lists
#   → Use when: storing text data
print("\n" + "=" * 50)
print("STRINGS")
print("=" * 50)

s = "  Hello, World!  "
print(s.strip())                           # "Hello, World!"   ← remove leading/trailing spaces
print(s.strip().lower())                   # "hello, world!"   ← all lowercase
print(s.strip().upper())                   # "HELLO, WORLD!"   ← all uppercase
print(s.strip().replace("World", "Python")) # "Hello, Python!"
print(s.strip().split(", "))               # ['Hello', 'World!']  ← split into list
print("-".join(["a", "b", "c"]))           # a-b-c   ← join list into string
print(s.strip().startswith("Hello"))       # True
print("Hello" in s)                        # True    ← membership check
print(s.strip().count("l"))                # 3       ← count occurrences

# F-STRINGS — best way to format strings (Python 3.6+)
# Syntax: f"text {variable} text {expression}"
name, score = "Alice", 95.678
print(f"Name: {name}, Score: {score:.2f}")   # Name: Alice, Score: 95.68  (.2f = 2 decimal places)
print(f"{'Left':<10}|{'Right':>10}")         # Left      |     Right      (<10 left-align, >10 right-align)
print(f"{12345:,}")                           # 12,345   (comma separator)

# REGEX — pattern matching in strings
# import re to use regex
import re
text   = "My phone: 9876543210 and email: alice@gmail.com"
phones = re.findall(r'\d{10}', text)           # find 10-digit numbers
print(phones)    # ['9876543210']

emails = re.findall(r'[\w.]+@[\w.]+\.\w+', text)  # find emails
print(emails)    # ['alice@gmail.com']

clean = re.sub(r'\s+', ' ', "too   many   spaces").strip()  # replace pattern
print(clean)     # too many spaces


# ════════════════════════════════════════════════════
# 6. FUNCTIONS
# ════════════════════════════════════════════════════
# WHAT IS A FUNCTION?
#   → A REUSABLE block of code that takes inputs, does something, returns output
#   → def keyword defines it, return sends back the result
#   → Default arguments make parameters optional
#   → *args   = accept ANY NUMBER of positional arguments → stored as TUPLE
#   → **kwargs = accept ANY NUMBER of keyword arguments  → stored as DICT
print("\n" + "=" * 50)
print("FUNCTIONS")
print("=" * 50)

# Basic function
def add(a, b):
    return a + b

print(add(3, 4))    # 7

# DEFAULT ARGUMENT — if not provided, use the default value
def greet(name, greeting="Hello"):   # greeting has a default
    return f"{greeting}, {name}!"

print(greet("Alice"))           # Hello, Alice!  ← used default "Hello"
print(greet("Bob", "Hi"))       # Hi, Bob!       ← overrode default

# *args — accepts ANY number of positional args, stored as a TUPLE
def total(*args):
    print(f"  args = {args}")    # args is a tuple
    return sum(args)

print(total(1, 2, 3, 4, 5))
# →   args = (1, 2, 3, 4, 5)    ← printed from INSIDE the function
# →   15                          ← return value printed by outer print()

# **kwargs — accepts ANY number of keyword args, stored as a DICT
def display(**kwargs):
    print(f"  kwargs = {kwargs}")   # kwargs is a dict
    return ", ".join(f"{k}={v}" for k, v in kwargs.items())

print(display(name="Alice", age=25))
# →   kwargs = {'name': 'Alice', 'age': 25}   ← printed from INSIDE the function
# →   name=Alice, age=25                        ← return value printed by outer print()

# COMBINED — name (required), *layers (extra positional), **params (extra keyword)
def create_model(name, *layers, **params):
    print(f"Model: {name}, Layers: {layers}, Params: {params}")

create_model("ResNet", 64, 128, 256, lr=0.001, dropout=0.3)
# Model: ResNet, Layers: (64,128,256), Params: {'lr':0.001,'dropout':0.3}

# LAMBDA — anonymous one-line function
# Syntax: lambda arguments: expression
square  = lambda x: x ** 2
is_even = lambda x: x % 2 == 0
print(square(5))     # 25
print(is_even(4))    # True

# MAP — apply a function to EVERY item in a list → returns iterator
# FILTER — keep only items where function returns True
# REDUCE — combine all items into one value (like rolling sum/product)
from functools import reduce
nums    = [1, 2, 3, 4, 5]
doubled = list(map(lambda x: x * 2, nums))        # apply x*2 to each
evens_f = list(filter(lambda x: x % 2 == 0, nums)) # keep only even
product = reduce(lambda x, y: x * y, nums)         # 1*2*3*4*5
print(doubled)    # [2, 4, 6, 8, 10]
print(evens_f)    # [2, 4]
print(product)    # 120

# CLOSURE — inner function that "remembers" the outer function's variables
# even after the outer function has finished running
def make_multiplier(factor):
    def multiply(x):
        return x * factor   # ← "factor" is remembered from outer scope
    return multiply          # returns the FUNCTION itself (not the result)

double = make_multiplier(2)   # double is now a function
triple = make_multiplier(3)
print(double(7))    # 14   (7 * 2)
print(triple(7))    # 21   (7 * 3)

# TYPE HINTS — annotate parameter types and return type (Python 3.5+)
# NOT enforced at runtime — just for readability, IDE autocomplete, and documentation
# Syntax: def func(param: type, param2: type = default) -> return_type:
from typing import Optional, Dict, Union

def add_typed(a: int, b: int) -> int:          # takes 2 ints, returns int
    return a + b

def greet_typed(name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"

def get_first(items: list) -> Optional[int]:   # Optional = can return None
    return items[0] if items else None

def process(data: Dict[str, int]) -> list:
    return sorted(data.items(), key=lambda x: x[1], reverse=True)

def parse(value: Union[int, str]) -> str:      # Union = accepts int OR str
    return str(value)

print(add_typed(3, 4))             # 7
print(greet_typed("Alice"))        # Hello, Alice!
print(get_first([10, 20, 30]))     # 10
print(get_first([]))               # None

# WALRUS OPERATOR := (Python 3.8+)
# Assigns AND returns the value in one step (inside conditions, while loops)
# Avoids calling a function twice just to check and use the result
data_list = [1, 5, 2, 8, 3, 9, 4]

# Without walrus: need to compute twice
# filtered = [x for x in data_list if compute(x) > 5]
# With walrus: compute once, use in same expression
filtered = [y for x in data_list if (y := x * 2) > 8]
print(filtered)   # [10, 16, 18]  ← x*2 computed once, used as y

# Common use: while loop that reads AND checks in one line
import io
stream = io.StringIO("line1\nline2\nline3\n")
while line := stream.readline():   # assign + check in one go
    print(f"  Got: {line.strip()}")
# Got: line1   Got: line2   Got: line3


# ════════════════════════════════════════════════════
# 7. DECORATORS
# ════════════════════════════════════════════════════
# WHAT IS A DECORATOR?
#   → A function that WRAPS another function to add extra behavior
#   → Written with @decorator_name above the function definition
#   → Used for: logging, timing, authentication, caching, retry logic
#   → Works like: new_func = decorator(old_func)
#   → @functools.wraps(func) preserves original function name & docstring
print("\n" + "=" * 50)
print("DECORATORS")
print("=" * 50)

import time
import functools

# DECORATOR 1: timer — measure how long a function takes
def timer(func):
    @functools.wraps(func)          # keeps original function name
    def wrapper(*args, **kwargs):   # wraps the function
        start  = time.perf_counter()
        result = func(*args, **kwargs)   # call the REAL function
        print(f"  {func.__name__} took {time.perf_counter()-start:.4f}s")
        return result
    return wrapper    # return the wrapped version

# DECORATOR 2: logger — prints what's called and what it returns
def logger(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"  Calling {func.__name__}({args}, {kwargs})")
        result = func(*args, **kwargs)
        print(f"  Returned: {result}")
        return result
    return wrapper

@timer           # same as: slow_add = timer(slow_add)
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

# DECORATOR WITH ARGUMENTS — outer function takes the argument
def repeat(times):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(times=3)     # call it 3 times automatically
def say_hello():
    print("  Hello!")

say_hello()
# Hello!   Hello!   Hello!

# LRU CACHE — memoization: store previously computed results
# Instead of recalculating, it returns cached result instantly
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2: return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(30))           # 832040  (fast because results are cached)
print(fibonacci.cache_info())  # CacheInfo(hits=28, misses=31, ...)


# ════════════════════════════════════════════════════
# 8. GENERATORS
# ════════════════════════════════════════════════════
# WHAT IS A GENERATOR?
#   → A function that YIELDS values ONE AT A TIME instead of all at once
#   → Uses "yield" keyword instead of "return"
#   → LAZY evaluation — computes next value only when asked
#   → MEMORY EFFICIENT — doesn't store all values in memory at once
#   → Use for: large datasets, infinite sequences, data pipelines
#   → A regular function returns ALL data at once (stored in memory)
#   → A generator returns ONE item at a time (barely uses memory)
print("\n" + "=" * 50)
print("GENERATORS")
print("=" * 50)

# Basic generator — "pauses" at yield, "resumes" when next() is called
def count_up(start, end):
    while start <= end:
        yield start    # pause here, give back the value, remember position
        start += 1     # resume from here on next() call

for n in count_up(1, 5):
    print(n, end=" ")   # 1 2 3 4 5
print()

# Infinite generator — never stops, take what you need with next()
def fibonacci_gen():
    a, b = 0, 1
    while True:          # infinite loop! but only computes when asked
        yield a
        a, b = b, a + b

fib     = fibonacci_gen()
first_8 = [next(fib) for _ in range(8)]   # pull 8 values
print(first_8)   # [0, 1, 1, 2, 3, 5, 8, 13]

# GENERATOR vs LIST — memory comparison
import sys
list_obj = [x**2 for x in range(10000)]    # stores ALL 10000 values
gen_obj  = (x**2 for x in range(10000))    # stores NONE — computes on demand
print(f"List:      {sys.getsizeof(list_obj):,} bytes")   # List:      85,176 bytes
print(f"Generator: {sys.getsizeof(gen_obj)} bytes")      # Generator: 200 bytes  ← tiny!

# GENERATOR PIPELINE — chain generators (processes data step by step)
def read_numbers():
    for i in range(10):
        yield i                 # step 1: produce 0–9

def square_them(nums):
    for n in nums:
        yield n ** 2            # step 2: square each number

def only_big(nums, threshold=20):
    for n in nums:
        if n > threshold:
            yield n             # step 3: filter, keep > 20

pipeline = only_big(square_them(read_numbers()))   # chained!
print(list(pipeline))   # [25, 36, 49, 64, 81]   (5²=25, 6²=36, ...)


# ════════════════════════════════════════════════════
# 9. CLASSES & OOP
# ════════════════════════════════════════════════════
# WHAT IS OOP (Object-Oriented Programming)?
#   → Organize code into CLASSES (blueprints) and OBJECTS (instances)
#   → 4 pillars:
#     1. ENCAPSULATION  — bundle data + methods in one class, hide internals
#     2. INHERITANCE    — child class reuses parent class code
#     3. POLYMORPHISM   — same method name, different behavior per class
#     4. ABSTRACTION    — expose only what's needed, hide complexity
#
# WHAT IS A CLASS?
#   → A blueprint/template for creating objects
#   → class Animal: defines the blueprint
#   → cat = Animal("Whiskers","Meow") creates an OBJECT (instance)
#
# KEY METHODS:
#   → __init__    : constructor — runs when object is created
#   → __repr__    : "official" string representation (for debugging)
#   → __str__     : "human-friendly" string (for print())
#   → __len__     : what len(obj) returns
#   → __eq__      : what obj1 == obj2 does
#   → @classmethod: method on the CLASS itself (not instance), gets cls
#   → @staticmethod: utility method, no self or cls needed
#   → @property   : access a method like an attribute (no parentheses)
print("\n" + "=" * 50)
print("CLASSES & OOP")
print("=" * 50)

class Animal:
    kingdom = "Animalia"   # CLASS VARIABLE — shared across all instances
    _count  = 0

    def __init__(self, name, sound):      # runs when Animal("x","y") is called
        self.name  = name                 # INSTANCE VARIABLE — unique per object
        self.sound = sound
        Animal._count += 1               # track how many animals created

    def speak(self):                      # regular instance method — needs self
        return f"{self.name} says {self.sound}"

    @classmethod                          # called on CLASS not instance
    def get_count(cls):                   # cls = the class itself (not an object)
        return cls._count

    @staticmethod                         # no self or cls — just a utility function
    def is_valid_name(name):
        return bool(name and name.isalpha())

    def __repr__(self):    # what you see in the console / debugger
        return f"Animal({self.name!r})"

    def __str__(self):     # what print() shows
        return self.name

    def __len__(self):     # what len() returns
        return len(self.name)

    def __eq__(self, other):   # defines ==
        return isinstance(other, Animal) and self.name == other.name


# INHERITANCE — Dog INHERITS from Animal (gets all Animal's code for free)
class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, sound="Woof")   # call parent's __init__
        self.breed   = breed
        self._tricks = []                      # _ prefix means "private by convention"

    def speak(self):                           # OVERRIDES parent's speak (polymorphism)
        return f"{self.name} ({self.breed}): Woof!"

    def learn(self, trick):
        self._tricks.append(trick)

    @property                      # @property: access like an attribute, not method
    def tricks(self):              # call as dog.tricks (no parentheses)
        return self._tricks.copy() # return copy so original can't be modified directly

    @tricks.setter                 # @property.setter: dog.tricks = [...] calls this
    def tricks(self, new_tricks):
        if not isinstance(new_tricks, list):
            raise TypeError("Must be a list")
        self._tricks = new_tricks


cat = Animal("Whiskers", "Meow")
dog = Dog("Rex", "Labrador")

print(cat.speak())                        # Whiskers says Meow
print(dog.speak())                        # Rex (Labrador): Woof!  ← overridden method
print(repr(cat))                          # Animal('Whiskers')
print(str(cat))                           # Whiskers
print(len(cat))                           # 8   ← len("Whiskers")
print(cat == Animal("Whiskers", "Purr"))  # True  ← our __eq__ compares names
print(Animal.get_count())                 # 2   ← tracked via class variable
print(Animal.is_valid_name("Rex"))        # True
print(Animal.is_valid_name("Rex123"))     # False

dog.learn("sit")
dog.learn("shake")
print(dog.tricks)        # ['sit', 'shake']
dog.tricks = ["roll over"]   # uses the setter
print(dog.tricks)        # ['roll over']

# isinstance = "is this object an instance of this class?"
print(isinstance(dog, Animal))   # True  ← Dog inherits Animal
print(isinstance(dog, Dog))      # True
print(issubclass(Dog, Animal))   # True


# ════════════════════════════════════════════════════
# 10. DATACLASSES
# ════════════════════════════════════════════════════
# WHAT IS A DATACLASS?
#   → A class that AUTO-GENERATES __init__, __repr__, __eq__ for you
#   → Use @dataclass decorator — saves writing boilerplate code
#   → Just declare fields with type hints and defaults
#   → field(default_factory=...) needed for mutable defaults (lists, dicts)
#   → __post_init__ runs AFTER __init__ for validation/extra setup
#   → Use when: you just want to hold data (config, results, records)
print("\n" + "=" * 50)
print("DATACLASSES")
print("=" * 50)

from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    name:          str               # required field (no default)
    learning_rate: float = 0.001     # optional with default
    epochs:        int   = 10
    layers:        List[int] = field(default_factory=lambda: [64, 128, 64])
    # ↑ use field(default_factory=...) for mutable defaults (lists/dicts)
    # NOT layers: List[int] = [64, 128, 64]  ← that would share the same list!

    def __post_init__(self):         # runs after __init__ for validation
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

    @property
    def depth(self):
        return len(self.layers)


cfg  = Config("ResNet", learning_rate=0.01, epochs=50)
print(cfg)        # Config(name='ResNet', learning_rate=0.01, epochs=50, layers=[64,128,64])
print(cfg.depth)  # 3

cfg2 = Config("ResNet", learning_rate=0.01, epochs=50)
print(cfg == cfg2)   # True  ← @dataclass auto-generates __eq__ that compares all fields


# ════════════════════════════════════════════════════
# 11. ERROR HANDLING
# ════════════════════════════════════════════════════
# WHAT IS ERROR HANDLING?
#   → Gracefully handle unexpected situations without crashing the program
#   → try     : code that might raise an error
#   → except  : what to do IF a specific error occurs
#   → else    : runs ONLY if NO error occurred
#   → finally : ALWAYS runs (cleanup — close files, release resources)
#
# COMMON EXCEPTIONS:
#   → ValueError       : wrong value (int("abc"))
#   → TypeError        : wrong type  (1 + "a")
#   → KeyError         : dict key doesn't exist
#   → IndexError       : list index out of range
#   → ZeroDivisionError: divide by zero
#   → FileNotFoundError: file doesn't exist
print("\n" + "=" * 50)
print("ERROR HANDLING")
print("=" * 50)

# CUSTOM EXCEPTION — create your own error types by inheriting Exception
class AppError(Exception):
    def __init__(self, code, message):
        self.code = code
        super().__init__(f"[{code}] {message}")

def divide(a, b):
    try:
        result = a / b                # might raise ZeroDivisionError or TypeError
    except ZeroDivisionError:
        raise AppError("DIV_ZERO", "Cannot divide by zero")
    except TypeError as e:
        raise AppError("TYPE_ERR", str(e))
    else:
        print("  Division successful")   # only runs if NO error
        return result
    finally:
        print("  divide() always runs this")   # ALWAYS runs

try:
    print(divide(10, 2))   # runs fine → 5.0
    print(divide(10, 0))   # raises AppError
except AppError as e:
    print(f"Error code: {e.code}, message: {e}")


# ════════════════════════════════════════════════════
# 12. CONTEXT MANAGERS
# ════════════════════════════════════════════════════
# WHAT IS A CONTEXT MANAGER?
#   → Automatically handles SETUP and CLEANUP using "with" statement
#   → __enter__ runs at the start of the "with" block (setup)
#   → __exit__  runs when the "with" block ends (cleanup — even if error)
#   → Most common use: open files (auto-closes even if exception occurs)
#   → Also used for: DB connections, timers, locks, transactions
#
# WHY USE IT?
#   → Without it: you must manually close/cleanup (easy to forget)
#   → With it:    cleanup happens AUTOMATICALLY
#   → Example: "with open('file.txt') as f:" auto-closes the file
print("\n" + "=" * 50)
print("CONTEXT MANAGERS")
print("=" * 50)

class Timer:
    def __enter__(self):                # runs at start of "with" block
        self.start = time.perf_counter()
        print("  Timer started")
        return self                     # returned as "t" in "with Timer() as t:"

    def __exit__(self, exc_type, exc_val, exc_tb):  # runs when "with" block ends
        self.elapsed = time.perf_counter() - self.start
        print(f"  Timer stopped: {self.elapsed:.4f}s")
        return False   # False = don't suppress exceptions (let them propagate)

with Timer() as t:
    total = sum(range(1_000_000))

print(f"  Sum = {total:,}")
# Timer started
# Timer stopped: 0.0234s
# Sum = 499,999,500,000

# SHORTCUT — use @contextmanager instead of class
# yield acts as the "with" block — code before yield is setup, after is cleanup
from contextlib import contextmanager

@contextmanager
def managed_db(name):
    print(f"  Connecting to {name}")    # __enter__ equivalent
    try:
        yield {"db": name, "connected": True}   # the "as" variable
    finally:
        print(f"  Closing connection to {name}")  # __exit__ equivalent

with managed_db("PostgreSQL") as conn:
    print(f"  Using: {conn}")
# Connecting to PostgreSQL
# Using: {'db': 'PostgreSQL', 'connected': True}
# Closing connection to PostgreSQL


# ════════════════════════════════════════════════════
# 13. COLLECTIONS MODULE
# ════════════════════════════════════════════════════
# WHAT IS THE COLLECTIONS MODULE?
#   → Built-in module with special container types
#   → Counter    : count occurrences of items
#   → defaultdict: dict that auto-creates missing keys with a default value
#   → deque      : double-ended queue — fast append/pop from BOTH ends
#   → namedtuple : (already covered in TUPLES section above)
print("\n" + "=" * 50)
print("COLLECTIONS")
print("=" * 50)

from collections import Counter, defaultdict, deque

# COUNTER — counts how many times each item appears
words   = "the quick brown fox jumps over the lazy dog the".split()
counter = Counter(words)
print(counter.most_common(3))   # [('the', 3), ('quick', 1), ('brown', 1)]
print(counter["the"])           # 3   ← direct count lookup
print(counter["missing"])       # 0   ← returns 0 for missing keys (not KeyError!)

# DEFAULTDICT — like dict, but auto-creates missing keys with a default
# Instead of KeyError when key missing, it creates it with the default value
# defaultdict(list) → missing key creates empty list []
# defaultdict(int)  → missing key creates 0
# defaultdict(set)  → missing key creates empty set
positions = defaultdict(list)
for i, word in enumerate(words):
    positions[word].append(i)     # no KeyError even if word is new
print(positions["the"])           # [0, 6, 9]   ← positions where "the" appears

# DEQUE — double-ended queue, maxlen auto-drops oldest when full
# Faster than list for left-side operations (appendleft, popleft)
# maxlen: if you add beyond the limit, oldest item is automatically removed
window = deque(maxlen=3)          # sliding window of size 3
for i in range(6):
    window.append(i)
    print(list(window))
# [0]
# [0, 1]
# [0, 1, 2]
# [1, 2, 3]  ← 0 was dropped (oldest)
# [2, 3, 4]
# [3, 4, 5]


# ════════════════════════════════════════════════════
# 14. SORTING TRICKS
# ════════════════════════════════════════════════════
# SORTING:
#   → sorted(iterable, key=..., reverse=False) — returns NEW sorted list
#   → list.sort(key=..., reverse=False)        — sorts IN PLACE (no return)
#   → key=lambda x: x["field"]  → sort by a specific field
#   → key=lambda x: (a, b)      → sort by MULTIPLE fields (a first, then b)
#   → Use reverse=True for descending, or -x inside tuple for desc per field
print("\n" + "=" * 50)
print("SORTING")
print("=" * 50)

people = [
    {"name": "Charlie", "age": 30, "score": 92},
    {"name": "Alice",   "age": 25, "score": 95},
    {"name": "Bob",     "age": 30, "score": 87},
]

# Sort by ONE field
by_score = sorted(people, key=lambda x: x["score"], reverse=True)  # highest first
print([p["name"] for p in by_score])   # ['Alice', 'Charlie', 'Bob']

# Sort by MULTIPLE fields: age ascending, then score descending
# Trick: -x["score"] makes it sort descending for that field
multi = sorted(people, key=lambda x: (x["age"], -x["score"]))
print([p["name"] for p in multi])   # ['Alice', 'Charlie', 'Bob']
# ↑ Alice (25) first, then Charlie (30, score 92) before Bob (30, score 87)

# Sort strings by length, then alphabetically (2-level sort)
words = ["banana", "apple", "cherry", "fig", "date"]
print(sorted(words, key=lambda x: (len(x), x)))
# ['fig', 'date', 'apple', 'banana', 'cherry']


# ════════════════════════════════════════════════════
# 15. ITERTOOLS
# ════════════════════════════════════════════════════
# WHAT IS ITERTOOLS?
#   → Built-in module for EFFICIENT looping / combinatorics
#   → All functions return ITERATORS (memory efficient, like generators)
#   → chain       : combine multiple iterables into one
#   → combinations: all unique pairs/groups (order doesn't matter)
#   → permutations: all ordered arrangements (order DOES matter)
#   → product     : cartesian product (like nested for loops)
#   → accumulate  : running totals/products
print("\n" + "=" * 50)
print("ITERTOOLS")
print("=" * 50)

import itertools

# CHAIN — combine multiple lists into one sequence
print(list(itertools.chain([1,2], [3,4], [5,6])))
# [1, 2, 3, 4, 5, 6]

# COMBINATIONS — all unique groups of r items (like picking cards)
# Order doesn't matter: (1,2) and (2,1) are the SAME combination
print(list(itertools.combinations([1,2,3,4], 2)))
# [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]

# PERMUTATIONS — all ordered arrangements of r items
# Order MATTERS: (1,2) and (2,1) are DIFFERENT permutations
print(list(itertools.permutations([1,2,3], 2)))
# [(1,2),(1,3),(2,1),(2,3),(3,1),(3,2)]

# PRODUCT — cartesian product (like nested for loops)
# repeat=3 means 3 nested loops over [0,1] → binary combinations
print(list(itertools.product([0,1], repeat=3)))
# [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]

# ACCUMULATE — running totals (or any running operation)
import operator
print(list(itertools.accumulate([1,2,3,4,5])))                    # running sum: [1,3,6,10,15]
print(list(itertools.accumulate([1,2,3,4,5], operator.mul)))      # running product: [1,2,6,24,120]


print("\nAll done! ✓")
