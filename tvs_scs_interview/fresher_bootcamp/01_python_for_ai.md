# Module 01 — Python for AI (Taught, with Code)

The exact Python you need for GenAI work. Type every example yourself.

---

## 🧠 THEORY: why Python?
Almost all AI/LLM tooling is Python (OpenAI SDK, LangChain, FastAPI). You don't need to be a Python master — you need to be **solid** on the building blocks below. Let's learn each.

---

## 1. Variables & data types
```python
name = "Keerthi"        # str (text)
age = 24                # int (whole number)
price = 9.99            # float (decimal)
is_ready = True         # bool (True/False)
skills = ["python", "rag", "agents"]   # list (ordered, changeable)
person = {"name": "Keerthi", "role": "AI Engineer"}  # dict (key→value)
```
**Why it matters:** LLM APIs give you back **dicts** and **lists** (JSON). You'll read them constantly.

```python
print(person["name"])        # Keerthi   — get a value by key
print(skills[0])             # python    — get by index (starts at 0)
skills.append("fastapi")     # add to a list
```

---

## 2. Functions — reusable blocks
```python
def greet(name):
    return f"Hello, {name}!"     # f-string inserts the variable

message = greet("Keerthi")
print(message)                   # Hello, Keerthi!
```
- `def` defines it, `return` gives back a value
- **Default arguments:**
```python
def ask(question, model="gpt-4o-mini"):
    return f"Asking {model}: {question}"
print(ask("hi"))                 # uses default model
print(ask("hi", "gpt-4o"))       # override
```
**Why it matters:** every tool an agent calls is a function.

---

## 3. Conditionals & loops
```python
score = 0.8
if score > 0.9:
    print("high")
elif score > 0.5:
    print("medium")     # this runs
else:
    print("low")

for skill in skills:            # loop over a list
    print(skill)

for i in range(3):              # 0, 1, 2
    print(i)
```

---

## 4. Classes & objects (OOP) — the blueprint idea
```python
class Agent:
    def __init__(self, name, model):   # runs when you create one
        self.name = name               # store data on the object
        self.model = model

    def run(self, task):               # a method (function on the object)
        return f"{self.name} doing '{task}' with {self.model}"

a = Agent("Researcher", "gpt-4o-mini")   # create an object (instance)
print(a.run("find news"))                # Researcher doing 'find news' with gpt-4o-mini
```
- `class` = blueprint, object = one made from it
- `__init__` = the setup method; `self` = "this object"
- **Inheritance** (build on another class):
```python
class ResearchAgent(Agent):        # inherits everything from Agent
    def search(self):
        return f"{self.name} searching..."
```
**Why it matters:** LangChain, FastAPI, Pydantic — all classes. You'll subclass and use them.

---

## 5. Error handling — don't let it crash
```python
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Caught error: {e}")     # Caught error: division by zero
finally:
    print("always runs")
```
**Why it matters:** LLM/API calls fail (timeouts, rate limits). Real code wraps them in try/except and handles failure gracefully — a big interview point.

---

## 6. Comprehensions — Pythonic loops
```python
nums = [1, 2, 3, 4]
squares = [n*n for n in nums]              # [1, 4, 9, 16]
evens = [n for n in nums if n % 2 == 0]    # [2, 4]
```
Cleaner than a for-loop for building lists. You'll use these to transform API results.

---

## 7. Working with JSON (critical for AI)
```python
import json

data = {"topic": "AI", "tags": ["llm", "rag"]}
text = json.dumps(data)          # dict → JSON string
back = json.loads(text)          # JSON string → dict
print(back["tags"][1])           # rag
```
**Why it matters:** LLM structured outputs and API responses are JSON. You constantly convert dict ↔ JSON.

---

## 8. Calling an API with requests
```python
import requests
r = requests.get("https://api.github.com/users/keerthivanan")
print(r.status_code)             # 200 = OK
print(r.json()["public_repos"])  # read a field from the JSON response
```

---

## 9. async / await — the big one (do this slowly)

### 🧠 Theory (the toast analogy)
Normal code **waits and freezes** on slow things (API calls). Async lets your program **do other work while waiting** — like starting the eggs while the toast toasts. It doesn't work *faster*; it stops *wasting time waiting*.

**Use async for I/O-bound work** (LLM calls, DB, network) — where you wait on something external.

```python
import asyncio

async def slow_task(name, seconds):
    print(f"{name} started")
    await asyncio.sleep(seconds)     # 'await' = pause here, let others run
    print(f"{name} done")
    return name

async def main():
    # run BOTH at the same time instead of one after the other
    results = await asyncio.gather(
        slow_task("A", 2),
        slow_task("B", 2),
    )
    print(results)                   # ['A', 'B']  — total ~2s, not 4s

asyncio.run(main())
```
- `async def` = a function that can pause/resume
- `await` = "pause here until this finishes; meanwhile do other work"
- `asyncio.gather(...)` = run several things concurrently
**Why it matters:** an async web server (FastAPI) handles many users at once because it never sits frozen waiting on the LLM. This is the #1 backend concept for AI services.

---

## ✏️ PRACTICE (do these — type them)
1. Write a function `word_count(text)` that returns the number of words. (Hint: `text.split()`)
2. Make a class `Shipment` with `tracking_number` and `status`, and a method `describe()` that returns `"1234 is out for delivery"`.
3. Given `prices = [10, 25, 5, 40]`, use a comprehension to get only prices above 20.
4. Wrap `int("hello")` in try/except and print a friendly message.
5. Write two async functions that "sleep" 1 second each and run them concurrently with `asyncio.gather`. Confirm it takes ~1s, not 2s.

**Answers are in `01_practice_answers.md`** — but try first!

---

## ✅ You now know
Variables, functions, conditionals/loops, classes+inheritance, error handling, comprehensions, JSON, API calls, and async. **That's the Python foundation for everything ahead.**

Next → `02_your_first_llm_call.md`
