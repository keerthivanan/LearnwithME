# Module 01 — Practice Answers

Try the exercises yourself first, then check here.

---

**1. Word count**
```python
def word_count(text):
    return len(text.split())

print(word_count("hello there friend"))   # 3
```
`text.split()` breaks the string into a list of words; `len()` counts them.

---

**2. Shipment class**
```python
class Shipment:
    def __init__(self, tracking_number, status):
        self.tracking_number = tracking_number
        self.status = status

    def describe(self):
        return f"{self.tracking_number} is {self.status}"

s = Shipment("1234", "out for delivery")
print(s.describe())     # 1234 is out for delivery
```

---

**3. Prices above 20 (comprehension)**
```python
prices = [10, 25, 5, 40]
high = [p for p in prices if p > 20]
print(high)             # [25, 40]
```

---

**4. try/except**
```python
try:
    n = int("hello")
except ValueError as e:
    print(f"That's not a number: {e}")
# That's not a number: invalid literal for int() with base 10: 'hello'
```

---

**5. Concurrent async**
```python
import asyncio

async def nap(name):
    print(f"{name} sleeping")
    await asyncio.sleep(1)
    print(f"{name} awake")
    return name

async def main():
    result = await asyncio.gather(nap("A"), nap("B"))
    print(result)       # ['A', 'B']  — ~1 second total

asyncio.run(main())
```
Both sleep at the same time → ~1s total, not 2s. That's the power of async: it doesn't wait idle.

---

## If any of these felt hard
Go back to that section in `01_python_for_ai.md` and re-type the example. Repetition builds fluency. Don't move on until these feel comfortable — everything ahead uses them.
