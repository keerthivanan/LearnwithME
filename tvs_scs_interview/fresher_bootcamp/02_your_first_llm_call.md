# Module 02 — Your First LLM Call (Theory + Code)

Now you talk to an actual LLM in Python and understand what's happening.

---

## 🧠 THEORY: what is an LLM, really?

An **LLM (Large Language Model)** is a program trained on massive text that does ONE thing: **predict the next token** (a token ≈ ¾ of a word). You give it text; it continues that text, one token at a time, based on probabilities.

Key truths to hold:
- **It's stateless.** Each API call is independent — it has NO memory of past calls unless YOU send the history back.
- **You pay per token** (input + output). More text = more cost.
- **It's non-deterministic** (unless temperature=0) — same prompt can give different answers.

### Tokens
Text is broken into tokens. "Hello world" ≈ 2 tokens. "unbelievable" might be 3 (un-believ-able). Roughly **1 token ≈ 4 characters ≈ ¾ word**. You'll hear "context window = 128,000 tokens" — that's the max text (input + output) per call.

### Temperature
Controls randomness:
- **0.0** = deterministic, picks the most likely token → use for **facts, extraction, classification**
- **0.7** = creative, varied → use for **writing, brainstorming**
- Rule for production automations: keep it **low (0–0.2)** for consistency.

---

## 💻 CODE: call the LLM

Install + run:
```powershell
$env:OPENAI_API_KEY = "your-key"
uv run --with openai python first_call.py
```

`first_call.py`:
```python
from openai import OpenAI

client = OpenAI()     # reads OPENAI_API_KEY from the environment

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Answer in one sentence."},
        {"role": "user", "content": "What is RAG in AI?"}
    ],
    temperature=0.2,
)

print(response.choices[0].message.content)
```

### Understand every part:
- **`messages`** is a list of turns. Each has a **role**:
  - `system` = the rules/persona (set once, steers everything)
  - `user` = what the human says
  - `assistant` = what the model said (you add these to give it history)
- **`response.choices[0].message.content`** = the actual text answer (buried in the response object)
- **`temperature=0.2`** = keep it consistent

---

## 💻 CODE: give it MEMORY (multi-turn)

Since the LLM is stateless, YOU keep the conversation and resend it:
```python
from openai import OpenAI
client = OpenAI()

messages = [{"role": "system", "content": "You are a concise assistant."}]

def chat(user_text):
    messages.append({"role": "user", "content": user_text})     # add user turn
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    answer = resp.choices[0].message.content
    messages.append({"role": "assistant", "content": answer})   # add AI turn (memory!)
    return answer

print(chat("My name is Keerthi."))
print(chat("What's my name?"))     # It remembers — because we resent the history
```
**This is the whole secret of chatbot "memory":** you store the messages list and send it every time. When it gets too long, you trim or summarize (that's "context window management").

---

## 💻 CODE: see token usage & cost
```python
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hi"}],
)
print(resp.usage)     # prompt_tokens, completion_tokens, total_tokens
```
`gpt-4o-mini` is ~$0.15 per 1M input tokens — extremely cheap. Watching `usage` is how you monitor cost in production.

---

## 🧠 THEORY: streaming (why chat UIs feel fast)
Instead of waiting for the full answer, you can **stream** tokens as they generate:
```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Count to 5 slowly"}],
    stream=True,
)
for chunk in stream:
    piece = chunk.choices[0].delta.content or ""
    print(piece, end="", flush=True)     # prints as it arrives
```
That's why ChatGPT types word-by-word — it's streaming. In a web app you'd stream over WebSocket/SSE (backend module).

---

## ✏️ PRACTICE
1. Change the system prompt to "You are a pirate" and ask a question. See how the system role steers everything.
2. Ask the same question at `temperature=0` twice, then at `temperature=1` twice. Notice 0 is consistent, 1 varies.
3. Build the memory `chat()` function and confirm it remembers your name across calls.
4. Print `resp.usage` and note the token counts.

---

## ✅ You now know
What an LLM is (next-token predictor, stateless, priced per token), tokens, temperature, the messages/roles structure, how "memory" actually works, cost tracking, and streaming. **You've talked to an LLM in code.**

Next → `03_prompt_engineering.md`
