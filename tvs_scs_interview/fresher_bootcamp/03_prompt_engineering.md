# Module 03 — Prompt Engineering (Theory + Code)

How to get **reliable, correct** output from an LLM. This is a real skill — and heavily tested.

---

## 🧠 THEORY: what prompt engineering actually is
The LLM does whatever your instructions steer it toward. **Prompt engineering = writing instructions that get the exact output you need, every time.** Vague prompt → vague/inconsistent output. Precise prompt → reliable output.

---

## The techniques (each with a real before/after)

### 1. Be specific + give a role (system prompt)
❌ Vague:
```
"Tell me about this order."
```
✅ Specific:
```
"You are a logistics assistant. Given the order text, extract the tracking number,
the destination city, and the status. Reply in one short sentence."
```
The **system prompt** sets the persona + rules once and steers every reply.

### 2. Force the OUTPUT FORMAT (huge for real apps)
❌ "Extract the invoice details." → you get a paragraph you can't parse.
✅
```
Extract these fields and return ONLY valid JSON:
{"vendor": "...", "amount": 0.0, "date": "YYYY-MM-DD"}
If a field is missing, use null. Do not add any text outside the JSON.
```
Now you get parseable JSON. **This is how real systems get structured data.**

### 3. Few-shot (show examples)
Give 2-3 input→output examples so the model copies the pattern:
```
Classify the message intent as: tracking, booking, or complaint.

Message: "Where is my parcel?"        Intent: tracking
Message: "I want to send a box"       Intent: booking
Message: "My package arrived broken"  Intent: complaint

Message: "When will it arrive?"        Intent:
```
The model completes: `tracking`. Few-shot dramatically improves consistency.

### 4. Chain-of-Thought (make it reason)
For anything needing logic, add:
```
"Think step by step before giving the final answer."
```
The model reasons first → far fewer mistakes on math/logic/multi-step tasks.

### 5. Grounding (kill hallucination)
```
"Answer ONLY using the context below. If the answer is not in the context,
say 'I don't have that information.' Do not use outside knowledge.

Context:
{retrieved_text}

Question: {question}"
```
This is the heart of RAG (next modules). Grounding stops the model from making things up.

### 6. Use delimiters (separate instructions from data)
```
Summarize the text between triple quotes.
"""
{user_text}
"""
```
Delimiters prevent the user's text from being read as instructions (basic prompt-injection defense).

---

## 💻 CODE: structured extraction (the money skill)
```python
from openai import OpenAI
import json
client = OpenAI()

invoice_text = "Invoice from ABC Corp, #INV-2024-99, dated 2024-03-15, total $1,250.00"

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0,
    response_format={"type": "json_object"},          # forces valid JSON
    messages=[
        {"role": "system", "content":
            "Extract invoice fields. Return ONLY JSON with keys: vendor, invoice_number, date, amount. "
            "Use null for missing fields. Amount as a number, no currency symbol."},
        {"role": "user", "content": invoice_text},
    ],
)

data = json.loads(resp.choices[0].message.content)   # now it's a Python dict
print(data["vendor"])     # ABC Corp
print(data["amount"])     # 1250.0
```
Notice: `temperature=0` (consistency) + `response_format=json_object` (guaranteed JSON) + a precise system prompt. **This pattern powers real document-processing systems.**

---

## 🧠 THEORY: context window management
Everything you send counts toward the token limit: system prompt + few-shot + conversation history + retrieved context + the question. As chats grow, you must:
- **Trim** old turns, or **summarize** them
- Retrieve **only the most relevant** context (don't stuff everything)
- Put key instructions at the **start and end** (models weight the edges)

---

## 🧠 THEORY: reducing hallucination (interview favorite)
The model predicts *plausible* text, not *true* text. To keep it factual:
1. **Ground it** — "answer only from the context"
2. **Give it tools/RAG** — real data instead of guessing
3. **Low temperature**
4. **Ask it to say "I don't know"** when unsure
5. **Validate the output** (schema/rules)

---

## ✏️ PRACTICE
1. Write a prompt that classifies a sentence's sentiment as ONLY `positive`, `negative`, or `neutral` (no other words). Test 3 sentences.
2. Write a structured-extraction prompt that pulls `name`, `email`, `phone` from a messy text into JSON. Use `response_format=json_object` + temperature 0.
3. Take a paragraph and write a grounded prompt: answer a question using ONLY that paragraph; test a question whose answer ISN'T in it — confirm it says "I don't have that information."
4. Add 3 few-shot examples to improve any of the above.

---

## ✅ You now know
Role/system prompts, forcing output format (JSON), few-shot, chain-of-thought, grounding, delimiters, context management, and hallucination reduction — plus the code pattern for reliable structured extraction. **This is a core, testable skill.**

Next → `04_embeddings_and_vectors.md` (then you build RAG)
