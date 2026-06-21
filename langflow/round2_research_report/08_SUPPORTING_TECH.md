# 08 — Supporting Tech Fundamentals (Know These Cold)

The interviewer will dip into these to check your foundations. Each explained simply with why it matters here.

---

## 1. APIs & REST

**API (Application Programming Interface):** rules that let two programs talk and exchange data. When your MCP server calls DuckDuckGo or Gmail, it uses their APIs.

**REST:** the most common style of web API. Key ideas:
- **Resources** are addressed by **URLs** (e.g., `/api/v1/flows`)
- You act on them with **HTTP methods**
- It's **stateless** (each request stands alone)
- Data is usually **JSON**

**HTTP methods (memorize):**
| Method | Action | Example |
|--------|--------|---------|
| GET | Read | get a list of flows |
| POST | Create | create a new flow |
| PUT / PATCH | Update | update a flow |
| DELETE | Remove | delete a flow |

**HTTP status codes (know the common ones):**
- `200` OK, `201` Created
- `401` Unauthorized, `403` Forbidden (no permission)
- `404` Not Found
- `422` Unprocessable (bad input)
- `500` Server Error

> *(Fun fact: that's exactly how I built your flow — POST to Langflow's REST API. The 403 we hit early was "need an API key"; 422 was "command not allowed".)*

---

## 2. JSON

**JSON (JavaScript Object Notation):** a lightweight text format for data — key/value pairs and arrays. Used everywhere to send data between systems.

```json
{
  "topic": "AI in Logistics",
  "email": "user@gmail.com",
  "sections": ["Summary", "Key Findings", "References"]
}
```
- `{}` = object (key/value), `[]` = array (list)
- Values: string, number, boolean, null, object, array

**Why it matters here:** MCP messages, Langflow flows, and API requests are all JSON.

---

## 3. Python essentials

**Function:**
```python
def web_search(query):
    return results
```

**Class & Object (OOP):**
```python
class Agent:          # blueprint
    def __init__(self, name):
        self.name = name
    def run(self):
        ...

a = Agent("Research")  # object (instance)
```

**Inheritance:** one class builds on another.
```python
class ResearchAgent(Agent):   # inherits Agent's behavior
    pass
```

**Decorator:** wraps a function to add behavior. You used this!
```python
@mcp.tool()            # registers the function as an MCP tool
def web_search(query): ...
```

**Error handling:** catch failures so the program doesn't crash.
```python
try:
    send_email(...)
except Exception as e:
    return f"Failed: {e}"
```

**Environment variables:** keep secrets out of code.
```python
import os
password = os.environ.get("GMAIL_APP_PASSWORD")
```

---

## 4. LLM concepts

**LLM:** a model that predicts text. Text in → text out. It has no memory and can't act on its own.

**Prompt engineering:** writing clear instructions to get reliable output. Your Research Agent prompt forces the exact report structure.

**System prompt vs user input:** the system prompt sets the role/rules ("You are a Research Agent..."); the user input is the actual task ("topic: AI").

**Tool calling (function calling):** instead of answering, the LLM outputs a request to call a tool with arguments; the framework runs it and feeds back the result. **This is the engine of agents and of MCP.**

**Temperature:** randomness knob. Low (0–0.2) = factual/consistent (use for research/extraction); high = creative.

**Tokens:** chunks of text the model reads/writes; you pay per token. ~4 chars ≈ 1 token.

**Hallucination:** the model inventing facts. Reduce with grounding ("answer only from provided context/tool results"), RAG, and using tools for real data.

---

## 5. RAG (Retrieval-Augmented Generation)

**What:** before answering, fetch relevant documents and put them in the prompt, so the LLM answers from YOUR data.

**Pipeline:**
```
Documents → split into chunks → embed (turn into vectors) → store in a vector DB
User question → embed → search vector DB → top chunks → prompt → LLM answers
```

**Why it matters:** it's the #1 enterprise GenAI pattern, and a likely follow-up question. Your assignment is *tool-augmented* (search) rather than *document-augmented* (RAG) — but the idea of "give the model real info before it answers" is the same.

---

## 6. Embeddings & Vectors (quick)

**Embedding:** turning text into a list of numbers (a vector) that captures meaning. Similar meanings → nearby vectors. Used in RAG to find relevant chunks by similarity.

**Vector database:** stores vectors and finds the most similar ones fast (e.g., Chroma, FAISS, Pinecone).

---

## 7. Webhooks vs APIs (quick)

- **API call:** YOU ask the other system for data (you pull).
- **Webhook:** the other system notifies YOU when something happens (it pushes).

---

## 8. The words to drop naturally (to sound senior)
- "tool calling", "agent loop", "separation of concerns"
- "STDIO vs SSE transport", "JSON-RPC", "decoupling"
- "grounding to reduce hallucination", "stateless", "idempotent"
- "environment variables for secrets", "graceful error handling"

---

## 9. If they ask something you don't know
> "I haven't worked with that specific piece yet, but based on [related thing I do know], I'd approach it by [reasoned guess]. Is that the direction your team uses?"

Honesty + reasoning beats bluffing every time.

---

## ✅ You now have the full stack for this round:
1. Agentic AI (`01`) · 2. MCP deep dive (`02`) · 3. The servers (`03`)
4. Build guide (`04`) · 5. Full trace (`05`) · 6. Mega Q&A (`07`) · 7. Fundamentals (this)

Read `02` and `07` twice. That's where this interview is won. 🚀
