# Fresher GenAI Bootcamp — Start Here

This is a **teaching course** — not definitions. Each module EXPLAINS the idea (theory), shows you the CODE (coding), and gives you PRACTICE. Do them in order. By the end you'll be strong in both.

---

## How this works (read this)
Every module has 3 parts:
1. **🧠 THEORY** — the idea explained simply, with *why* it matters
2. **💻 CODE** — real, runnable Python you type yourself
3. **✏️ PRACTICE** — small exercises to lock it in

**Golden rule: TYPE the code yourself, don't copy-paste.** Your fingers learn what your eyes skip.

---

## The path (each builds on the last)
| Module | You'll learn | Coding or Theory |
|--------|-------------|------------------|
| **01 — Python for AI** | The Python you actually need (funcs, classes, errors, async) | Coding |
| **02 — Your First LLM Call** | Tokens, temperature, calling GPT in code | Both |
| **03 — Prompt Engineering** | Getting reliable output from an LLM | Both |
| **04 — Embeddings & Vectors** | What a vector *really* is + similarity search | Both |
| **05 — Build a RAG System** | Chat with your own PDF (the #1 project) | Coding |
| **06 — Build an Agent** | An LLM that uses tools (reason-act loop) | Coding |
| **07 — Serve with FastAPI** | Turn it into a real API | Coding |

---

## Setup (do this once)
You already have Python + uv. To run any example:
```powershell
uv run --with openai --with <other-packages> python your_file.py
```
For LLM calls you'll need an **OpenAI API key** (you have one). Set it:
```powershell
$env:OPENAI_API_KEY = "your-key-here"
```

---

## The mindset for a fresher
- **Understand, then build.** Read the THEORY, then DO the CODE. Both, always.
- **It's okay to be slow.** Understanding beats speed.
- **Break things.** Change the code, see what happens — that's how you learn.
- **Explain it back.** If you can teach a concept to someone else, you know it.

---

## What "strong in both" means (your goal)
- **Theory strong:** you can explain *what* RAG/agents/embeddings are and *why* — in an interview, clearly.
- **Coding strong:** you can *build* a small version yourself — in Python — from scratch.

This course gives you both. Let's go.

Next → `01_python_for_ai.md`
