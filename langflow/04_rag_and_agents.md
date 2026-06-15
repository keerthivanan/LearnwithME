# RAG & Agents in Langflow

The two most important things to build — and the two recruiters ask about most.

---

# PART 1: RAG (Chat With Your Documents)

**RAG = Retrieval-Augmented Generation.** It lets the AI answer using YOUR documents (PDFs, policies, manuals) instead of just its training data.

## The RAG flow has TWO parts

### Part A — Ingestion (load your docs into a vector store) — done once
```
[File: policy.pdf] → [Split Text] → [Embeddings] → [Vector Store]
```
- **File**: loads your PDF
- **Split Text**: chops it into small chunks (e.g., 1000 characters each)
- **Embeddings**: turns each chunk into a vector (list of numbers capturing meaning)
- **Vector Store**: saves the vectors so you can search them

### Part B — Retrieval + Answer (every time a user asks) — runs per question
```
[Chat Input] → [Embeddings] → [Vector Store search] → [Prompt with {context}] → [Model] → [Chat Output]
```
- User's question gets embedded
- Vector Store finds the most similar chunks (the relevant context)
- Those chunks go into the Prompt as `{context}`
- The Model answers using ONLY that context

## The RAG prompt (the key piece)
```
Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have that information."

Context:
{context}

Question:
{question}
```

## Why RAG matters
The AI now answers about YOUR specific data — your company policies, your product docs — accurately, with no hallucination. This is the #1 enterprise GenAI use case.

> **Connection to your DHL project:** RAG is how you'd let the DHL agent answer from real DHL policy documents instead of hard-coded rules.

---

# PART 2: AGENTS (AI That Uses Tools)

An **Agent** is an AI that can DECIDE to use tools to get information or take action — instead of just replying from memory.

## The simplest agent flow
```
[Chat Input] → [Agent] → [Chat Output]
                  ↑
            [Tools: Search, Calculator, API Request...]
```

You connect **tools** to the Agent. When a user asks something, the agent **reasons**:
- "This needs current info → I'll use the Search tool"
- "This needs math → I'll use the Calculator"
- "This needs the DHL status → I'll use the API Request tool"

It calls the tool, gets the result, and answers. **This is exactly the pattern from your DHL Command Center** — the router + agents + tools.

## Building an agent in Langflow
1. Drag the **Agent** component
2. Set its model (OpenAI) + a system prompt ("You are a DHL support agent...")
3. Drag **tools** (e.g., API Request, Search) and connect them to the Agent's **tools** port
4. Connect Chat Input → Agent → Chat Output
5. Test in Playground

## Agent vs plain Model — when to use which?
| Use a plain **Model** when... | Use an **Agent** when... |
|------------------------------|--------------------------|
| Simple Q&A or text generation | The AI needs to fetch live data or take actions |
| No external data needed | It needs tools (search, APIs, calculations) |
| Fastest, cheapest | More powerful, handles multi-step tasks |

---

## Multi-agent in Langflow
You can connect **multiple agents**, or use an agent whose tools are *other flows*. Langflow supports the "agent as tool" pattern — one orchestrator agent calling specialist sub-flows. (Same idea as your DHL triage router → 5 specialists.)

---

## The mental shortcut
- **RAG** = give the AI the right *knowledge* (from your docs)
- **Agent** = give the AI the right *abilities* (tools/actions)
- **Combine them** = an agent that can both look things up in your docs AND take actions. That's a production AI assistant.

---

Next → **05_langflow_vs_n8n.md**
