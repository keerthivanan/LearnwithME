# Langflow — Start Here 🚀

Your complete, structured course to learn Langflow from zero. Read the files in order.

---

## What is Langflow (in one line)
Langflow is an **open-source, visual drag-and-drop builder for AI/LLM applications** — built on top of LangChain. You connect "components" on a canvas to build chatbots, RAG systems, and AI agents **without writing much code**.

Think: **n8n, but built specifically for AI/LLM workflows.** (You already know n8n from your DHL project — this will click fast.)

---

## Why learn Langflow?
- It's one of the most popular tools for **prototyping AI agents and RAG** visually
- Backed by **DataStax / IBM** — used in real companies
- Great for interviews: shows you can build GenAI apps, not just talk about them
- You can export flows as **JSON**, run them as an **API**, or as an **MCP server**

---

## Your learning path (read in this order)
| File | What you'll learn |
|------|-------------------|
| **00_START_HERE.md** (this) | Overview + how to run Langflow |
| **01_core_concepts.md** | Components, flows, ports, the canvas |
| **02_build_your_first_flow.md** | Build a working chatbot step by step |
| **03_components_deep_dive.md** | Every key component explained |
| **04_rag_and_agents.md** | RAG pipelines + AI agents with tools |
| **05_langflow_vs_n8n.md** | Map it to what you already know (n8n) |
| **06_interview_qa.md** | Interview questions & answers |

---

## How to run Langflow on your laptop

You already installed it with `uv`. To start it:

```powershell
langflow run
```

Then open **`http://localhost:7860`** in your browser.

**If `langflow` isn't found**, use the full path:
```powershell
& "C:\Users\91709\.local\bin\langflow.exe" run
```

**If it won't start** (port busy or error), try a different port:
```powershell
& "C:\Users\91709\.local\bin\langflow.exe" run --port 7861
```
Then open `http://localhost:7861`.

> First start takes 1-3 minutes (it sets up its database and interface). Be patient the first time.

---

## The mental model (remember this)

```
Langflow = a canvas where you drag COMPONENTS and connect them with wires.

  [Chat Input] → [Prompt] → [Language Model] → [Chat Output]
                                  ↑
                            (your OpenAI key)

  Each box = a component (a node).
  Each wire = data flowing from one component to the next.
  The whole thing = a "Flow".
  Test it in the "Playground" (a chat window).
```

That's 80% of Langflow. The rest is just learning which components exist and how to wire them.

---

## What you can build with Langflow
- **Chatbots** (with memory)
- **RAG systems** (chat with your documents)
- **AI Agents** (that use tools — search, calculator, APIs)
- **Multi-step pipelines** (extract → transform → generate)
- **APIs** (turn any flow into a REST endpoint)

---

Next → open **01_core_concepts.md**
