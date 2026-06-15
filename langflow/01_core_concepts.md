# Langflow Core Concepts

Master these 7 concepts and you understand Langflow.

---

## 1. The Canvas (Workspace)
The big empty area where you build. You drag components onto it and connect them. Like a whiteboard for your AI app.

---

## 2. Components (the building blocks)
A **component** is a single box that does ONE job. Examples:
- **Chat Input** — where the user's message enters
- **Prompt** — a template for instructions to the AI
- **OpenAI / Language Model** — the AI brain
- **Chat Output** — where the AI's reply comes out

You find all components in the **left sidebar**, grouped by category (Inputs, Outputs, Models, Agents, Data, Vector Stores, etc.).

Each component has:
- **Input ports** (left side) — data coming IN
- **Output ports** (right side) — data going OUT
- **Fields** — settings you fill in (like the model name, temperature, your API key)

---

## 3. Ports & Edges (the wires)
- A **port** is a connection point on a component (a little circle).
- An **edge** is the wire you draw between two ports.
- You connect an **output port** of one component to an **input port** of another.

**Important:** ports are **typed**. A "Message" output connects to a "Message" input. Langflow color-codes them so you know what connects to what. If a wire won't connect, the types don't match.

```
[Chat Input] ──Message──→ [Prompt] ──Message──→ [Model] ──Message──→ [Chat Output]
```

---

## 4. Flow (your whole app)
A **Flow** is the complete connected graph of components — your finished AI app. You can have many flows, each saved as a project. A flow can be exported as a **JSON file** (and shared/imported, like your n8n workflows).

---

## 5. The Playground (test it live)
The **Playground** is a built-in chat window to test your flow. Click **"Playground"** (top right), type a message, and watch your flow run. This is where you see if it works — like the chat UI in your DHL project.

---

## 6. Global Variables (secrets & reuse)
Instead of pasting your OpenAI key into every component, store it once as a **Global Variable** (Settings → Global Variables → add `OPENAI_API_KEY`). Then reference it everywhere. Keeps secrets safe and reusable.

---

## 7. Tweaks & API
Every flow can be:
- **Run via API** — Langflow gives you a code snippet (Python/JS/curl) to call your flow from anywhere
- **Tweaked** — override component settings at run-time without editing the flow
- **Published as MCP** — exposed as an MCP server so AI assistants can use it

---

## The 3 most common component categories you'll use

| Category | Examples | Purpose |
|----------|----------|---------|
| **Inputs/Outputs** | Chat Input, Chat Output, Text Input | Get data in and out |
| **Models** | OpenAI, Anthropic, Ollama | The AI brain |
| **Prompts** | Prompt template | Tell the AI what to do |

Add **Memory**, **Agents**, **Tools**, **Vector Stores**, and **Embeddings** as you grow.

---

## Visual: the simplest possible flow

```
┌────────────┐     ┌──────────┐     ┌─────────────────┐     ┌─────────────┐
│ Chat Input │────→│  Prompt  │────→│ Language Model  │────→│ Chat Output │
└────────────┘     └──────────┘     └─────────────────┘     └─────────────┘
                        │                    │
                   "You are a           (OpenAI key,
                    helpful bot..."      temperature)
```

This is a working chatbot. You'll build exactly this in the next file.

---

Next → **02_build_your_first_flow.md**
