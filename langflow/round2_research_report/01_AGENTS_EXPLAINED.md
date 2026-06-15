# 01 — AGENTS EXPLAINED (Deeply)

Everything about agents, agentic AI, and multi-agent orchestration — from zero.

---

## 1. What is an LLM vs. an Agent?

**A plain LLM** (like GPT-4o) only does ONE thing: text in → text out. Ask it "what's the weather in Chennai today?" and it CAN'T actually check — it just guesses from old training data.

**An agent** is an LLM **plus the ability to use tools and decide when to use them**. Give the agent a "weather tool," and now when you ask about the weather, it *reasons*: "I need live data → I'll call the weather tool → here's the answer."

> **Definition to memorize:** An agent is an LLM that can reason about a goal, choose tools to act in the world, observe the results, and loop until the goal is done.

---

## 2. The agent loop (how an agent actually "thinks")

This is the heart of agentic AI. An agent runs a loop:

```
1. THINK    — "What do I need to do? Do I need a tool?"
2. ACT      — call a tool (e.g., web_search("quantum computing 2025"))
3. OBSERVE  — read the tool's result
4. REPEAT   — think again with the new info; call another tool if needed
5. ANSWER   — when it has enough, produce the final answer
```

This is often called the **ReAct pattern** (Reason + Act). The LLM decides each step; the tools do the real-world actions.

**Example for your Research Agent:**
```
THINK:   "I need current info on the topic."
ACT:     web_search("quantum computing breakthroughs 2025")
OBSERVE: [5 search results with titles, snippets, URLs]
THINK:   "I have enough. Now write the report."
ANSWER:  Summary / Key Findings / References
```

---

## 3. What makes something "agentic AI"?

A workflow is **agentic** when the AI itself **decides what to do**, instead of you hard-coding every step.

- **Not agentic:** "Always call search, then always summarize." (fixed script)
- **Agentic:** "Here's a goal and some tools — figure out how to achieve it." (the AI decides which tools, how many times, in what order)

Your assignment is agentic because each agent *decides* when to call its tool.

---

## 4. Tools — what they really are

A **tool** is just a function the agent can call, described in a way the LLM understands:
- a **name** (`web_search`)
- a **description** ("search the web for current info")
- an **input schema** (it takes a `query` string)

The LLM reads these descriptions and decides which tool fits the task. The tool runs real code (an actual web search) and returns a result the LLM reads.

> Your two tools: `web_search` (Research Agent) and `send_email` (Email Agent). In this assignment they come from **MCP servers** (file `02`).

---

## 5. Multi-agent orchestration

**One agent** is good at one job. **Multi-agent** means several specialized agents, each expert at one thing, working together — usually in a **pipeline** or with an **orchestrator**.

Your assignment is a **pipeline (sequential) multi-agent system**:
```
Research Agent  ───►  Email Agent
 (specialist 1)        (specialist 2)
```

**Why split into two agents instead of one big agent?**
- **Separation of concerns** — each agent has ONE clear job and ONE toolset. Easier to build, test, and reason about.
- **Clarity of prompts** — the research prompt is about researching; the email prompt is about emailing. No mixing.
- **Reusability** — you could swap the Email Agent for a "Save to Notion" agent without touching research.
- **This is exactly how real production agent systems are designed.**

(This is the same idea as your DHL project's 5 specialist agents — you already know this pattern!)

---

## 6. Agent-to-agent communication (the part they grade)

"Agent-to-agent communication" sounds fancy but it means: **the output of one agent becomes the input of the next.**

```
Research Agent produces:  "SUMMARY: ... KEY FINDINGS: ... REFERENCES: ..."
                                   │
                                   ▼  (this text is PASSED to)
Email Agent receives:     "Here is a report. Send it to me@gmail.com: <the report>"
```

In Langflow you'll literally draw a **wire** from the Research Agent's output port to the Email Agent's input. That wire IS the agent-to-agent communication. **You must be able to point at it and say "this is the handoff."**

---

## 7. Orchestration — who runs the show?

**Orchestration** = coordinating the agents: deciding the order, passing data between them, handling the overall flow.

In your assignment, **Langflow is the orchestrator.** The canvas defines: input → Research Agent → Email Agent → output. Langflow runs them in order and moves the data along the wires.

(In code, a framework like LangGraph or CrewAI would orchestrate. Here, Langflow's visual flow does it.)

---

## 8. Key terms cheat-sheet (say these correctly)
| Term | Meaning |
|------|---------|
| LLM | The base text model (GPT-4o) |
| Agent | LLM + tools + the reason/act loop |
| Tool | A function the agent can call |
| Agentic | The AI decides the steps itself |
| ReAct | Reason + Act loop pattern |
| Multi-agent | Several specialized agents cooperating |
| Pipeline | Agents in sequence, output→input |
| Orchestration | Coordinating the agents (Langflow does it here) |
| Agent-to-agent | One agent's output feeds the next |

---

## The sentence that proves you understand agents
> "An agent is an LLM wrapped in a reason-act loop with access to tools — it decides which tool to call, observes the result, and repeats until the goal is met. My system orchestrates two such agents in a pipeline: a Research Agent that uses a search tool, handing its report to an Email Agent that uses an email tool. The wire between them is the agent-to-agent communication, and Langflow is the orchestrator."

---

Next → `02_MCP_DEEP_DIVE.md` (the most important file)
