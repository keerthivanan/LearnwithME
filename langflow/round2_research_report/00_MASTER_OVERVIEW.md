

# 00 — MASTER OVERVIEW: Know The Whole Thing

Read this first. It gives you the complete picture so everything else makes sense.

---

## What you're building (in one breath)
A system in Langflow where you type a **topic** and an **email address**, and:
1. A **Research Agent** searches the web and writes a structured report
2. It hands that report to an **Email Agent**
3. The Email Agent emails the full report to the address you gave

And the catch that makes it a real test: **the search tool and the email tool must each come from an MCP server** — not built-in Langflow tools.

---

## The 4 big ideas you must KNOW (this whole course explains each deeply)

### Idea 1 — Agents
An **agent** is an LLM that can *decide to take actions* using tools, not just chat. The Research Agent decides to search; the Email Agent decides to send mail. → File `01`

### Idea 2 — Multi-agent orchestration
Two agents working as a pipeline, where **one agent's output becomes the next agent's input**. Research Agent → Email Agent. That handoff is "agent-to-agent communication." → File `01`

### Idea 3 — MCP (Model Context Protocol)
A standard way to give agents tools. Instead of building a tool *inside* Langflow, you run a separate **MCP server** that exposes the tool, and the agent calls it over a standard protocol. This is THE core concept they're testing. → File `02`

### Idea 4 — Langflow as the orchestrator
Langflow is the visual canvas where you place the agents, wire them together, and connect them to the MCP servers via the **MCP Tools** component. → Files `03`, `04`

---

## The full picture (memorize this diagram)

```
USER TYPES:  topic = "Quantum computing in 2025"   email = "me@gmail.com"
                       │                                  │
                       ▼                                  │
            ┌──────────────────────┐                      │
            │   RESEARCH AGENT     │                      │
            │  (an LLM that can    │   calls tool over    │
            │   use tools)         │──────MCP────────► [SEARCH MCP SERVER]
            │                      │ ◄────results──────  (web_search tool)
            │  writes a report:    │
            │  Summary / Findings  │
            │  / References        │
            └──────────┬───────────┘
                       │  report text  (AGENT → AGENT handoff)
                       ▼
            ┌──────────────────────┐
            │    EMAIL AGENT       │   calls tool over
            │  (an LLM that can    │──────MCP────────► [EMAIL MCP SERVER]
            │   use tools)         │ ◄────"sent!"──────  (send_email tool)
            │  composes + sends    │
            └──────────┬───────────┘
                       ▼
              "✅ Report emailed to me@gmail.com"
```

---

## The 3 questions the interviewer WILL ask (and you must nail)
1. **What is an MCP server and how does it expose tools to agents?**
2. **How does Langflow's MCP Tools component connect to an MCP server?**
3. **What's the difference between native tools and MCP-served tools?**

→ All answered deeply in file `02` and `06`. By the end of this course you'll answer these in your sleep.

---

## The hard requirements (you lose points if you miss these)
- ✅ Both agents in ONE Langflow flow
- ✅ Search tool comes from an MCP server (NOT Langflow's native search)
- ✅ Email tool comes from an MCP server (NOT Langflow's native email)
- ✅ You can SEE the Research output going INTO the Email agent (agent-to-agent)
- ✅ The email body has the FULL report (not just a one-line summary)
- ✅ One user input triggers everything (topic + email together)

---

## Your course map
| File | What you'll KNOW after it |
|------|---------------------------|
| `00_MASTER_OVERVIEW` | The whole system at a glance (this file) |
| `01_AGENTS_EXPLAINED` | Agents, agentic AI, multi-agent, agent-to-agent — deeply |
| `02_MCP_DEEP_DIVE` | MCP from first principles — the make-or-break topic |
| `03_THE_TWO_MCP_SERVERS` | The exact tools, the code, how to run them |
| `04_BUILD_IN_LANGFLOW` | Every node, every wire, every click — with WHY |
| `05_HOW_IT_ALL_WORKS` | A full run traced moment-by-moment |
| `06_INTERVIEW_QA` | Every possible question + a strong answer |
| `07_BONUS_FEATURES` | Memory, error handling, remote MCP via ngrok |

Read them in order. Don't skip `02` — that's what they're really testing.

---

Next → `01_AGENTS_EXPLAINED.md`
