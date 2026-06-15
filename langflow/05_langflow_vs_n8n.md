# Langflow vs n8n — Learn Faster Using What You Know

You built a whole DHL system in n8n. Langflow will click fast because the ideas map across.

---

## The big similarity
**Both are visual, node-based builders.** You drag boxes onto a canvas and connect them with wires. Same mental model.

---

## The big difference
| | **n8n** | **Langflow** |
|--|---------|-------------|
| Built for | **General automation** (any app, any API, schedules, webhooks) | **AI/LLM apps specifically** (chatbots, RAG, agents) |
| Strength | Connecting 500+ apps, triggers, business workflows | Deep LLM tooling — RAG, embeddings, agents, prompts |
| Language under the hood | TypeScript/Node | **Python / LangChain** |
| Best for | "When email arrives, process it and update CRM" | "Chat with my PDFs" / "an agent that uses tools" |
| Triggers | Rich (webhook, cron, app events) | Mostly chat/API driven |

**Simple way to say it:** n8n automates *business processes*; Langflow builds *AI brains*. They're complementary.

---

## Concept mapping (your n8n knowledge → Langflow)

| n8n concept | Langflow equivalent |
|-------------|---------------------|
| Node | Component |
| Connection/wire | Edge |
| Workflow | Flow |
| Chat Trigger | Chat Input |
| AI Agent node | Agent component |
| Tool (toolCode) | Tool component / Custom Component |
| OpenAI Chat Model node | OpenAI / Language Model component |
| Window Buffer Memory | Chat Memory component |
| Webhook | Run flow via API / Chat Input |
| Execute / Playground | Playground |
| Export workflow JSON | Export flow JSON |

**See? You already understand Langflow** — it's the same ideas with AI-focused components.

---

## Your DHL project, rebuilt in Langflow (conceptually)
```
n8n version:
[Webhook] → [Normalize] → [Triage Router] → [5 Agents + 12 tools] → [Response]

Langflow version:
[Chat Input] → [Agent with a system prompt] 
                  ↑
            [Tools: track_shipment, file_claim, reschedule... as Custom Components]
            → [Chat Output]
```
In Langflow you'd likely use **one strong Agent with many tools** (or an orchestrator agent calling sub-flows), rather than n8n's classifier-then-route pattern. Both are valid.

---

## When to use which (interview-ready answer)
> "I'd use **n8n** when the work is broad business automation — triggers, connecting many SaaS apps, scheduled jobs. I'd use **Langflow** when the core is the AI itself — building a RAG pipeline or an agent with tools, where I want fine control over prompts, embeddings, and vector stores. They're not competitors; I've used n8n as the automation/orchestration layer and Langflow-style flows as the AI brain."

That answer shows you understand BOTH tools and when each fits — strong signal.

---

## Which should YOU focus on?
- You already have a **real, working n8n project** (DHL) — that's your headline.
- Learn Langflow to **broaden** your GenAI toolbox and speak to RAG/agents fluently.
- Together they make you versatile: "I can build the automation AND the AI brain."

---

Next → **06_interview_qa.md**
