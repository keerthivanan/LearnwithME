# TVS SCS — Senior GenAI & Agentic AI Engineer — Analysis & Strategy

---

## What this role REALLY is (read this first)

This is an **engineering-first, production-focused** senior role. The single most important line in the whole JD:

> "TVSSCS is hiring a Senior GenAI & Agentic AI Engineer who has **shipped production AI systems to real users — not just built prototypes.**"

And the deal-breaker:
> "**Applications without documented production AI project experience will not be shortlisted.**"

**Translation:** they don't care that you *know* LangChain. They care that you've **deployed** something, **monitored** it, had it **break**, and **fixed** it. Everything you say must sound like it came from production, not a tutorial.

---

## The bar (be honest with yourself)
- **3-7 years** ML/AI/LLM engineering
- **2 production-deployed AI systems** you can describe in detail (problem, scale, your ownership, tech stack)
- **1 production incident** you debugged
- Real hands-on with: **agentic framework** (LangChain/LlamaIndex/AutoGen/CrewAI), **RAG + vector DB**, **FastAPI**, **Docker/K8s**, an **LLM API**, **MLOps tooling**

If you're early-career, this is a **stretch role**. That's okay — but you win it by (1) presenting your REAL built projects credibly, and (2) speaking about production concerns (latency, cost, failure, monitoring) fluently. **Never fabricate years or fake projects — it collapses under follow-up questions.** Present what you genuinely built, strongly.

---

## Your unfair advantage: you HAVE built real systems
In our work you actually built and deployed:
1. **DHL AI Command Center** — a live multi-agent system on n8n (triage router + 5 specialist agents + 12 tools), omnichannel (voice/SMS/email/chat), a Vapi voice bridge, real data, deployed on a live server, with monitoring via execution logs.
2. **DHL Voice AI** — Vapi conversational agent calling your n8n brain as a custom-LLM (real-time voice).
3. **Langflow multi-agent MCP system** — Research + Email agents with MCP-served tools.

These are **genuine production-shaped agentic AI systems.** File `01` turns them into the exact "production system" narratives this JD demands. **This is your strongest asset — lead with it.**

---

## The 5 technical pillars they'll test (your study map)
| Pillar | File | Weight |
|--------|------|--------|
| **Agentic AI** (LangChain, LangGraph, ReAct, multi-agent, memory, HITL, tools) | 02 | ⭐⭐⭐⭐⭐ |
| **RAG** (vector DBs, hybrid search, re-rank, chunking, embeddings, routing) | 03 | ⭐⭐⭐⭐⭐ |
| **MLOps & Deployment** (Docker/K8s, CI/CD, vLLM/Triton, MLflow, fine-tuning) | 04 | ⭐⭐⭐⭐ |
| **Observability** (LangSmith/LangFuse, metrics, tracing, alerting) | 05 | ⭐⭐⭐⭐ |
| **Backend** (FastAPI, async, auth, rate limiting, DBs, Celery) | 06 | ⭐⭐⭐ |
| Plus **LLM fundamentals** (prompting, context, cost) | 07 | ⭐⭐⭐ |

You also have deep material already in the `practical/` folder (transformers, fine-tuning, RAG, deployment, cloud) — reuse it.

---

## How to talk (the production voice)
Weak (tutorial): "I used LangChain to build an agent."
Strong (production): "I built a multi-agent system where a router classifies the request and dispatches to specialist agents, each with scoped tools. In production I watched p95 latency and token cost per conversation, added a fallback when the primary model failed, and caught a bug where the tool wasn't receiving its argument — the agent was passing it under a different key — which I diagnosed from the execution traces and fixed."

**Always mention: latency, cost, failure modes, monitoring, and what broke.** That's the senior signal.

---

## The application asks for specifics — prepare these NOW (file 01)
1. **Production system #1** — problem, scale, your role, tech stack
2. **Production system #2** — same
3. **One incident you debugged** — what broke, how you diagnosed, what you fixed
4. **Links** — your GitHub (`github.com/keerthivanan/LearnwithME`), the DHL/Langflow projects

---

## 30-second intro (adapt honestly)
> "I'm a GenAI engineer focused on shipping agentic systems, not demos. I've built and deployed a multi-agent customer-service system — a triage router dispatching to five specialist agents with scoped tools, running omnichannel across voice, SMS, and chat, deployed live with execution-trace monitoring. I work in Python with FastAPI-style webhooks, LLM function-calling with structured output, and I care about the production concerns: latency, token cost, failure fallbacks, and tracing. I'm looking to go deeper on production LLM infrastructure at TVS SCS's scale."

---

## Honesty guardrail
Where you lack depth (e.g., you haven't run vLLM or fine-tuned with QLoRA at scale), say:
> "I understand the concept and how it fits — [explain] — but I haven't run it at production scale yet. I'd ramp on that quickly."
Studying files 02-07 lets you explain every concept credibly even where hands-on is light. **Understanding + honesty > pretending.**

---

Next → `01_your_production_projects.md` (your ammunition)
