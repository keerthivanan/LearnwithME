# DHL AI Command Center — Import & Run Guide

This is ONE workflow that works like an entire DHL service + operations floor.

---

## 🏗️ The Architecture (what you built)

```
                    💬 Customer Message In
                            │
                            ▼
                  🧭 TRIAGE ROUTER (AI Classifier)
              reads the message, picks the right specialist
                            │
        ┌──────────┬────────┼────────┬───────────┐
        ▼          ▼        ▼        ▼           ▼
   📦 TRACKING  📝 BOOKING ⚠️ CLAIMS 🛃 CUSTOMS  🚛 DELIVERY
     Agent       Agent     Agent    Agent       Agent
        │          │        │        │           │
   ┌────┴───┐ ┌────┴────┐ ┌─┴──┐ ┌───┴───┐  ┌────┴────┐
  track   guide  rec  quote book claim comp  reqs duties resched change
   (2 tools)   (3 tools)  (2 tools)  (2 tools)  (2 tools)
        └──────────┴────────┼────────┴───────────┘
                            ▼
                  📊 Response + Audit Log
```

**1 router + 5 AI agents + 11 tools + shared memory + shared GPT brain = ONE workflow.**

---

## ⚡ Setup (2 minutes)

### Step 1 — Import
1. Open n8n → `http://localhost:5678`
2. New Workflow → top-right menu (⋮) → **Import from File**
3. Select `DHL_AI_Command_Center.json`

### Step 2 — Add OpenAI key (only ONCE for the whole system)
1. Click the **🧠 GPT Brain (shared)** node
2. Credentials → **Create New** → paste your OpenAI API key → Save
3. Done — that one brain powers the router and all 5 agents

> Get a key: platform.openai.com → API Keys → Create new key

### Step 3 — Talk to it
1. Click **Chat** at the bottom of the canvas
2. Start typing customer messages

---

## 🧪 Demo Script (run these in order — it's impressive)

| # | Type this | What happens |
|---|-----------|--------------|
| 1 | `Where is my package 1234567890?` | Router → Tracking Agent → calls track_shipment → "out for delivery, Chennai, by 8 PM" |
| 2 | `When exactly will it arrive?` | Shared memory remembers the shipment — no need to repeat the number |
| 3 | `I need to ship 3 boxes, 18kg, Pune to Frankfurt by Friday` | Router → Booking Agent → recommends service + can create booking |
| 4 | `My package arrived broken!` | Router → Claims Agent → empathy + files claim + escalates to human |
| 5 | `Shipment 1111222233 is stuck in customs, what do I do?` | Router → Customs Agent → lists exact documents needed |
| 6 | `I missed my delivery, can I pick it up from the hub?` | Router → Delivery Agent → arranges hub pickup |
| 7 | `How does DHL compare to FedEx?` | Any agent → politely refuses competitor talk (guardrail) |

**The magic to point out in the interview:** you never tell it which agent to use — the **Triage Router decides automatically** from the customer's words. That's real multi-agent orchestration.

---

## 🎯 What Each Agent Replaces (the business case)

| Agent | Human role it replaces | DHL daily pain |
|-------|----------------------|----------------|
| 📦 Tracking & Status | L1 support reps | 60-70% of all queries are "where is my package" |
| 📝 Booking & Quotes | Booking desk coordinators | 8-15 min of manual data entry per booking |
| ⚠️ Claims & Complaints | Claims handlers | Sensitive, slow, needs careful handling |
| 🛃 Customs Support | Customs ops team | Hours chasing missing documents daily |
| 🚛 Delivery Management | Dispatch reschedulers | Every failed re-attempt costs $8-15 |

**One workflow, running 24/7, handling the volume of dozens of staff.**

---

## 🔌 Going to Production (what you'd say in the interview)

This demo uses mock data so it runs anywhere. To make it production-real:

1. **Real tracking** → swap the `track_shipment` tool's mock block for an HTTP call to `https://api-eu.dhl.com/track/shipments` (free key at developer.dhl.com)
2. **Real channels** → add a Twilio/WhatsApp node and an Email Trigger so customers reach it on voice, SMS, WhatsApp, email — not just chat
3. **Real systems** → the `create_booking` and `file_claim` tools would call DHL's actual booking/CRM APIs
4. **Real audit** → the Response + Audit node would write to a database (Postgres) instead of passing through
5. **Guardrails** → add a moderation check before responses go out

The architecture doesn't change — only the tool internals. That's the whole point of building it tool-first.

---

## 💡 Why This Maps Perfectly to the Happy Robots JD

| JD requirement | Where it's demonstrated |
|----------------|------------------------|
| "low-code/no-code GenAI conversational solutions" | The whole workflow is built in n8n's visual builder |
| "workflows, business rules, orchestration logic" | Triage Router + 5-agent orchestration |
| "prompt engineering, policy engineering, guardrail config" | Each agent's system prompt = role + policy + guardrails |
| "multi-channel across voice, email, SMS, chat" | Chat now; channel nodes plug into the same workflow |
| "enable use cases: scheduling, tracking, document processing" | Tracking, booking, customs docs all covered |
| "API integrations with internal/third-party systems" | Tools are the integration points (DHL API, CRM, etc.) |
| "Responsible AI, escalation, audit" | Claims escalation + Response/Audit node |
