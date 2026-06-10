# Simple Things to Know — Plain English Essentials

No jargon. Every concept explained simply so it sticks before the interview.

---

## What This Job Actually Is

You are the person who takes an AI that lives in a lab and makes it work in real life.

Specifically: a business team (say, the HR department) wants to stop manually processing 500 invoices every month. You take that problem, build an AI system on the Happy Robots platform that reads those invoices, pulls out the important numbers, checks if they're correct, and sends them to the right place — automatically. Then you make sure it keeps working perfectly every single day, without anyone having to babysit it.

You're not a researcher. You're not building AI from scratch. You're the person who makes AI automation actually reliable and real.

---

## What Is an "AI Worker"?

Think of it like hiring a very focused digital employee who only does one job but does it 24 hours a day, 7 days a week, without sick days.

Examples of AI Workers:
- **Scheduling Worker:** Handles all appointment booking calls so human agents don't have to
- **Invoice Worker:** Reads invoices, extracts the data, sends it to SAP automatically
- **Tracking Worker:** Answers "where is my shipment?" questions across email, WhatsApp, chat

Your job is to build, train, configure, and maintain these AI Workers.

---

## What Is "Low-Code / No-Code"?

**Simple version:** You're building with ready-made LEGO pieces, not carving wood from scratch.

Low-code means you configure the platform — you connect blocks, set rules, write prompts, define workflows. You write minimal custom code only for things the platform can't do out of the box.

It's powerful because:
- Faster to build (days, not months)
- Easier to change and fix
- Non-developers can understand and help maintain it

---

## What Is Prompt Engineering?

**Simple version:** Writing clear, specific instructions for an AI so it gives you exactly the output you need, every time.

Bad prompt (for invoice processing): *"Read this invoice and give me the details."*
Good prompt: *"You are an invoice data extraction AI. Extract these exact fields from the invoice text below and return them as JSON: vendor name, invoice number, invoice date (YYYY-MM-DD format), total amount (number only, no currency symbol). If a field is not present in the document, return null. Do not guess or estimate any values."*

The better your instructions, the more reliable the AI. This is a real skill that takes practice and testing.

---

## What Is Policy Engineering?

**Simple version:** Writing the rules for what the AI is allowed and not allowed to do.

An appointment scheduling AI's policy might say:
- **Allowed:** Book, reschedule, cancel appointments
- **Not allowed:** Give medical advice, access patient medical records, discuss billing
- **Always escalate:** If someone mentions a medical emergency

Policy engineering is the governance layer that keeps the AI in its lane.

---

## What Are Guardrails?

**Simple version:** The safety net that catches the AI if it tries to do something it shouldn't — even if its instructions said not to.

LLMs sometimes make mistakes and produce outputs that violate their own instructions. Guardrails are technical filters that check every AI response BEFORE it reaches the user.

Like a content filter: if the AI accidentally starts giving medical advice, the guardrail blocks that response and replaces it with "I can only help with appointment scheduling."

---

## What Is Voice AI?

**Simple version:** Instead of typing to an AI, you talk to it. And it talks back.

The process:
1. You speak → microphone
2. **STT (Speech-to-Text):** converts your speech into text the AI can process
3. AI understands what you said, decides what to do
4. AI generates a text response
5. **TTS (Text-to-Speech):** converts the text response into natural-sounding speech
6. You hear the response

**Why it's harder than chat:**
- Responses must be short — you can't scroll up when you're on a call
- The words must sound natural when spoken aloud (no lists, no special characters)
- Silence is uncomfortable on a call — the AI must acknowledge immediately ("One moment...")
- The voice tone must match the context — a payment failure message should sound serious, not cheerful

---

## What Is a Model Card?

**Simple version:** A one-page specification sheet for an AI solution. Like a product data sheet but for AI.

It tells anyone who reads it:
- What this AI does and what it doesn't do
- How accurate it is (and on what types of inputs)
- Where it can make mistakes
- What data it uses and how privacy is protected
- Who reviewed and approved it
- How to turn it off if something goes wrong

Why it matters: Without a model card, when something goes wrong, nobody knows what the AI was supposed to do, who approved it, or what the failure rate was supposed to be. A model card makes everything auditable.

---

## What Is Data Annotation?

**Simple version:** Labeling examples so the AI (or its evaluation system) knows what "correct" looks like.

Imagine you have 500 customer emails. Data annotation means reading each one and writing: "This email = appointment request" or "This email = complaint" or "This email = question about opening hours."

Why it matters: AI evaluation requires knowing what the correct answer is. Without annotated examples, you can't measure if the AI is accurate.

**Synthetic data:** When you don't have enough real examples, you use AI to generate fake-but-realistic ones. Like writing 50 made-up customer emails that sound real, so you have enough examples to test with.

---

## What Is Data Lineage?

**Simple version:** Knowing where every piece of data came from and what happened to it.

Imagine an invoice gets processed wrong. Data lineage lets you trace: where did this invoice come from? What did the OCR extract? What did the AI pull from it? What validation ran? What decision was made? Who approved it?

Without data lineage, you're just guessing when something goes wrong.

---

## What Is MLOps (in this context)?

**Simple version:** Treating AI configurations like software code — versioned, tested before deployment, with rollback if something breaks.

Traditional software: you write code → test it → deploy it → monitor it → fix bugs.

GenAI automation MLOps: you write prompts and configurations → test them against labeled examples → deploy to production → monitor accuracy and cost → improve when quality drops.

**Key insight:** AI automations are NOT set-it-and-forget-it. The real world is messier than test data. You must continuously monitor and improve.

---

## What Is BUIT?

**BUIT = Business Unit IT**

Every large company has different business divisions (HR, Finance, Supply Chain, Customer Service). Each of those has its own IT team — the BUIT.

The BUIT tells you what needs to be automated and owns the systems the automation must connect to. You take their requirements and build the solution on Happy Robots.

Your job: understand what they need (discovery), design a solution, configure it, test it, deploy it, and keep it running. When they ask for something impossible or risky, you push back with data and propose a better approach.

---

## What Is "Responsible AI"?

**Simple version:** Making sure your AI automation doesn't cause harm.

4 things to check:
1. **Privacy:** Is the AI seeing personal information it shouldn't? (Names, bank details, medical info)
2. **Fairness:** Does the AI work equally well for everyone? (Does it fail more for non-English speakers?)
3. **Explainability:** If the AI makes a decision, can you explain WHY? (Critical for finance and healthcare)
4. **Compliance:** Does the automation follow relevant laws? (GDPR, India's DPDP Act, healthcare regulations)

In practice: you test, document, add guardrails, keep humans in the loop for high-stakes decisions, and write model cards that prove you checked all of this.

---

## 3 Things That Will Impress

**1. Say "in production" not "I've built"**
Wrong: "I've built invoice processing with LLMs."
Right: "I've deployed invoice processing that handles 800 invoices/month at 97.2% accuracy. I track this weekly and run prompt improvements monthly."

**2. Show that you think about failure**
Wrong: "The AI extracts the invoice data and sends it to SAP."
Right: "The AI extracts with Pydantic validation on every output. Anything failing validation goes to a human review queue — so the 3% edge cases never silently corrupt SAP data."

**3. Tie everything to business outcomes**
Wrong: "I optimized the prompt."
Right: "After prompt optimization, AP team manual processing time dropped from 40 hours/week to 6 hours/week, with the AI handling 85% of invoices automatically."

---

## Words to Use and Avoid

| Use | Instead of |
|-----|-----------|
| "In production..." | "In a demo / project / POC..." |
| "I track accuracy by..." | "It seems to work well..." |
| "The business outcome was..." | "The technical implementation was..." |
| "I configured guardrails to prevent..." | "I told the AI not to..." |
| "I own the reliability of..." | "I was involved in building..." |
| "My rollback procedure is..." | "If something breaks, we'd fix it..." |
| "The model card documents..." | "We have some documentation..." |
