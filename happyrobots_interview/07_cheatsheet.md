# Master Cheat Sheet — Read the Night Before

---

## Happy Robots in 3 Sentences (Memorize This)
Happy Robots is a GenAI automation platform that enables enterprises to deploy AI Workers — intelligent agents that handle real business processes like appointment scheduling, shipment tracking, document ingestion, and vendor coordination across voice, chat, email, and SMS channels. The platform is low-code/no-code, meaning solutions are built by configuring prompts, policies, workflows, and integrations — not building applications from scratch. This Manager role is the execution layer: taking BUIT-defined requirements and turning them into production-grade, compliant, cost-optimized AI automations that reliably run at scale.

---

## The 7 Pillars — One Line Each

| Pillar | Your Job |
|--------|---------|
| **1. Configuration & Deployment** | Design, configure, and ship AI Workers on the platform |
| **2. Data & Annotation** | Build evaluation datasets, annotate, generate synthetic data, maintain lineage |
| **3. Testing & Responsible AI** | Test rigorously, write model cards, audit for bias, ensure compliance |
| **4. Production Support** | Monitor accuracy/latency/cost, diagnose failures, iterate continuously |
| **5. MLOps & Platform Ops** | Version prompts, manage lifecycle, maintain rollback readiness |
| **6. Integration** | Connect AI Workers to enterprise APIs, manage auth, data flows |
| **7. Use-Case Enablement** | Build scheduling, shipment, document, voice AI Workers end-to-end |

---

## Prompt Engineering vs Policy Engineering vs Guardrails

| | What it is | What it controls |
|--|-----------|-----------------|
| **Prompt Engineering** | Instructions to the LLM | HOW it does the task (accuracy, format, quality) |
| **Policy Engineering** | Rules for AI behavior | WHAT it's allowed/not allowed to do |
| **Guardrail Config** | Technical enforcement layer | Catching policy violations AFTER LLM output, BEFORE user sees it |

---

## Voice AI Stack (Know This Cold)

```
User speaks → [STT] → text → [NLU/LLM] → intent + entities → 
[Workflow] → action → [LLM] → response text → [TTS] → spoken audio → User hears
```

**Voice prompt rules:**
- Max 2 sentences per response
- No markdown, no lists, no symbols
- Natural spoken phrasing
- Always use SSML for date/time, emphasis, pauses

**Key voice metrics:** WER (STT accuracy), Intent Accuracy, MOS (TTS naturalness), Containment Rate

---

## Responsible AI — 4-Layer Framework (Memorize)

```
1. DATA PRIVACY     → PII detection → anonymize/redact before LLM call → self-host for sensitive data
2. OUTPUT VALIDATION → Pydantic validation → failed outputs → human review queue → never silent fail
3. AUDIT TRAILS     → trace_id on every item → log input hash, model version, output hash, action, timestamp
4. BIAS TESTING     → test across: language variants, input formats, apparent demographics → document in model card
```

---

## Model Card — 8 Sections

1. Solution overview (what it does, platform, version, owner)
2. Intended use (what it does + what it explicitly does NOT do)
3. Performance metrics (accuracy by category, latency p50/p95/p99, test dataset size)
4. Known limitations and failure modes
5. Responsible AI evaluation results (bias, adversarial, privacy)
6. Data governance (what data, where it goes, PII handling, retention)
7. Operational parameters (cost/call, rate limits, rollback procedure)
8. Approval and sign-off (who approved, when, next review date)

---

## Production Readiness Checklist

Before any AI Worker goes live:
- [ ] Evaluation dataset built (100+ examples, labeled, versioned)
- [ ] Accuracy baseline established and meets threshold (95%+ standard cases)
- [ ] Pydantic output validation in place for every LLM response
- [ ] Human review queue configured for failed validations
- [ ] Guardrails tested (adversarial test attempted and blocked)
- [ ] Audit log table live and logging correctly
- [ ] Monitoring dashboard live (error rate, latency, cost)
- [ ] Rollback procedure documented and tested
- [ ] Model card written and BUIT sign-off obtained
- [ ] Prompt versions committed to Git

---

## Data Annotation Quality Control

- **Inter-Annotator Agreement:** Kappa > 0.8 = good, < 0.6 = fix the schema first
- **Sampling audit:** Review 10% of full annotated set before final approval
- **Synthetic data rule:** Always human-review 20-30% of synthetic examples before using in eval
- **Lineage requirement:** Every dataset has metadata: source, generator, date, review status, schema version

---

## MLOps for GenAI (Prompt-as-Code)

```
STAGING: Edit prompt → Run eval → Score vs baseline
                                    ↓ Pass
PEER REVIEW: PR with eval results attached
                                    ↓ Approved
CANARY: 10% traffic → Monitor 24h
                                    ↓ No degradation
PRODUCTION: 100% traffic → Update registry
                                    ↓
MONITOR: Accuracy spot-check weekly, cost daily, latency real-time
```

**Rollback target:** Under 15 minutes. Trigger: accuracy drops >5%, error rate >3%, latency p95 >SLA.

---

## Cost Optimization — 5 Levers

1. **Model routing** — GPT-4o-mini for simple tasks (16x cheaper than GPT-4o)
2. **Prompt compression** — remove verbosity, use structured input format
3. **Caching** — cache results for repeated identical inputs
4. **Batching** — group similar requests into one API call where possible
5. **Context trimming** — for conversation history, include only last N turns needed

**Key numbers:**
- GPT-4o-mini: ~$0.15/M input tokens, ~$0.60/M output tokens
- GPT-4o: ~$2.50/M input tokens, ~$10/M output tokens
- 1 token ≈ 4 characters ≈ 0.75 words
- 10-page PDF ≈ 7,500–10,000 tokens

---

## Document Processing Key Steps

INTAKE → CLASSIFICATION → PRE-PROCESSING (OCR if scanned) → LLM EXTRACTION → PYDANTIC VALIDATION → BUSINESS RULES → ROUTING → SYSTEM HANDOFF → AUDIT LOG

**India-specific:** Handle GST numbers (15-char), CGST/SGST/IGST, Indian number format (12,50,000), IFSC codes.

---

## The 3 Things That Win in This Interview

**1. Production mindset, not demo mindset**
Don't say: "I've built LLM applications."
Say: "I've deployed AI Workers processing [X volume/day] with [Y]% accuracy. I track this weekly and iterate when I see drops."

**2. Show you think about failure, not just success**
Don't say: "I configure the prompt so the AI extracts the invoice data."
Say: "I configure extraction with Pydantic validation on every output, guardrails for low-confidence responses, and a human review queue for anything that fails — so nothing bad reaches SAP."

**3. Business outcomes, not technical features**
Don't say: "I used SSML to configure the TTS voice."
Say: "After configuring SSML for our voice scheduling worker, patient satisfaction with the voice experience went from 3.1 to 4.3 MOS, and our containment rate increased from 58% to 74%."

---

## Interview Day Checklist

**Night before:**
- [ ] Read this cheat sheet fully (45 minutes)
- [ ] Review your 3 best STAR stories with numbers
- [ ] Check LinkedIn profiles of your interviewers
- [ ] Select 4 questions from 06_questions_to_ask.md
- [ ] Sleep by 11pm

**Morning of:**
- [ ] Re-read Sections 1-3 of this cheat sheet (20 minutes)
- [ ] Confirm interview link / venue / time zone
- [ ] Have notepad ready

**In the interview:**
- [ ] Lead every technical answer with "In production, my approach is..."
- [ ] Include specific numbers in every STAR answer
- [ ] Ask at least 3 questions
- [ ] Close with: "Is there anything about my background that gives you pause?"

---

## If You Don't Know Something

Never bluff. Say exactly this:

"I haven't worked with [specific tool/technology] directly, but based on my experience with [related area], my approach would be [logical transfer]. I'd ramp on [specific thing] specifically — what does the team currently use for this?"

This shows: honesty + transferable reasoning + curiosity. Always better than a bluffed answer that unravels on follow-up.
