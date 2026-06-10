# Interview Q&A — Manager, GenAI Automation Support

Organized by the 7 responsibility pillars from the JD.

---

## PILLAR 1: GenAI Product Configuration & Deployment

---

**Q: How do you approach configuring a new AI Worker from scratch on a low-code platform?**

**A:**
"I follow a 5-phase configuration approach:

**Phase 1 — Discovery and scope definition**
Before opening the platform, I sit with the BUIT team to map the business process: what triggers this workflow, what data flows through it, what actions does it take, what are the failure modes. I produce a one-page Solution Design with agreed accuracy targets, agreed scope, and agreed human-in-the-loop points. No configuration until BUIT signs off.

**Phase 2 — Data and prompt strategy**
I identify what knowledge the AI Worker needs, where it lives, and how to inject it efficiently. I design the system prompt structure: role definition → policy section → tone/channel instructions → output format. I build the evaluation dataset (minimum 100 examples) before writing the production prompt — this forces me to define 'correct' concretely.

**Phase 3 — Platform configuration**
Build the workflow in the platform: trigger → LLM call node → validation node → routing node → integration node → logging node. Configure guardrails: content filters, format validators, confidence thresholds, escalation rules. Configure channel: is this voice, chat, email, SMS? Each needs channel-specific prompt tuning.

**Phase 4 — Testing**
Run evaluation dataset through the configured worker — baseline accuracy. Functional testing: does each workflow path work? Integration testing: does the CRM/ERP call return correctly formatted data? Adversarial testing: can a user bypass guardrails? Performance testing: latency under load.

**Phase 5 — Phased deployment and monitoring**
Deploy to 10% of traffic. Monitor for 48 hours. No regression → expand to 100%. Set up ongoing monitoring: accuracy sampling, error rate alerts, cost tracking dashboard. Document in model card and version registry."

---

**Q: What is policy engineering and how is it different from prompt engineering?**

**A:**
"They work together but at different levels.

Prompt engineering is about HOW the AI does its task — the instructions, examples, and output format that make it accurate and consistent. It's the technical craft of making the model perform well.

Policy engineering is about WHAT the AI is permitted to do — the rules that define its decision-making space, regardless of how the task is performed. It's governance.

A policy document answers:
- What actions can this AI Worker take autonomously?
- What requires a human approval step?
- What must it never do under any circumstances?
- When must it escalate to a human?
- How does it handle out-of-scope requests?

In practice, I embed policy constraints in the system prompt AND in a separate guardrail layer. The system prompt communicates the policy to the model in natural language. The guardrail layer enforces it technically — so even if the LLM fails to follow the policy, the guardrail catches violations before output reaches the user.

Example: policy says 'never provide medical advice'. System prompt says 'do not provide medical advice'. Guardrail uses a topic classifier to block any response classified as medical_advice. Two layers of enforcement."

---

**Q: How do you configure a multi-channel AI experience differently for voice versus chat?**

**A:**
"Voice and chat require fundamentally different prompt and TTS configuration even when the underlying workflow is the same.

**For chat:**
- Responses can be longer (users can scroll)
- Can use formatting: bullet points, bold, line breaks
- Users can re-read, no time pressure
- Latency tolerance: 2-4 seconds is acceptable
- Retry: user can ask to repeat or rephrase

**For voice:**
- Responses must be short — maximum 2 sentences per turn (cognitive load of listening)
- No markdown — it will be read literally by TTS ('asterisk asterisk important asterisk asterisk')
- Natural spoken phrasing: 'Got it! I'm looking that up for you now' not 'Processing your request'
- Latency critical — users hang up if silent for >3 seconds; use acknowledgment phrases ('One moment...')
- Numbers: 'March twentieth' not '3/20' — configure date/number interpretation in TTS SSML
- TTS configuration: select voice model for tone (warm, professional), set speech rate, add pauses at natural sentence breaks

I also handle error states differently. In chat, I can show a formatted error message. In voice, I say 'I didn't quite get that — could you repeat that?' and add a brief pause before the question so the user has processing time.

The system prompt for a voice worker explicitly states: 'You are operating on a voice channel. Keep all responses to 1-2 short sentences. Never use lists, markdown, or symbols. Use natural spoken English only.'"

---

## PILLAR 2: Data Preparation, Annotation & Model Enablement

---

**Q: Walk me through how you'd build an evaluation dataset for a new AI Worker.**

**A:**
"I treat evaluation dataset creation as one of the most important investments in an AI deployment — because without it, you're flying blind.

**Step 1 — Define the coverage matrix**
What are all the intent categories? All the edge cases? All the channel/language variants? I map these on a spreadsheet — this becomes my coverage checklist.

**Step 2 — Source real examples if available**
If the business has historical interaction data (call logs, email archives, chat transcripts), I extract a sample — removing PII. Real data is always preferred because it captures actual user language, not what we imagine users say.

**Step 3 — Generate synthetic for gaps**
For intents/scenarios with no real data, or where real data can't be used, I generate synthetic examples using GPT-4o with few-shot prompting. I generate with variety: different levels of formality, different frustration levels, different languages (English, Tamil, Hindi for Indian deployments), abbreviations, typos for robustness testing.

**Step 4 — Annotate**
Label each example: intent, key entities, expected AI output, expected routing decision, expected response quality (pass/fail with reason). Define the annotation schema in a guidelines document first — ambiguous labels create noisy eval sets.

**Step 5 — Quality review**
Sample 20% for independent re-review. Discard low-quality or ambiguous examples. Track inter-annotator agreement.

**Step 6 — Split and version**
Split into evaluation (permanent, never changes once locked) and improvement (added to as new failure patterns are discovered in production). Version both with metadata: dataset_id, creation_date, source, review_status."

---

**Q: What is data lineage and why does it matter for AI deployments?**

**A:**
"Data lineage is the ability to trace any piece of data — from where it originated, through every transformation, to where it ends up.

In AI automation, this matters for three reasons:

**1. Debugging:** When an AI Worker gives a wrong output, you need to trace: what was the input? What was the system prompt version? What did the LLM receive? What did it return? What validation ran? What action was taken? Without lineage, debugging is guesswork.

**2. Compliance:** Regulated industries (healthcare, banking) require evidence of what data was processed and how. An auditor asks: 'Was patient data used to fine-tune this model?' Without lineage, you can't answer.

**3. Data quality accountability:** If a downstream system gets bad data, lineage tells you which step introduced the error. Was it the OCR? The extraction prompt? The validation rules? The integration mapping?

In practice, I implement lineage by:
- Assigning a unique trace_id to every item that enters a workflow
- Logging at each step: trace_id, step_name, input_hash, output_hash, model_version, timestamp
- Storing these logs in an append-only audit table
- For datasets: metadata files that record every transformation applied

This makes every AI-generated output auditable, explainable, and debuggable."

---

## PILLAR 3: Testing, Validation & Responsible AI

---

**Q: What do you include in a model card for an AI automation solution?**

**A:**
"A model card is the single document that tells any stakeholder everything they need to know about an AI solution — what it does, how well it works, where it fails, and who is responsible.

My model card template has 8 sections:

**1. Solution Overview**
What AI Worker is this? What business process does it automate? What LLM/platform version? Date created, owner.

**2. Intended Use**
What is this designed to do? What is it explicitly NOT designed to do? What user types does it serve?

**3. Performance Metrics**
Accuracy by intent/task category. Latency (p50, p95, p99). Throughput. Test dataset size and composition. Date last evaluated.

**4. Known Limitations and Failure Modes**
Where does it underperform? What input types cause errors? What is the workaround (human review queue)?

**5. Responsible AI Evaluation**
Bias testing results: accuracy across language variants, demographic proxies. Adversarial testing results. PII handling confirmation. Escalation logic summary.

**6. Data Governance**
What data does this solution process? Where does it go? How is PII handled? What is the retention policy? Does data leave the enterprise infrastructure?

**7. Operational Parameters**
Average cost per call. Rate limits and capacity. Dependencies (upstream APIs). Alert thresholds. Rollback procedure.

**8. Approval and Review**
Who reviewed this for production readiness? Who from BUIT approved go-live? Review date. Next scheduled review date.

The model card is a living document — updated after every significant change or incident."

---

**Q: How do you test an AI Worker for bias?**

**A:**
"I test along three bias dimensions specific to AI Workers:

**1. Linguistic bias**
Indian enterprise deployments see significant variation in English — Indian English phrasing, code-switching with Tamil/Hindi words, varying formality. I create evaluation subsets:
- Pure standard English
- Indian English ('kindly do the needful', 'please revert')
- Mixed language ('I need to reschedule my appointment, kal ke liye')
- Non-native fluency (grammatical errors, missing articles)

I compare accuracy across subsets. If the AI performs 92% on standard English but 74% on Indian English → that's a bias I must fix before deployment in India.

**2. Content bias**
For document processing: does accuracy hold across different document templates, vendors, amounts, date formats? Some prompts learn to extract 'the big number' as the total amount — they break when the biggest number is a line item, not the total.

**3. Demographic proxy bias**
For customer-facing workers: I test with inputs that have different implied demographics via names, phrasing patterns, apparent socioeconomic context. The AI should route, classify, and respond equivalently regardless.

**Documentation:**
Every bias test result is recorded in the model card — both passing and failing results. Failures are categorized by severity: blocking (can't deploy), significant (deploy with mitigation), minor (document and monitor).

**Fix approach:**
Linguistic bias → add diverse linguistic examples to the few-shot section of the prompt. Content bias → add format-diverse examples, add pre-normalization steps. Demographic bias → identify the proxy signal causing differential behavior, remove it from prompt context."

---

## PILLAR 4: Production Support & Continuous Improvement

---

**Q: How do you monitor an AI automation in production?**

**A:**
"I monitor across four dimensions:

**1. Accuracy quality (weekly spot-check)**
Pull a random sample of 50-100 outputs from the past week. Have a human review them against the correct expected output. Calculate accuracy rate. Compare to the established baseline. If it drops more than 3 percentage points → investigate.

**2. Technical health (real-time dashboard)**
- Error rate: % of calls that throw exceptions, timeout, or return malformed output
- Latency: p50, p95, p99 — alert if p95 exceeds SLA threshold
- Availability: % of time the workflow is accepting requests
- Guardrail activation rate: how often are guardrails firing? A sudden spike means either input quality changed or someone is probing the system

**3. Cost monitoring (daily)**
- Total tokens consumed per AI Worker
- Cost per successfully processed item
- Week-over-week trend — alert if >15% cost increase without proportional volume increase

**4. Business outcomes (weekly with BUIT)**
- Throughput: how many items processed this week vs. last week?
- Human escalation rate: what % of interactions required human intervention? This should trend down over time as the model improves.
- Resolution rate: % of interactions fully resolved by AI without fallback
- BUIT satisfaction: are the downstream systems receiving correct data?

**Alerting:**
I set automated alerts in the monitoring dashboard:
- Error rate > 2% for 30 consecutive minutes → PagerDuty/Slack alert
- Accuracy spot-check drops > 5% → Slack alert, engineering ticket
- Daily cost > 120% of 7-day average → email to manager

**Review cadence:**
- Daily: quick scan of technical health dashboard (5 minutes)
- Weekly: accuracy spot-check + cost review + BUIT weekly sync
- Monthly: full performance review + improvement planning"

---

**Q: How do you approach root cause analysis when a production AI Worker fails?**

**A:**
"I use a structured 5-layer RCA approach:

**Layer 1 — Reproduce the failure**
Find the exact input(s) that caused the failure. Pull from audit logs. Replay them in a staging environment to confirm the failure is reproducible.

**Layer 2 — Identify the failure point**
Which node in the workflow failed? Was it the LLM call (bad output), the validation (correctly caught a bad output), the integration (API downstream failed), or the routing logic?

**Layer 3 — Categorize the root cause**
- Input pattern change: new format, new language, new edge case type
- Prompt failure: the LLM misinterpreted an instruction in this case
- Model version change: LLM provider silently updated model behavior
- Integration failure: downstream API changed response format
- Volume impact: performance degrades under load that wasn't tested

**Layer 4 — Fix and verify**
Apply the fix. Retest on the original failing examples. Run full regression on the evaluation dataset — verify the fix doesn't break anything else.

**Layer 5 — Prevent recurrence**
Add the failing examples to the permanent evaluation set (so they're in the regression test forever). Update monitoring to catch this class of failure earlier. Document in an incident log: date, description, impact, root cause, fix, prevention.

I also run a blameless post-mortem with the team for any significant failure — the goal is improving the system, not assigning fault."

---

## PILLAR 5: MLOps & Platform Operations

---

**Q: How do you handle model or LLM version upgrades in production?**

**A:**
"LLM version upgrades are one of the trickiest operational challenges because the provider controls the timeline, not you, and even 'minor' model updates can change output behavior.

My process:

**1. Monitor for upgrade announcements**
Subscribe to provider changelogs. When GPT-4o-mini gets updated, I know before it affects production.

**2. Test in staging before it hits production**
If the platform allows model version pinning, I pin production to the current version while testing the new version in staging. Run the full evaluation dataset against the new model version. Compare accuracy, output format, latency.

**3. Threshold for action:**
- Same or better accuracy, compatible output format → schedule upgrade in maintenance window
- Accuracy within 2% but output format differs → update prompt and validators before upgrading
- Accuracy drops > 3% → investigate deeply, consider staying on old version, contact provider

**4. Upgrade procedure:**
- Update staging → run full eval → document results
- Upgrade production during low-traffic window
- Monitor intensively for 24 hours post-upgrade
- Update version registry

**5. Emergency rollback:**
If a surprise model update from the provider (no advance notice) degrades production:
- Pin to the last known-good model version (if provider supports it)
- If not → revert to the last prompt version that worked with the old model
- Escalate to provider with evidence of regression

**Lesson I've learned:** never rely on model version being stable. Always pin in production configs and have a tested rollback ready."

---

## PILLAR 6: Integration & Backend Enablement

---

**Q: An integration with an enterprise API is failing intermittently in production. How do you diagnose and fix it?**

**A:**
"Intermittent failures are the hardest to debug because they don't reproduce on demand. My approach:

**Step 1 — Collect failure evidence**
From audit logs: what time did failures occur? What was the request? What was the response (HTTP status code, error body)? Is there a pattern — time of day, specific input types, specific operations?

**Step 2 — Categorize by HTTP status**
- 401/403 Unauthorized → authentication issue: token expired, key rotated, permissions changed
- 429 Too Many Requests → rate limit hit: need exponential backoff, request queuing
- 500 Internal Server Error → the API itself is failing: is this correlated with their maintenance window?
- 503 Service Unavailable → temporary outage: retry logic should handle this
- Timeout → either the API is slow or our timeout setting is too aggressive

**Step 3 — Verify retry logic**
Every external API call in our workflows must have retry logic with exponential backoff. If a 503 error isn't being retried, that's a configuration gap I fix immediately.

**Step 4 — Rate limit analysis**
If 429s: how many calls are we making per minute? What's the API's rate limit? Am I calling it per item when I could batch? Do I need to add a request queue with a rate limiter?

**Step 5 — Coordinate with the API owner**
If the issue is on their side: provide specific timestamps, error logs, request IDs. Ask: did anything change in their API on this date? Is there a rate limit tier we need to upgrade?

**Prevention:**
Every integration config includes: retry policy (max retries, backoff strategy), timeout setting, circuit breaker threshold (if error rate > 50% → stop calling, alert), fallback behavior (if API is down → queue for retry later, don't fail the whole workflow)."

---

## PILLAR 7: Voice & Use-Case Enablement

---

**Q: How do you configure a scheduling AI Worker for voice channel?**

**A:**
"I'll walk through the full configuration for a voice scheduling worker:

**1. Conversation design**
Voice AI needs a conversation flow design before any platform configuration. I map every possible conversation branch:
- Happy path: user asks to book → AI confirms doctor and date → checks availability → confirms slot → sends reminder
- Reschedule path: user wants to change → AI identifies existing booking → proposes alternatives → confirms
- Cancellation: user cancels → AI confirms cancellation → offers rebooking
- Edge cases: no availability, unrecognized doctor name, unclear date, system error

**2. System prompt for voice**
```
You are a voice scheduling assistant for [Clinic Name].
Your ONLY function is appointment scheduling: book, reschedule, or cancel.
VOICE RULES:
- Maximum 2 sentences per response
- Never use lists, bullets, or markdown
- Use natural spoken English
- After confirming a booking, always say the date and time clearly
- If you need to pause to look up availability, say: 
  "Let me check that for you — one moment."
```

**3. Intent and entity extraction**
Configure the NLU to extract: intent (book/reschedule/cancel/info), doctor name, preferred date/time, patient name, patient ID if mentioned. Use structured output format for reliable JSON extraction.

**4. TTS configuration**
Select voice: warm, professional female or male voice appropriate for healthcare context. Set speech rate to slightly slower than default — healthcare calls often include elderly patients. Configure SSML for key moments: pause before reading back the confirmed date/time, emphasis on 'confirmed' and the doctor's name.

**5. Session context management**
Voice conversations need context management: the AI must remember what was said earlier in the same call. Configure conversation history to include the last 5-10 turns. Set session timeout: if silent for 30 seconds → say 'Are you still there?' → second timeout → gracefully end call.

**6. Escalation**
Configure: if user says 'transfer', 'agent', 'human', 'representative' → immediately offer transfer. If NLU intent confidence < 65% for two consecutive turns → offer transfer. If user expresses frustration sentiment (detected by tone/words) → empathize + offer transfer."

---

**Q: How do you configure context-aware tone and rhythm in a TTS voice AI?**

**A:**
"Context-aware TTS means the voice delivery adapts to the semantic context of what's being said — urgent information sounds urgent, confirmations sound warm, error messages sound apologetic.

**How I configure it:**

**1. SSML templates per context type**
I define SSML templates for different response categories:

Confirmation:
```xml
<speak>
  <prosody rate="95%" pitch="+2%">
    Your appointment has been confirmed.
  </prosody>
  <break time="400ms"/>
  <prosody rate="90%">
    <say-as interpret-as="date" format="mdy">{{date}}</say-as>
    at {{time}}, with {{doctor_name}}.
  </prosody>
</speak>
```

Urgent escalation:
```xml
<speak>
  <prosody rate="105%" pitch="+5%">
    This requires immediate attention.
  </prosody>
  <break time="200ms"/>
  Transferring you to our support team now.
</speak>
```

Apology/error:
```xml
<speak>
  <prosody rate="90%" pitch="-2%">
    I'm sorry, I wasn't able to complete that request.
  </prosody>
  <break time="400ms"/>
  Let me connect you with someone who can help.
</speak>
```

**2. Dynamic context signals to SSML mapping**
The workflow passes a context_type field (confirmation, error, urgent, information) along with the response text. The TTS node wraps the response in the corresponding SSML template before sending to the TTS engine.

**3. Testing voice quality**
I run listening tests with 5-10 human evaluators who rate: naturalness (1-5), tone match to context (1-5), clarity of key information (1-5). Any dimension below 4.0 → adjust SSML parameters.

**4. Indian English considerations**
For Indian deployments: configure the TTS to correctly pronounce Indian names, cities, and terms. AWS Polly and Azure Neural TTS have Indian English voice models that handle these better than default US English models."
