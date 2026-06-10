# Behavioral Interview Questions — Manager, GenAI Automation

---

## STAR Format — Always

**S** — Situation (2 sentences max, context only)
**T** — Task (what YOU were accountable for, not the team)
**A** — Action (what YOU specifically did — 60% of the answer)
**R** — Result (quantified outcome + what you changed afterward)

At Manager level, every answer must show: **ownership, judgment, cross-functional influence, outcome accountability.**

---

## Category 1: Deployment & Configuration

---

### "Tell me about an end-to-end AI automation you configured and deployed."

**What they're evaluating:** Do you own the full lifecycle or just one piece? Can you articulate architecture decisions?

**STAR template:**
- S: Business context — what problem, which team, what channel
- T: You were responsible for designing, configuring, and deploying the AI Worker
- A (structure your answer around the 7 pillars):
  1. Prompt design: what was your system instruction strategy? How did you handle edge cases in prompts?
  2. Workflow: how did you orchestrate multi-step logic?
  3. Integrations: which systems did you connect? What auth challenges?
  4. Testing: what was your evaluation approach? What accuracy did you baseline?
  5. Guardrails: what safety rules did you configure?
  6. Deployment: what was your rollout strategy? Phased? Blue-green?
  7. Monitoring: what did you set up to watch after launch?
- R: Volume handled, accuracy achieved, time saved, cost per transaction

**Key phrases:**
- "Before writing a single prompt, I defined what 'correct' looked like and built 80 evaluation examples..."
- "I used a phased rollout — 10% of traffic first, monitored for 48 hours, then expanded..."
- "The integration challenge was OAuth token refresh — I added a retry handler that..."

---

### "Describe a time you had to configure a multi-channel AI experience (voice + digital)."

**What they're evaluating:** The voice AI dimension is specific to this JD. They want someone who's thought beyond chat.

**Key things to cover:**
- How the input pathway differs: voice → STT → text → LLM → response → TTS → audio
- How prompt design must account for voice: shorter responses, no markdown, natural spoken phrasing
- How TTS configuration works: tone selection (professional vs. warm), speech rate, pause insertion
- Intent fidelity: the spoken response must convey urgency/empathy/clarity that matches the context
- How you tested voice quality: human listening tests, word error rate on STT, naturalness scoring on TTS

**If you haven't done voice specifically:** Be honest, frame your transferable experience:
"My voice AI exposure has been through [related context]. What I understand from working with multi-channel architectures is that voice adds a layer of... I'd ramp on the HappyRobots voice configuration specifically, but the core prompt engineering and workflow logic transfers directly."

---

## Category 2: Data & Model Enablement

---

### "Tell me about a time you worked with annotation or data preparation for an AI model."

**What they're evaluating:** Real data work — not just calling APIs. Do you understand data quality as a foundation?

**STAR structure:**
- S: AI model or evaluation pipeline needed labeled data
- T: You were responsible for creating/managing the annotation process
- A:
  1. Annotation schema design: what categories/labels? How did you define edge cases?
  2. Annotation tooling: what did you use? (Label Studio, Prodigy, custom tooling, spreadsheet)
  3. Quality control: inter-annotator agreement (did two annotators agree?), audit sampling
  4. Synthetic data: if real data was scarce, how did you generate synthetic examples?
  5. Dataset management: how did you version the dataset, track lineage?
- R: Dataset size, annotation quality score, downstream model/eval accuracy improvement

**Key concept to demonstrate:** You understand that bad labels = bad evaluation = false confidence in production quality. Data quality is upstream of everything.

---

### "How do you approach generating synthetic data when real data is insufficient?"

**Strong answer:**
"Synthetic data generation is a lever I use specifically when real data has three problems: it's scarce, it's sensitive (PII that can't be used in a training pipeline), or it doesn't cover edge cases well.

My typical approach:

**1. Define what's missing.** Run the existing evaluation dataset through the model and find the failure categories. Those failure categories are where I need more data — not generic volume.

**2. Prompt-based generation.** I use a strong LLM (GPT-4o or Claude) with few-shot examples to generate synthetic variants. For example: 'Generate 20 examples of appointment requests where the user mentions a preferred doctor, in these 5 linguistic styles: [formal, casual, confused, urgent, non-native speaker].'

**3. Human review of synthetic data.** Synthetic data is never automatically trusted. I sample 20-30% for human review before adding to the evaluation set. Low-quality generations get discarded.

**4. Avoid distributional collapse.** If all synthetic data sounds like GPT wrote it, the evaluation is now testing GPT against GPT. I ensure variety by varying generation parameters, using multiple generator prompts, and mixing with real data.

**5. Track lineage.** Every dataset artifact has a metadata file: source (real/synthetic), generation date, generator model and prompt version, annotator, quality review status. This is non-negotiable for audit-readiness."

---

## Category 3: Testing & Responsible AI

---

### "Walk me through your testing approach for an AI automation before it goes to production."

**What they're evaluating:** Do you have a systematic quality gate, or do you just 'try it and see'?

**Full structured answer:**

"I run four test phases before any AI solution goes to production:

**Phase 1 — Unit testing (individual components)**
- Test each prompt in isolation with the evaluation dataset
- Measure field-level accuracy: which fields are error-prone?
- Test guardrails: can I make the AI violate its own policy? (adversarial testing)
- Target: 95%+ accuracy on standard cases, 85%+ on edge cases

**Phase 2 — Integration testing (connected components)**
- Test the full workflow end-to-end with mock data
- Verify API integrations: does the CRM call return the right format? Does the error handler trigger correctly?
- Test auth and credential flows: what happens when a token expires mid-workflow?
- Test failure modes: what happens when an upstream API is down? Does the fallback work?

**Phase 3 — Performance testing (scale and speed)**
- Simulate concurrent load: 50 simultaneous requests, 500, 5000 — where does latency degrade?
- Measure p50, p95, p99 latency
- Test under real document sizes, not just small test fixtures
- Token cost measurement under realistic volume

**Phase 4 — Responsible AI validation**
- Bias testing: run the same workflow with inputs that vary by apparent demographic, language, name origin — does accuracy hold?
- Adversarial testing: try to make the AI say something harmful, reveal system prompts, bypass guardrails
- Explainability check: for each output, can I trace the decision logic? (Important for regulated workflows)
- Privacy check: does any PII accidentally appear in logs, outputs, or error messages?

Only after all four phases pass at defined thresholds does the solution go to a limited production pilot."

---

### "Tell me about a time you caught a bias or fairness issue in an AI solution."

**What they're evaluating:** Have you actually looked for this, or is it theoretical?

**Structure:**
- S: AI classification or extraction task deployed or near-deployment
- T: You were responsible for Responsible AI validation
- A:
  1. You designed a bias testing matrix: varied inputs by language, name origin, industry terminology, formality level
  2. You ran the evaluation and found: [e.g., accuracy dropped significantly for inputs in Tamil/Hindi vs English, or extracted amounts were wrong when formatted in Indian number system]
  3. You documented the finding as a bias flag — specific category, severity, affected volume
  4. Fix: updated prompt with explicit handling instructions and examples, or added a pre-processing step to normalize input format
  5. Re-tested: verified improvement across all demographic categories
  6. Documented in model card: known limitation, mitigation applied, residual risk level
- R: Caught before production, fixed, documented transparently

**Key point:** Bias isn't always demographic — it can be format bias, language bias, complexity bias, length bias. Show breadth of thinking.

---

### "What is a Model Card and have you ever written one?"

**Answer:**

"A Model Card is a standardized documentation artifact that describes an AI model or AI-powered solution's capabilities, limitations, intended use, and performance characteristics. It's the equivalent of a product specification sheet for an AI deployment.

A well-written Model Card covers:
1. **Model overview** — what it does, what LLM/platform it uses, version, date
2. **Intended use** — what use cases it's designed for, what it's NOT designed for
3. **Performance metrics** — accuracy on each category, latency benchmarks, test dataset size
4. **Limitations and known failure modes** — documented edge cases and their frequency
5. **Responsible AI evaluation results** — bias testing results, fairness metrics, adversarial testing summary
6. **Data governance** — what data was used, PII handling, retention policy
7. **Operational parameters** — cost per call, rate limits, dependencies, rollback procedure
8. **Approval and sign-off** — who reviewed and approved this for production

Yes, I've written Model Cards for [specific deployments]. The biggest value is not the documentation itself — it's that writing one forces you to think clearly about what the system does, where it can fail, and who is accountable if it does. It also makes handover to other teams or BUIT stakeholders much cleaner."

---

## Category 4: Production Support & MLOps

---

### "An AI automation you deployed starts showing degraded accuracy 3 weeks after launch. How do you handle it?"

**Full answer — this is a critical Manager scenario:**

"Three weeks is the classic timeline for real-world data to diverge from test data — users find edge cases, data formats change, upstream systems behave differently than in testing.

**Immediate (Day 1):**
- Quantify the degradation: pull a sample of recent outputs, compare to expected — what's the actual error rate and on what categories?
- Assess impact: are these errors causing wrong decisions downstream, or going to a human review queue? Severity determines urgency.
- Communicate to BUIT: 'We've identified a quality drop in [specific area], we're investigating. Here's the current error rate and the impact. ETA for fix: [X].'

**Root cause (Day 1-2):**
- Look at the failing examples — what do they have in common? New input format? New edge case type? Longer inputs than test data? Different language/dialect?
- Check if any upstream system changed: did the CRM API start returning a different field format? Did the document template change?
- Check the LLM provider: did the model version update? (OpenAI occasionally silently updates models)
- Compare current prompt version to the state at launch — any changes?

**Fix:**
- For prompt failures: add few-shot examples covering the new failure category
- For input format changes: add pre-processing normalization
- For volume-related failures: check token limits, chunking strategy
- For model version drift: pin to a specific model version

**Governance:**
- Update evaluation dataset with the failing examples so they're in the regression set permanently
- Document in the change log: what degraded, root cause, fix applied, re-test results
- Update the Model Card with the new known limitation and fix

**Prevention going forward:**
- Add automated accuracy sampling — weekly pull of 50 random outputs for human spot-check
- Add a drift detection alert: if error rate exceeds X% for 3 consecutive days, auto-alert

The lesson I always take from these: test data is a snapshot; production is a living stream. Your monitoring is only as good as your sampling strategy."

---

### "How do you manage prompt versions in a production environment?"

**Answer:**

"I treat prompts exactly like code — versioned, tested before deployment, with a clear rollback procedure.

**Version control:**
- All prompts live in a Git repository, not in the platform UI directly
- Each prompt file has a version header: name, version number, date, author, change summary
- Changes are made in a branch, reviewed in a PR, merged only after test results are attached

**Version registry:**
- I maintain a registry table (in a database or even a managed spreadsheet for smaller operations) that tracks: solution name, current production version, last change date, accuracy score at last test, who approved for production
- This answers the question 'what's running in production right now?' in under 30 seconds

**Deployment process:**
1. New prompt version developed in staging environment
2. Run full evaluation dataset — must match or exceed production baseline accuracy
3. If passes: deploy to 10% traffic (canary)
4. Monitor for 24 hours — no degradation → expand to 100%
5. Update registry with new production version

**Rollback:**
- Rollback = revert to the previous version in Git + redeploy
- Target rollback time: under 15 minutes
- This is why we never make changes directly in the UI — there's no rollback from a UI edit

**Why this matters for HappyRobots specifically:**
In a low-code platform, the temptation is to just edit the prompt in the UI and hit save. That's fine for prototyping. For production, it's how you create invisible drift, untested changes, and accountability gaps."

---

## Category 5: Cross-Functional & Leadership

---

### "How do you work with a BUIT team that has unrealistic expectations for what AI can do?"

**Answer:**

"This is one of the most common dynamics in GenAI automation work, and it's one of the reasons the manager role is important.

Business teams have often seen demos of AI doing amazing things and assume production AI works the same way. My approach is to reset expectations early — before design, not after a failed deployment.

**I use a specific framing:**
'AI automation works on a spectrum from fully reliable to fully unreliable depending on the task type. Classification tasks with clear categories: 97%+ reliable. Open-ended generation with no validation: 60-70% reliable. Our job is to design workflows that put the reliable AI in the automated path and route the unreliable cases to human review.'

**Then I map their requirements to that spectrum:**
- 'Automatically approve vendor invoices' → where on the spectrum? What's the failure blast radius? → I propose a threshold-based design: auto-approve below $10K from approved vendors, human review above
- 'AI should answer any customer question' → too broad → I narrow: 'AI handles the top 20 question types covering 80% of volume; anything outside that routes to an agent with AI-prepared context'

**The documentation that helps:**
I always write a Solution Design Document before configuration begins — it captures agreed scope, agreed accuracy targets, agreed failure handling, and who approves the go-live. BUIT signs off. This prevents the 'that's not what I expected' conversation 3 months later.

The goal is never to under-deliver. It's to agree on what 'delivery' means before you build."

---

### "Tell me about a time you had to explain an AI decision to a non-technical stakeholder."

**Answer structure:**
- S: AI automated a decision that a stakeholder challenged ("why did the system reject this vendor?")
- T: You needed to explain the AI's reasoning in business terms, not technical terms
- A:
  1. Retrieved the audit log for that specific decision — exact input, prompt version, output, confidence score
  2. Translated: "The system flagged this vendor because the invoice date was 97 days ago — our policy threshold is 90 days. That threshold was set by your AP team during configuration."
  3. Showed them the explainability trail: input → rule → output — not a black box
  4. Identified if the threshold was wrong or the input was wrong → proposed adjustment
  5. Updated documentation to make this threshold visible and adjustable by BUIT without needing engineering
- R: Stakeholder trust restored, threshold made configurable in the UI, similar complaints dropped

**Key principle to state:** Explainability is a design requirement, not an afterthought. Every consequential AI decision must have a traceable reason. I build audit logs from day one.
