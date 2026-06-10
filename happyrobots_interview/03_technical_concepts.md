# Technical Concepts Deep-Dive — Happy Robots Manager Role

---

## SECTION 1: LLM Prompting & Policy Engineering

### Prompt Engineering vs Policy Engineering vs Guardrail Configuration

These three are related but distinct — the JD mentions all three separately, so know the difference.

**Prompt Engineering:**
Crafting the instructions that tell the LLM what to do, how to do it, and what format to output.
- System prompt: role definition, task instructions, output format, edge case handling
- Few-shot examples: 2-5 input/output demonstrations
- Chain-of-thought: "think step by step before answering"
- Grounding instruction: "base your answer ONLY on the context provided below"

**Policy Engineering:**
Defining the behavioral rules for the AI Worker — what it is ALLOWED to do, what it is NOT ALLOWED to do, and how it handles the boundary cases.

A policy is a set of rules that govern the AI's decision-making space:
```
POLICY: Appointment Scheduling Worker
ALLOWED:
  - Schedule, reschedule, cancel appointments
  - Retrieve availability for doctors in the network
  - Send confirmation messages

NOT ALLOWED:
  - Provide medical advice or diagnosis
  - Access patient medical records beyond appointment history
  - Make commitments outside the current system's capabilities

ESCALATION RULES:
  - If user expresses emergency or distress → immediately transfer to human
  - If user asks for information outside allowed scope → decline, offer transfer
  - If confidence in intent < 70% → clarify before acting
```

Policy engineering is often done in a policy layer separate from the main system prompt — it's a governance document that the AI is instructed to follow.

**Guardrail Configuration:**
Technical enforcement layer that catches policy violations AFTER the LLM generates a response but BEFORE it reaches the user.

Types of guardrails:
- **Content filters:** block responses containing prohibited keywords, harmful content, PII leakage
- **Topic filters:** block responses on out-of-scope topics (if an appointment bot starts discussing politics)
- **Format validators:** ensure the response is valid JSON, contains required fields, within length limits
- **Confidence thresholds:** if the LLM's classification confidence is below 70%, route to human
- **Injection detection:** block attempts to override the system prompt via user input

Tools: AWS Bedrock Guardrails, Llama Guard, custom Pydantic validators, NVIDIA NeMo Guardrails

---

### Production Prompt Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     SYSTEM PROMPT                           │
│  [Role Definition]                                          │
│  You are a scheduling AI Worker for [Company]. You help     │
│  customers book, reschedule, and cancel appointments.       │
│                                                             │
│  [Policy Section]                                           │
│  ALWAYS: Confirm the customer's name before booking.        │
│  NEVER: Discuss topics outside appointment management.      │
│  ESCALATE IF: Customer mentions medical emergency.          │
│                                                             │
│  [Tone & Channel Instructions]                              │
│  This is a VOICE channel. Keep responses under 2 sentences. │
│  Use natural spoken language — no lists, no markdown.       │
│                                                             │
│  [Output Format]                                            │
│  Return JSON: {"intent": "...", "action": "...",            │
│  "response_text": "...", "confidence": 0.0-1.0}             │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    FEW-SHOT EXAMPLES                        │
│  Example 1: User: "I need to see Dr. Kumar next Friday"     │
│  Output: {"intent": "schedule_new", "action":               │
│  "check_availability", "response_text": "I'd be happy to   │
│  help you schedule with Dr. Kumar. May I have your name?",  │
│  "confidence": 0.97}                                        │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                   CONVERSATION CONTEXT                      │
│  [Previous turns of the conversation — last N turns]        │
│  Customer: "I'd like to reschedule my appointment"          │
│  Agent: "Of course. What's the appointment date?"           │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    CURRENT USER INPUT                       │
│  "Actually it was this Thursday with Dr. Sharma"            │
└─────────────────────────────────────────────────────────────┘
```

---

### Common Prompt Failure Modes (Know These)

| Failure Mode | Cause | Fix |
|-------------|-------|-----|
| Format inconsistency | Temperature too high, format not enforced | Use structured output API / function calling |
| Instruction override | User input contains prompt injection | Sanitize inputs, add injection detection guardrail |
| Context bleed | Previous conversation leaks into new session | Strict session management, clear context boundaries |
| Verbose responses in voice | Prompt doesn't specify voice brevity | Add explicit voice brevity instructions + max token limit |
| Hallucinated data | LLM generates facts not in context | Ground prompt: "Answer ONLY from context provided below" |
| Ambiguity collapse | Multiple valid interpretations → AI picks one silently | Add clarification instruction: "If ambiguous, ask once" |

---

## SECTION 2: Voice AI — TTS, STT, Intent Fidelity

This is a unique dimension of this role. Most GenAI engineers haven't worked in voice — knowing this well sets you apart.

### Voice AI Architecture (Full Stack)

```
USER SPEAKS
    ↓
[STT — Speech-to-Text]
Converts audio to text
Tools: Azure Speech, AWS Transcribe, Google Speech-to-Text, Whisper
Key metric: Word Error Rate (WER) — lower is better
    ↓
[NLU — Natural Language Understanding]
Extracts: Intent ("schedule_appointment"), Entities (doctor: "Dr. Kumar", date: "Friday")
Tools: LLM-based NLU or traditional (Rasa, Dialogflow, Luis)
    ↓
[Orchestration / Workflow]
Decides what action to take based on intent + entities
Calls backend APIs, retrieves data, performs actions
    ↓
[Response Generation]
LLM generates the spoken response
CRITICAL: must be optimized for SPOKEN delivery (not text)
    ↓
[TTS — Text-to-Speech]
Converts text response to natural audio
Tools: Azure Neural TTS, AWS Polly, Google Cloud TTS, ElevenLabs, Play.ht
Key parameters: voice, speed, pitch, pause placement (SSML)
    ↓
USER HEARS THE RESPONSE
```

---

### What "Context-Aware TTS" Means

Not all TTS is the same. A high-quality voice AI configures TTS to match:

**Tone:**
- Professional / Formal: "Your appointment has been confirmed for Thursday, March 20th at 2 PM."
- Warm / Empathetic: "Of course, I understand. Let me help you reschedule that right away."
- Urgent: "This is a time-sensitive notification — your shipment requires action."

**Rhythm and pacing:**
- Normal business context: standard speed
- Medical / elderly / accessibility context: slower, clearer pronunciation
- Notifications/alerts: slightly faster, direct

**SSML (Speech Synthesis Markup Language):**
SSML is the markup language for controlling TTS behavior:

```xml
<speak>
  Your appointment has been confirmed for 
  <say-as interpret-as="date" format="mdy">3/20/2025</say-as>.
  <break time="500ms"/>
  Please arrive 
  <emphasis level="moderate">15 minutes early</emphasis>
  for paperwork.
  <break time="300ms"/>
  Is there anything else I can help you with?
</speak>
```

**Intent Fidelity:**
The spoken response must convey the SAME intent as the written response — not just be grammatically correct. A message about a payment failure should sound urgent, not casual. A booking confirmation should sound warm, not robotic.

Testing intent fidelity: play the TTS output to human listeners and ask "what emotion does this convey?" Compare to intended emotion. If there's a mismatch → adjust tone, add SSML emphasis, choose a different voice model.

---

### Key Voice AI Metrics

| Metric | What It Measures | Target |
|--------|----------------|--------|
| WER (Word Error Rate) | STT accuracy | < 5% for English, < 10% for Indian English |
| Intent Accuracy | NLU correctly identifies intent | > 93% |
| Entity Accuracy | Correct extraction of names, dates, IDs | > 96% |
| MOS (Mean Opinion Score) | Human-rated naturalness of TTS (1-5) | > 4.0 |
| CSAT | Customer satisfaction after voice interaction | > 80% |
| Containment Rate | % calls resolved by AI without human transfer | > 70% |
| First Call Resolution | Issue resolved in one call | > 65% |

---

## SECTION 3: Data Annotation & Synthetic Data

### Annotation Types You Need to Know

**Intent Labeling:**
Each user utterance gets labeled with an intent category.
```
"I need to book an appointment" → intent: schedule_new
"Can I change my Tuesday slot?" → intent: reschedule
"Forget it, cancel everything" → intent: cancel
"What are your timings?" → intent: information_request
```

**Entity Labeling (Named Entity Recognition / NER):**
Tag specific values within text.
```
"I need to see [Dr. Kumar:DOCTOR] on [Friday March 20:DATE] at [2pm:TIME]"
```

**QA Labeling (for evaluation datasets):**
For each test case, record: input, expected output, actual output, is_correct (yes/no), error_category (if no).

**Sentiment Labeling:**
"I'm really frustrated with this" → sentiment: negative, escalation_needed: true

---

### Annotation Quality Control

**Inter-Annotator Agreement (IAA):**
Have two annotators label the same 100 examples independently. Measure how often they agree.
- Cohen's Kappa > 0.8 = strong agreement, annotation schema is clear
- Kappa < 0.6 = ambiguous schema, needs revision before full annotation begins

**Annotation Guidelines:**
A written document that defines every label with examples and edge cases. Without this, different annotators apply labels differently, creating noise in training data.

**Sampling Audit:**
After full annotation, randomly sample 10% for re-review by a senior annotator. Flag systematic errors.

---

### Synthetic Data Generation

**When to use it:**
- Real data is scarce (new product with no conversation history)
- Real data is sensitive (can't use customer PII in eval dataset)
- Coverage gaps (evaluation set has no examples of a specific edge case)
- Class imbalance (95% of intents are "schedule_new", only 2% are "cancel" — augment the minority class)

**Technique 1: Direct LLM Generation**
```
Prompt: "Generate 15 realistic examples of a customer calling to cancel an appointment.
Vary: level of frustration (1=calm, 5=very frustrated), reason given (personal/schedule conflict/no reason),
communication style (formal/casual/terse).
Format: one utterance per line, label the frustration level and reason in [brackets]."
```

**Technique 2: Paraphrase Augmentation**
Take real examples and generate paraphrases:
```
Original: "I want to reschedule my appointment"
Generated: ["Can I move my appointment?", "I need to change my booking", 
"Is it possible to pick a different time?", "I won't be able to make it — can we shift it?"]
```

**Technique 3: Back-Translation**
Translate to another language and back — produces natural variation without semantic change.
English → Tamil → English = different phrasing, same meaning.

**Data Lineage Tracking:**
Every dataset artifact must have:
```json
{
  "dataset_id": "sched_intents_v3",
  "creation_date": "2025-06-01",
  "source": "synthetic",
  "generator_model": "gpt-4o",
  "generator_prompt_version": "v2.1",
  "human_review_rate": 0.30,
  "human_review_pass_rate": 0.94,
  "total_examples": 500,
  "label_schema_version": "v4",
  "status": "approved_for_evaluation"
}
```

---

## SECTION 4: MLOps for GenAI Platforms

### What MLOps Means Here (Different from Traditional ML)

Traditional ML MLOps: train → serve → monitor → retrain model weights
GenAI platform MLOps: configure → deploy → monitor → improve prompts/configs (no weight retraining)

Your "model artifacts" are:
- System prompts (versioned text files)
- Policy documents (versioned JSON/YAML)
- Workflow configurations (platform export files)
- Evaluation datasets (versioned datasets)
- Integration configs (API endpoints, auth config, field mappings)

---

### Model/Prompt Version Registry

```
| Solution         | Component     | Prod Version | Last Updated | Accuracy | Deployed By |
|-----------------|--------------|-------------|-------------|---------|------------|
| Scheduling Worker | System Prompt  | v4.2        | 2025-05-15  | 96.3%   | K. Vanan   |
| Scheduling Worker | Policy Doc     | v2.0        | 2025-04-01  | -       | K. Vanan   |
| Invoice Extraction| Extraction Prompt | v7.1    | 2025-06-01  | 97.1%   | J. Priya   |
| Invoice Extraction| Validation Rules | v3.0     | 2025-05-20  | -       | K. Vanan   |
```

---

### Deployment Pipeline (CI/CD for GenAI Configs)

```
[Developer edits prompt in staging environment]
            ↓
[Automated evaluation run]
Eval dataset → new prompt version → score
Is score ≥ production baseline? → PASS / FAIL
            ↓ PASS
[Code review / peer review of prompt change]
            ↓ Approved
[Canary deployment: 10% traffic]
Monitor for 24h — error rate, latency, accuracy sampling
            ↓ No degradation
[Full deployment: 100% traffic]
Update version registry
Notify BUIT stakeholder
            ↓
[Post-deployment monitoring]
24h, 48h, 7d accuracy spot-checks
```

---

### Rollback Strategy

Every production deployment must have a documented rollback plan:

1. **What triggers rollback:** accuracy drops >5% vs baseline, error rate >3%, latency p95 >5s
2. **Who can trigger rollback:** on-call engineer, no approvals needed (speed matters)
3. **How to rollback:** revert to previous version in version registry, redeploy — target: <15 minutes
4. **Post-rollback:** root cause analysis within 24 hours, document in incident log

---

## SECTION 5: Document Processing

### Document Processing Pipeline Architecture

```
INTAKE
PDF/Image arrives via email attachment, API upload, or file system trigger
        ↓
CLASSIFICATION
What document type is this? (Invoice, Contract, ID document, Report)
LLM classifier or rule-based classifier based on layout/keywords
        ↓
PRE-PROCESSING
For digital PDFs → extract text directly (pdfplumber, PyMuPDF)
For scanned images → OCR (AWS Textract, Google Document AI, Azure Form Recognizer)
Output: clean text + layout metadata (bounding boxes, page numbers, sections)
        ↓
EXTRACTION
LLM extracts structured fields defined by the document type
Structured output / function calling for reliable JSON output
Pydantic validation for every extracted field
        ↓
VALIDATION
Business rules engine (NOT LLM):
- Required fields present?
- Formats correct (date, currency, account numbers)?
- Cross-field logic (line items sum = total amount?)?
- Deduplication check (same document ID seen before?)?
        ↓
ROUTING
All rules pass → auto-process queue
Validation failure → human review queue (flagged fields highlighted)
Confidence below threshold → human review queue
        ↓
SYSTEM HANDOFF
Write extracted data to target system (ERP, CRM, database)
Archive original document + extraction JSON in storage (S3, SharePoint)
Update audit log
Send notification to downstream team
```

---

### Tools for Document Processing

| Task | Tool Options |
|------|-------------|
| Digital PDF text extraction | pdfplumber, PyMuPDF, pdfminer |
| OCR for scanned documents | AWS Textract, Google Document AI, Azure Form Recognizer |
| Layout-aware extraction | AWS Textract (tables, forms), Google Document AI |
| LLM extraction | GPT-4o with structured output, Claude 3.5 Sonnet |
| Output validation | Pydantic (Python) |
| Document classification | LLM zero-shot or fine-tuned BERT |

---

## SECTION 6: API Integration Patterns

### RESTful API Integration in Automations

Every AI Worker needs to connect to backend systems. Key patterns:

**Authentication:**
- API Key: add to header `Authorization: Bearer {key}` — store in Secrets Manager
- OAuth 2.0: client credentials flow for server-to-server, authorization code for user context
- SAML/SSO: enterprise identity provider integration

**Error Handling:**
```python
import httpx
import asyncio

async def call_crm_api(record_id: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{CRM_BASE_URL}/records/{record_id}",
                    headers={"Authorization": f"Bearer {get_secret('crm_token')}"},
                    timeout=10.0
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:  # Rate limit
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue
            elif e.response.status_code == 401:  # Auth failure
                refresh_token()  # Refresh and retry once
                continue
            else:
                raise
    raise MaxRetriesExceeded(f"CRM API failed after {retries} attempts")
```

**Webhook Receivers:**
```python
from fastapi import FastAPI, Request, HTTPException
import hmac, hashlib

app = FastAPI()

@app.post("/webhook/crm-update")
async def receive_crm_webhook(request: Request):
    # Verify webhook signature to prevent spoofing
    signature = request.headers.get("X-Webhook-Signature")
    body = await request.body()
    expected = hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    payload = await request.json()
    # Process the event
    await queue_for_processing(payload)
    return {"status": "accepted"}
```

---

## SECTION 7: Responsible AI Framework

### The Four Evaluation Dimensions (Know Each Deeply)

**1. Bias and Fairness**
Definition: The AI should perform equivalently across different groups of users.

Testing approach:
- Create evaluation subsets: English inputs vs. Tamil/Hindi inputs, formal vs. informal language, urban vs. rural phrasing
- Compare accuracy across subsets — are there significant drops?
- For classification: check if false positive/negative rates differ by demographic proxy

Metrics:
- Demographic parity: P(output=positive | group A) ≈ P(output=positive | group B)
- Equal opportunity: true positive rate should be equal across groups
- Disparate impact ratio: accuracy(minority group) / accuracy(majority group) > 0.8 is acceptable

**2. Ethical Compliance**
- Does the AI generate harmful, offensive, or manipulative content?
- Does it respect user autonomy (not coerce decisions)?
- Does it maintain privacy (not reveal one user's data to another)?
- Does it accurately represent its AI nature when asked?
- Is the AI honest about its limitations?

Testing: adversarial red-teaming — try to get the AI to violate each principle. Document results.

**3. Explainability and Transparency**
- For every significant decision the AI makes, can you trace WHY?
- Decision log: input → intent classification → action taken → reason for action
- For guardrail activations: what rule triggered, why
- For escalations: what signal caused the transfer

Techniques:
- Chain-of-thought logging: capture the model's reasoning chain alongside outputs
- Rule attribution: when a policy/guardrail fires, log which specific rule
- Confidence scores: include with every output

**4. Model Card and Decision Records**
See behavioral questions section for Model Card details.

Decision Records: document why key design decisions were made:
- Why was this model chosen over alternatives?
- Why is the accuracy threshold set at 95% and not 90%?
- Why does escalation trigger at < 70% confidence?
- Who approved these thresholds and when?

---

## SECTION 8: Cost & Token Management

### Token Budget Management

```python
# Count tokens before sending to manage cost
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def build_prompt_within_budget(
    system_prompt: str,
    conversation_history: list[dict],
    user_input: str,
    max_tokens: int = 4000
) -> list[dict]:
    """Trim conversation history to fit within token budget."""
    base_tokens = count_tokens(system_prompt) + count_tokens(user_input) + 500  # buffer
    available_for_history = max_tokens - base_tokens
    
    trimmed_history = []
    current_tokens = 0
    for turn in reversed(conversation_history):
        turn_tokens = count_tokens(str(turn))
        if current_tokens + turn_tokens > available_for_history:
            break
        trimmed_history.insert(0, turn)
        current_tokens += turn_tokens
    
    return [{"role": "system", "content": system_prompt}] + trimmed_history + [{"role": "user", "content": user_input}]
```

### Model Routing for Cost Optimization

```python
def select_model(task_type: str, input_complexity: str) -> str:
    routing_table = {
        ("classification", "simple"): "gpt-4o-mini",
        ("classification", "complex"): "gpt-4o-mini",
        ("extraction", "simple"): "gpt-4o-mini",
        ("extraction", "complex"): "gpt-4o",
        ("reasoning", "any"): "gpt-4o",
        ("summarization", "any"): "gpt-4o-mini",
        ("voice_response_generation", "any"): "gpt-4o-mini",
        ("document_analysis_long", "any"): "claude-3-5-sonnet",  # 200K context
    }
    return routing_table.get((task_type, input_complexity), "gpt-4o-mini")
```

### Cost Monitoring

Track daily:
- Total tokens consumed per solution/AI Worker
- Cost per call (input tokens × input price + output tokens × output price)
- Cost per successfully processed unit (successful invoice / successful booking)
- Week-over-week cost trend — alert if >15% increase without volume increase

Dashboard query example:
```sql
SELECT 
    solution_name,
    DATE(created_at) as date,
    SUM(input_tokens) as total_input_tokens,
    SUM(output_tokens) as total_output_tokens,
    SUM(input_tokens * 0.00015 + output_tokens * 0.0006) / 1000 as cost_usd,
    COUNT(*) as call_count,
    (SUM(input_tokens * 0.00015 + output_tokens * 0.0006) / 1000) / COUNT(*) as cost_per_call
FROM llm_call_log
WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY solution_name, DATE(created_at)
ORDER BY cost_usd DESC;
```
