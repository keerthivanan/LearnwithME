# Case Study Preparation — Full Deep Dive

## How to Approach ANY Case Study

When given a case, interviewers watch for:
1. Do you ask the right questions before jumping to solutions?
2. Do you structure your thinking (not rambling)?
3. Do you connect features to specific business outcomes?
4. Do you identify risks or assumptions proactively?
5. Do you quantify impact?

### Universal Case Study Framework

```
STEP 1: CLARIFY (2 minutes)
  → Understand the full context before saying anything else
  → "Before I design a solution, can I ask a few questions?"
  → Never skip this — it shows maturity

STEP 2: DIAGNOSE (3 minutes)
  → Summarize the core problem in one sentence
  → Identify the key stakeholders and what each needs
  → Identify the constraints (timeline, budget, tech)

STEP 3: DESIGN (5 minutes)
  → Map pain points → Edrevel modules
  → Prioritize with MoSCoW
  → Build a phased roadmap

STEP 4: QUANTIFY (2 minutes)
  → Estimate the impact: time saved, cost saved, risk reduced
  → Define success metrics: how will we know it worked?

STEP 5: RISKS & ASSUMPTIONS (1 minute)
  → What could go wrong?
  → What are you assuming that should be validated?
```

---

## Case Study 1: Global Pharmaceutical Company — GxP Compliance

**The Scenario:**
MedPharma Corp is a pharmaceutical company with 4,500 employees across India, Germany, and the US. They manufacture drugs that are subject to FDA (US), EMA (Europe), and CDSCO (India) regulations. Every employee in manufacturing, QA, and R&D must complete mandatory GxP (Good Practice) training every year. Current state: 55% completion rate tracked in three separate Excel files maintained by three HR coordinators in each country. Last audit flagged them for incomplete training documentation. Next audit is in 4 months. If they fail again, they risk a Warning Letter from FDA which could impact drug approval timelines.

---

**Discovery Questions You'd Ask:**

1. "How many employees need to complete GxP training? Is it all 4,500 or a specific subset?" → (Answer: ~2,800 in manufacturing, QA, and R&D)
2. "What are the different types of GxP training? Is it one course or a curriculum?" → (Answer: 7 mandatory programs — GDP, GMP, GLP, SOP training, equipment qualification, data integrity, document control)
3. "Are the training requirements the same across India, Germany, and US, or does each country have different requirements?" → (Answer: core programs are same, but some SOP content is country-specific)
4. "What's your primary concern about the next audit?" → (Answer: can't produce evidence of who completed what — the Excel files don't have timestamps or certificates)
5. "What does your current certificate process look like?" → (Answer: manual email from HR coordinator with a PDF certificate — 3 days after completion)
6. "What HR system do you use?" → (Answer: SAP SuccessFactors)
7. "What languages are needed?" → (Answer: English, German, Hindi)

---

**Your Solution Design:**

**Phase 1 (Weeks 1-3): Audit-Critical Setup**

| Pain Point | Edrevel Solution | Business Impact |
|-----------|----------------|----------------|
| 3 Excel trackers across 3 countries | Single compliance dashboard — real-time, global | One source of truth for auditors |
| No certificates | Auto-issued digital certificates on completion | Immediate audit evidence |
| 55% completion | Role-based assignment + automated reminder sequences | Target 90%+ in 4 months |
| No country-specific content management | Multi-tenant setup with country admin roles | India admin manages India content; Germany manages German SOPs |
| Manual HR admin work | SAP SuccessFactors integration — users auto-provisioned | Eliminates manual user management |

**Phase 2 (Weeks 4-8): Content Modernization**

- Use AI Course Creator to rebuild top 7 GxP courses from source policy documents
- Add knowledge checks (pre/post) to prove learning, not just completion
- Multi-language: English base → translate to German and Hindi using AI translation + SME review
- Add expiry dates: certifications expire every 12 months → auto-trigger renewal assignments

**Phase 3 (Months 3-12): Advanced Analytics**

- Correlate training completion with audit findings (which departments that complete training have fewer deviations?)
- Skills assessment layer: are employees truly competent, or just completing?
- Manager dashboard: QA Manager sees their team's certification status in real-time

---

**4-Month Roadmap to Audit:**

```
Week 1-2:   Platform setup, SSO config with SAP SF, user import (2,800 employees)
Week 3:     Upload existing 7 GxP courses, configure role-based assignments
Week 4:     Configure reminder sequences (30, 14, 7, 3, 1 day before deadline)
            Enable auto-certificates, compliance dashboard live
Week 5:     Admin training for HR coordinators in all 3 countries
Week 6:     Communication campaign to all 2,800 employees (email + manager cascade)
            Go-live: employees start receiving assignments and completing courses
Week 7-12:  Weekly progress reviews, targeted outreach to non-starters
Week 13-16: Final push — daily automated reminders, personal follow-up on last 10%
Week 16:    Audit: pull single compliance report — all 2,800 employees, all 7 programs
            Certificates downloadable per employee, full timestamp evidence
```

**Success Metrics:**
- Completion rate: 55% → 92% by audit date
- Audit report generation time: 2 weeks (manual) → 15 minutes (1-click)
- HR coordinator time on compliance tracking: 60% → 5%
- Time to certificate issuance: 3 days → immediate on completion

**ROI:**
- 3 HR coordinators × 60% time on manual tracking = 1.8 FTE equivalent
- At ₹8 lakh/year per coordinator: ₹14.4 lakh in capacity savings annually
- Platform cost: assume ₹12 lakh/year
- Net saving Year 1: ₹2.4 lakh + risk reduction (FDA Warning Letter avoided — can cost $10M+ in delayed approvals)

---

## Case Study 2: Technology Company — Rapid Upskilling for AI Transformation

**The Scenario:**
CloudTech India has 1,200 software engineers, 300 product managers, and 500 business functions employees. The CEO has announced a company-wide "AI-first" transformation — every technical employee must be proficient in using AI tools within 6 months. Currently: no structured AI training, engineers are self-learning from random YouTube videos, there's no way to know who knows what, and 3 key clients have asked about their team's AI capabilities in recent RFPs. The L&D Head has a ₹50 lakh budget and no dedicated course development resources.

---

**Discovery Questions:**

1. "When the CEO says 'proficient in AI' — what does that look like specifically? Using AI tools in their daily work? Building AI features? Understanding AI strategy?" → (Answer: two levels — "AI User" for most employees and "AI Builder" for senior engineers)
2. "What AI tools does the company already use or want to use?" → (Answer: GitHub Copilot, ChatGPT Enterprise, an internal ML platform)
3. "Do you have a competency framework? Or do we need to build one?" → (Answer: have a skills framework for engineering roles but nothing for AI skills)
4. "What's the consequence of not achieving this in 6 months?" → (Answer: losing client RFPs, potential talent attrition as engineers want AI training, CEO reputation)
5. "Are engineers allocated learning time, or does this need to fit around project delivery?" → (Answer: 2 hours per week protected L&D time, confirmed by CTO)

---

**Solution Design:**

**The Problem in One Line:**
"You need 1,200 engineers certified on AI within 6 months, with no L&D team and no pre-built content — and you need proof it worked."

**Phase 1 — Foundation (Month 1):**
- Build AI Skills Taxonomy using SkillsPro: 12 AI skills across 3 levels (Awareness, Practitioner, Expert)
- Baseline assessment: 30-minute assessment for all 1,500 technical employees — establishes current AI skill level
- Two learning paths defined: "AI User" (15 skills, 12 hours) and "AI Builder" (25 skills, 30 hours)

**Phase 2 — Content (Month 1-2):**
- Use AI Course Creator to build 8 foundational AI courses from Edrevel's content + internal SME input:
  - "Understanding Generative AI for Non-Technical Roles" (4 hours)
  - "GitHub Copilot for Engineers" (3 hours, product-specific)
  - "Prompt Engineering Fundamentals" (2 hours)
  - "AI Ethics and Responsible Use at CloudTech" (1 hour)
  - "Building with ChatGPT Enterprise API" (6 hours, technical)
  - "MLOps Basics" (4 hours, for senior engineers)
- Supplement with curated external content (Coursera, YouTube) via LXP curation features
- Total estimated content build: 3-4 weeks using AI Course Creator vs. 6+ months traditional

**Phase 3 — Deploy & Track (Month 2-6):**
- All 1,500 technical employees see their personalized path on Day 1 of Month 2
- Weekly nudge: "You're 45 minutes away from your next skill badge"
- Manager dashboard: tech leads see their team's AI skill progress — part of quarterly team reviews
- Certification events: "AI User Certified" and "AI Builder Certified" digital badges — sharable on LinkedIn (social proof and employer branding)

**Month 6 Deliverable for CEO:**
- Skills heatmap: % of team at each AI proficiency level vs. Month 0 baseline
- Certification numbers: X employees AI User certified, Y employees AI Builder certified
- Usage data: 87% of certified employees actively using AI tools in their work (Copilot usage data from GitHub integration)
- Client RFP response: "We have [X] AI-certified engineers available for this project"

---

## Case Study 3: Large Retail Chain — Deskless Worker Training

**The Scenario:**
RetailMax has 850 stores across India with 45,000 employees — mostly store associates, floor managers, and warehouse staff. 80% of employees don't have a company email address and use their personal Android smartphones. They need: product knowledge training for new launches (new product lines launch every 8 weeks), customer service skills, safety protocols (OSHA equivalent — fire safety, equipment use), and seasonal onboarding (1,500 new hires every October/November before festive season). Current state: PowerPoint slides emailed to store managers who are supposed to "train their team" in a 30-minute meeting. Compliance rate: untrackable. Customer satisfaction scores are below target, and the Head of Retail Operations believes inconsistent training is a key factor.

---

**The Unique Challenge:**
This is a "deskless worker" scenario — these employees are fundamentally different from knowledge workers:
- No company email → can't use traditional LMS enrollment
- No laptop → must be 100% mobile
- Short attention spans on the job → microlearning, not 45-minute courses
- High turnover → training must be scalable and fast
- Many are first-time formal learners → UX must be dead simple

---

**Solution Design:**

**Access Method:**
- WhatsApp / SMS enrollment link → employee clicks → account created with mobile number (no email required)
- Or: store manager invites team via a QR code displayed in the break room
- Mobile app with offline capability: content downloads when on WiFi (store manager's WiFi) → accessible without data during shifts

**Content Strategy (mobile-first mandatory):**
- Maximum 5 minutes per module (learners on 15-minute tea breaks)
- Video-first (many may have low literacy comfort with text — video is universal)
- Hindi + regional languages (Tamil, Telugu, Bengali for different store clusters)
- Voice-over + subtitles: accessible for different literacy levels
- Short knowledge checks: 3 multiple-choice questions, retake allowed
- Visual cues > text: product images, step-by-step demonstrations

**Training Programs:**
1. **Product Launch Training**: New product = new 8-minute course. AI Course Creator builds it from product catalog data + marketing brief. Live to stores within 48 hours of product launch.
2. **Customer Service Skills**: 6 × 5-minute modules. Role-play videos showing "good" and "bad" interactions.
3. **Safety Protocols**: 4 mandatory modules, certificate required before floor access. Auto-assign to all new hires.
4. **Seasonal Onboarding (October)**: Automated path triggered when new employee account created. Must complete first 5 modules before receiving store access.

**Manager Dashboard (simple, mobile-friendly):**
- Store manager sees: who has completed, who hasn't, who needs a follow-up conversation
- "3 of your 12 team members haven't completed the Product Launch training yet" → one-tap reminder sent

**Metrics:**
- Before Edrevel: training completion untrackable, customer satisfaction at 62%
- 6-month target: 80% completion rate on product knowledge, customer satisfaction → 72%
- 12-month target: consistency in training across all 850 stores measurable for the first time

---

## How to Structure Your Response in the Room

**The 2-Minute Setup:**
"Before I propose a solution, I want to ask a few clarifying questions — the worst thing I can do is design the wrong solution confidently. [Ask 3-4 targeted questions.] OK, let me make sure I've understood the core problem: [one-sentence summary]. Does that match your understanding? Great. Here's how I'd approach this."

**The Transition Line:**
"What you've described sounds like three interconnected problems: [Problem 1], [Problem 2], and [Problem 3]. The good news is each one maps to a specific Edrevel capability — let me walk through them."

**The Confidence Closer:**
"My proposed success metrics would be [specific numbers] by [specific timeline]. And I'd build a 30-day review checkpoint so we can course-correct early if something isn't tracking right. Does this approach make sense to you?"
