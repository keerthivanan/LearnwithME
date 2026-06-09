# Edrevel AI — Deep Company & Role Analysis

---

## What is Edrevel?

Edrevel is an **AI-powered Learning Experience Platform (LXP)** that helps organizations
train, upskill, and certify their workforce at scale. It targets three core markets:

1. **Corporate L&D (Learning & Development)** — onboarding, leadership development, upskilling
2. **Compliance Training** — mandatory regulatory training (HIPAA, GDPR, SOX, EHS, ISO)
3. **Workforce Development** — large-scale reskilling programs (e.g., bank training 10,000 employees on AI tools)

The core value proposition: **Replace slow, generic, manually managed training with intelligent, personalized, data-driven learning.**

---

## Edrevel's Core Products — Know These in Detail

### 1. AI Course Creator
**What it is:**
- An AI tool that automatically generates complete, structured eLearning courses
- Input: a topic, a document (PDF/Word), a job role description, or a URL
- Output: full course with learning objectives, modules, text content, quizzes, assessments, and completion criteria

**How it works technically:**
- Uses large language models (LLMs like GPT/Claude) to understand context and generate structured pedagogically-sound content
- Follows instructional design principles: bloom's taxonomy, scaffolding, knowledge checks
- Supports SCORM/xAPI export for compatibility with other LMS systems

**Why it's a game-changer:**
- Traditional L&D content creation: **6–8 weeks** per course (needs instructional designer, SME review, graphic designer, QA)
- AI Course Creator: **under 2 hours** for a first draft — then L&D team customizes, reviews, approves
- Especially valuable for **compliance** where content needs updates every time regulations change
- Enables subject matter experts (SMEs) who aren't L&D professionals to create content themselves

**Interview talking point:**
> "If a pharma company needs to update their GxP training because FDA regulations changed, their L&D team can push a course update same-day instead of waiting 6 weeks. That's audit confidence."

---

### 2. SkillsPro Engine
**What it is:**
- AI engine that maps every employee's current skills, identifies gaps vs. required skills, and recommends a personalized learning path

**How it works:**
- Step 1: Build a **Skills Taxonomy** — structured map of all skills in the organization (can use AI to generate from job descriptions)
- Step 2: For each employee: assess current skill level (via assessments, manager ratings, or AI inference from job title + experience)
- Step 3: Compare to **required skills** for their current role AND their aspirational role
- Step 4: Generate a **personalized learning path** — only what they need, in the right order
- Step 5: Track progress as they complete content — skill levels update dynamically

**Why it's different from just "recommending courses":**
- Most LMS systems assign the same 20 courses to everyone in a department
- SkillsPro says: "Priya already has SQL skills (level 3), skip those. She needs Power BI and stakeholder communication. Here's her path."
- Result: less time wasted on content they already know, faster time-to-competency

**Interview talking point:**
> "Think of Netflix recommendations but for career growth — the platform surfaces the exact next skill each employee needs based on their role, goals, and current capability level."

---

### 3. Learning Insights Dashboards
**What it is:**
- Analytics and reporting layer that gives HR leaders, L&D teams, compliance officers, and executives real-time visibility into learning activity and outcomes

**Key dashboards:**
- **Engagement Dashboard**: active learners, courses started/completed, time-on-platform, dropout points
- **Compliance Dashboard**: who is compliant/non-compliant by department, days until deadline, overdue employees
- **Skills Dashboard**: company-wide skills heatmap, gap areas by team/role/level
- **ROI Dashboard**: training hours invested vs. performance outcomes, cost savings
- **Manager Dashboard**: each manager sees their team's learning activity — who needs support

**Why this matters to different stakeholders:**
| Stakeholder | What they need to see | How the dashboard helps |
|------------|----------------------|------------------------|
| Compliance Officer | Who passed/failed mandatory training | Real-time compliance status, 1-click audit report |
| L&D Manager | What content works, what's ignored | Engagement rates, dropout analysis, content effectiveness |
| HR Director | Is training helping retention? | Skill progression over time, promotion correlation |
| CEO/CFO | Is training worth the investment? | ROI metrics, productivity impact |
| Line Manager | Which of my people need help? | Team-level completion and skill status |

---

## LXP vs LMS — The Most Important Distinction to Know

This question WILL come up. Know it cold.

### LMS (Learning Management System) — The Old Way
- Built in the **early 2000s** for compliance-driven, administrator-controlled training
- **Top-down model**: admin assigns courses → employees complete → system records completion
- **Admin's perspective**: control, compliance tracking, reporting
- **Learner's perspective**: mandatory, boring, "tick the box", irrelevant to daily work
- Examples: Moodle, Cornerstone OnDemand, SAP SuccessFactors Learning, TalentLMS
- Main use case: **compliance** (HIPAA, GDPR, safety training, certifications)
- Core question it answers: "Did this person complete this mandatory course?"

### LXP (Learning Experience Platform) — The Modern Way
- Built in the **2015-2020s** for engagement-driven, learner-centric development
- **Bottom-up model**: learner discovers, chooses, and controls their learning journey
- **Learner's perspective**: relevant, personalized, continuous, like Netflix/YouTube
- **Admin's perspective**: insight-driven, outcome-focused, culture of learning
- Examples: Degreed, EdCast, Edrevel, Percipio by Skillsoft, LinkedIn Learning
- Main use case: **upskilling, career development, knowledge sharing**
- Core question it answers: "Is this person growing the skills the business needs?"

### Edrevel's Unique Position: LXP + LMS Combined
| Feature | LMS | LXP | Edrevel |
|---------|-----|-----|---------|
| Compliance tracking | ✅ | ❌ | ✅ |
| Personalized learning paths | ❌ | ✅ | ✅ |
| AI content generation | ❌ | Sometimes | ✅ |
| Skills gap analysis | ❌ | ✅ | ✅ |
| Mandatory course assignment | ✅ | ❌ | ✅ |
| ROI analytics | Basic | ✅ | ✅ |
| Mobile-first | Sometimes | ✅ | ✅ |
| Social/collaborative learning | ❌ | ✅ | ✅ |

> **Interview line**: "Edrevel doesn't force a choice between compliance management and learning engagement — it delivers both in a single platform. That's why it wins over clients who've outgrown their old LMS but still can't abandon compliance requirements."

---

## Edrevel's Competitors — Be Aware

| Competitor | Strength | Edrevel's Edge |
|-----------|---------|----------------|
| **Degreed** | Curates external content (Coursera, YouTube, etc.) | Stronger AI content CREATION, not just curation |
| **EdCast** | Good knowledge sharing, enterprise | Edrevel's AI Course Creator is more advanced |
| **Cornerstone** | Deep LMS, enterprise compliance | Edrevel is more modern UX, better LXP features |
| **LinkedIn Learning** | Huge library, familiar brand | Edrevel lets you build custom content; LinkedIn is off-the-shelf only |
| **Workday Learning** | Integrated with HR system | Edrevel's AI features are more advanced |

> **Key message**: "Edrevel's differentiation is the AI layer — specifically the Course Creator and SkillsPro engine. Competitors curate or manage learning; Edrevel creates and personalizes it."

---

## The Role: Technical / Business Analyst — Broken Down

### What "Technical" Means in This Role
- NOT a software engineer or developer
- You need to understand HOW the platform works well enough to:
  - Configure it for clients (set up courses, assign users, build learning paths)
  - Validate that data coming into dashboards is correct
  - Troubleshoot when something isn't working as expected
  - Speak credibly to a client's IT team about integrations (SSO, HR system sync, API)

### What "Business Analyst" Means in This Role
- Requirements gathering: run discovery workshops, write solution specs
- Process mapping: document AS-IS → TO-BE learning workflows
- Stakeholder management: navigate client politics (L&D team vs IT vs Compliance)
- Solution design: which Edrevel modules solve which business problems
- Success measurement: define KPIs upfront, measure them post-implementation

### Three Roles You'll Play (depending on the day)
1. **Pre-sales support**: join sales calls, run discovery, help build proposals that win deals
2. **Implementation lead**: onboard new clients, configure platform, train their admins
3. **Customer success**: monitor usage data, identify at-risk accounts, drive adoption and expansion

### Key Relationships
| Who | What they need from you |
|-----|------------------------|
| **Sales team** | Discovery templates, solution briefs, demo configurations, ROI models |
| **Product team** | Client feedback, feature gaps, prioritized wish lists from real users |
| **Engineering team** | Clear technical specs when custom work is needed, integration requirements |
| **Clients** | Trust, expertise, solutions, responsiveness — you're their person inside Edrevel |
| **Client's IT team** | Technical specs for SSO (Single Sign-On), API integrations, data privacy |
| **Client's L&D team** | Platform training, content strategy, analytics interpretation |

---

## Industries You'll Serve and Their Specific Needs

### Healthcare / Pharma
- **Compliance**: GxP (Good Practice), HIPAA, Joint Commission accreditation
- **High stakes**: non-compliance = regulatory fines, license revocations, patient safety
- **Key need**: automated tracking, certificate management, audit-ready reports at any time
- **Sensitivity**: protected health information (PHI) — data privacy is non-negotiable

### Financial Services / Banking
- **Compliance**: SOX, KYC/AML, FINRA, FCA regulations
- **Scale**: thousands of employees, frequent regulation changes
- **Key need**: rapid course updates when rules change, role-based assignment, evidence of training for audits
- **Unique**: needs multi-language support for global operations

### Retail / Manufacturing / Logistics
- **Workforce**: large, often deskless (no desk, works on shop floor or delivery routes)
- **Key need**: mobile-first, offline access, short microlearning (no one watches 45-minute videos on a forklift)
- **Compliance**: safety training (OSHA), equipment operation, food safety

### Technology Companies
- **Focus**: upskilling > compliance (they care about skills development)
- **Key need**: AI/ML skills, cloud certifications, product knowledge
- **Culture**: engineers don't want to sit through boring courses — content must be engaging, technical, practical

---

## What "Digital Transformation of L&D" Means (for conversation)

**The Old World (before Edrevel-style platforms):**
- HR builds a PDF course in PowerPoint
- Emails it to all 500 employees
- Tracks completion in an Excel spreadsheet
- Sends reminders manually
- Issues PDF certificates via email
- At audit time: panic, manually extract data, hope Excel is right
- No idea if anyone actually learned anything vs. just clicked "complete"

**The New World (with Edrevel):**
- AI generates a course from a policy document in 90 minutes
- Automatically assigned to relevant employees by role
- Employees get personalized reminders on their preferred channel
- Completion tracked in real-time, certificates auto-issued
- Compliance officer has a live dashboard — audit report in 1 click
- Analytics show which modules had high dropout → content improved
- Skills data shows impact on employee performance over time

> **This is the story you tell clients — frame EVERYTHING in terms of this transformation.**
