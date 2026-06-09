# Technical Concepts — Full Deep Dive

---

## SECTION 1: AI in Learning — Everything You Need to Know

### 1.1 How AI Powers Learning Platforms

**Large Language Models (LLMs) in Course Creation:**
- LLMs (like GPT-4, Claude) are trained on vast text data and can generate structured, coherent educational content
- When you give it: "Create a course on GDPR data retention rules for employees in a finance company"
- It outputs: learning objectives → module 1 content → knowledge check questions → module 2 → final assessment
- The AI follows **Bloom's Taxonomy** (a framework for structuring learning from recall → application → analysis)
- L&D professionals then review, edit, and approve — AI does the heavy lifting, humans ensure quality

**Why AI-generated courses are "good enough":**
- Corporate training doesn't need to be literary masterpieces — it needs to be clear, accurate, and complete
- AI handles structure and language; humans add company-specific examples, brand voice, edge cases
- For compliance content (HIPAA, GDPR, safety), accuracy is verified by SMEs — AI generates the structure

**Personalization AI (Recommendation Systems):**
- Collaborative filtering: "Employees who had this role + these skill gaps → were most successful after taking Course X"
- Content-based filtering: match learner's skills, interests, and role to course metadata
- Reinforcement learning: system improves recommendations over time based on completion, assessment scores, satisfaction ratings

**Skills Inference (cutting-edge):**
- Instead of manually tagging every employee's skills, AI infers skills from: job title, work history, completed courses, assessment performance, even communication patterns
- Makes skills mapping at scale possible — manually surveying 10,000 employees on their skills is impossible; AI inference takes minutes

---

### 1.2 Adaptive Learning — How It Actually Works

**Traditional learning**: Everyone watches the same 45-minute video, takes the same quiz. Pass 70% → complete.

**Adaptive learning**:
- Pre-assessment before the course → identifies what the learner already knows
- Content difficulty adjusts based on performance: doing well → skip basic content, advance to harder material; struggling → show simpler explanation, give more examples
- Item Response Theory (IRT): statistical model that estimates a learner's ability level based on which questions they answer correctly
- Microlearning loops: fail a question → get a 60-second explanation → reattempted

**Business value:**
- Learners who know 70% of the content aren't forced to sit through all of it → time saved
- Learners who are struggling get more support → better knowledge retention
- Assessment data is more accurate → organization knows who's actually competent, not just who clicked "complete"

---

### 1.3 The Forgetting Curve — Why Traditional Training Fails (and How Edrevel Solves It)

**Ebbinghaus Forgetting Curve (1885 — still completely relevant):**
- After 1 day: people forget ~50% of new information
- After 1 week: forget ~70%
- After 1 month: forget ~80%

**The corporate training problem:**
- 3-day classroom training = learn everything on Day 1, Day 2, Day 3 → forget 80% by Week 4
- This is WHY compliance incidents happen despite training completion rates showing 100%

**How LXPs like Edrevel address it:**
- **Spaced repetition**: resurface content at calculated intervals (1 day, 1 week, 2 weeks, 1 month) → each time the learner recalls, the memory is reinforced
- **Microlearning**: instead of one 3-hour course, deliver 18 × 10-minute modules over 3 weeks — better retention
- **Performance support**: "just-in-time" content available at the moment of need (pull, not push)
- **Assessment scheduling**: automated knowledge checks 30, 60, 90 days post-training to verify retention

---

## SECTION 2: L&D Domain Knowledge — Deep Dive

### 2.1 The Kirkpatrick Model — The Gold Standard for Training Evaluation

This was developed in 1959 and is STILL the framework every L&D professional uses to evaluate training. Know all 4 levels and how Edrevel addresses each.

**Level 1 — Reaction: "Did they like it?"**
- How: post-training survey (star ratings, NPS, open feedback)
- Edrevel feature: built-in course rating system, learner satisfaction surveys
- Limitation: just because they liked it doesn't mean they learned anything
- Interview use: "Most platforms only get to Level 1 — they tell you the star rating. That's not enough."

**Level 2 — Learning: "Did they actually learn something?"**
- How: pre/post assessments, knowledge checks, skills assessments
- Edrevel feature: integrated assessment engine, before/after score comparison
- The metric: "Score improved from 48% to 82% after completing the course" = learning happened
- Interview use: "Edrevel's assessment data tells you not just who completed, but who actually knows the material."

**Level 3 — Behavior: "Are they doing things differently on the job?"**
- How: manager observations, performance metrics, 90-day post-training check-ins
- Edrevel feature: manager dashboards that track skill application, 360 feedback integration
- The hard part: this requires data OUTSIDE the platform — performance reviews, incident reports, sales metrics
- Interview use: "Edrevel's Skills Dashboard starts to bridge Level 3 — tracking whether skill scores improve over time, not just whether someone sat through a course."

**Level 4 — Results: "Did the business get better?"**
- How: correlation between training completion and business KPIs (sales numbers, error rates, customer satisfaction, compliance incidents)
- Edrevel feature: ROI dashboard correlating learning activity to business outcomes
- Example: "After rolling out the new product knowledge training, close rates improved by 12% for certified reps vs. uncertified reps"
- Interview use: "This is the holy grail — and the reason executives care about L&D. Edrevel's insights layer is specifically designed to help clients build this story."

---

### 2.2 SCORM vs xAPI — The Technical Standards Explained

**Why you need to know this:**
When clients want to migrate existing courses into Edrevel OR export Edrevel data to another system, these standards govern how content and data is structured. IT teams will ask about this.

**SCORM (Sharable Content Object Reference Model):**
- Created in 2001 — the old standard
- What it tracks: course launched, time spent, completion status (complete/incomplete), pass/fail, score
- How it works: course runs inside an iframe, reports back to LMS via JavaScript API
- Limitation: binary data only ("completed" or "not completed") — no nuance
- Limitation 2: only tracks activity INSIDE the LMS — can't track learning that happens elsewhere
- Versions: SCORM 1.2 (most common), SCORM 2004 (better but less adopted)

**xAPI (Experience API, also called "Tin Can API"):**
- Created in 2013 — the modern standard
- What it tracks: ANYTHING — "Keerthi watched 3 minutes of video X", "practiced skill Y in a simulation", "read article Z on Chrome"
- Statement format: **Actor → Verb → Object** (e.g., "Keerthi completed module 2 with a score of 87%")
- Stores data in a **Learning Record Store (LRS)** — separate from the LMS
- Works ACROSS platforms: tracks learning in the LMS, on YouTube, in a simulator, on a mobile app
- **Why it matters for Edrevel**: richer data → better analytics → better SkillsPro recommendations

**When a client asks "are you SCORM compatible?"**
Say: "Yes, Edrevel accepts SCORM 1.2 and 2004 content for import. And we natively track using xAPI, which gives you far richer learning data than SCORM — including mobile learning, video engagement, and microlearning activity that SCORM can't capture."

---

### 2.3 SSO — Single Sign-On (the integration clients always ask about)

**What it is:**
- Employees sign in to ONE system (usually their company identity provider like Microsoft Azure AD, Okta, Google Workspace) → automatically authenticated into Edrevel
- They don't need a separate username/password for Edrevel

**Why clients care:**
- Employees hate managing multiple passwords → adoption drops if login is a hurdle
- IT security policy often REQUIRES SSO for SaaS tools
- Onboarding new employees is automated — when IT creates an Azure AD account, the user automatically appears in Edrevel

**Technical protocols involved:**
- SAML 2.0 (Security Assertion Markup Language) — older but widely used
- OAuth 2.0 / OpenID Connect (OIDC) — modern, used by Google, Okta

**What you say to IT teams:**
"Edrevel supports SSO via SAML 2.0 and OIDC. We've integrated with Azure AD, Okta, Google Workspace, and Ping Identity. Your IT team just needs to provide the metadata XML or OIDC endpoints and we handle the configuration on our side — typically 2-4 hours of IT time total."

---

### 2.4 HRIS Integration (Human Resources Information System)

**What it is:**
- Connecting Edrevel to the client's HR system (Workday, SAP SuccessFactors, BambooHR, Oracle HCM) so that:
  - New employees are automatically created in Edrevel when they're added to HR
  - Job role changes automatically update their learning assignments
  - Departing employees are automatically deactivated

**Why clients need it:**
- A company with 3,000 employees can't manually add/remove users in Edrevel — HR system must drive it
- Role-based learning assignment depends on accurate job title/department data from HR
- Without integration: L&D admin spends hours every week manually syncing systems → error-prone

**How you talk about it:**
"Edrevel connects to your HR system via API or SFTP file sync. We support Workday, SuccessFactors, and BambooHR out of the box. Custom HRIS integration is available for enterprise clients. The typical sync runs nightly — new joiners are provisioned automatically by 8am their first day."

---

## SECTION 3: Business Analysis Frameworks — In Depth

### 3.1 MoSCoW Prioritization — Use This Every Time

**What it is:**
A framework for prioritizing requirements with stakeholders so everyone agrees on what gets built first.

**M — Must Have**: Non-negotiable. Project fails without it. Audit deadline, legal requirement, critical business function.
**S — Should Have**: Important, high value, but not blocking launch. Can be in Phase 1B.
**C — Could Have**: Nice to have. If time and budget allow. Enhances the solution.
**W — Won't Have (this time)**: Explicitly out of scope for NOW. Not "never" — just not this phase.

**Why "W" is important:**
It prevents scope creep. When a stakeholder requests something, if you don't explicitly say "this is out of scope this phase," they'll assume it's coming. "Won't have this phase" is a documented decision they agreed to — not a surprise at go-live.

**Example — Edrevel implementation:**
| Requirement | Priority | Reasoning |
|-------------|---------|-----------|
| Compliance course assignment by role | Must | Audit in 6 weeks |
| Automated completion certificates | Must | Legal requirement |
| Manager visibility dashboard | Should | High adoption impact |
| Custom branding (logo, colors) | Should | Professional, high value |
| Social learning (peer sharing, comments) | Could | Engagement boost, not critical |
| Gamification (badges, leaderboards) | Could | Nice for culture |
| Integration with performance management system | Won't (Phase 2) | Complex, not needed for audit |
| Custom mobile app with offline access | Won't (Phase 3) | Roadmap item |

---

### 3.2 User Story Writing — The Correct Format

**Format:** As a **[role]**, I want **[feature]** so that **[business outcome]**.

**Why this format works:**
- Keeps the focus on WHO benefits, WHAT they need, and WHY — not HOW to build it
- Engineering team can read it and understand context
- Product team can prioritize based on business impact (the "so that" part)

**Examples for Edrevel:**

*Compliance Officer:*
> "As a compliance officer, I want to see a real-time dashboard showing which employees are overdue for GDPR training, so that I can send targeted reminders and ensure 95% completion before the audit deadline."

*L&D Manager:*
> "As an L&D manager, I want to use AI Course Creator to generate a first draft course from a policy PDF, so that I can reduce course development time from 6 weeks to 3 days."

*Employee/Learner:*
> "As a new joiner, I want to see a personalized onboarding learning path on my first day, so that I know exactly what I need to complete and in what order without asking HR."

*Line Manager:*
> "As a team manager, I want to receive a weekly digest showing my team members' training progress, so that I can have informed conversations with those who are falling behind."

**Acceptance Criteria** (add this for technical clarity):
After each user story, define what "done" looks like:
> "Given I am logged in as a compliance officer, When I click 'Compliance Dashboard', Then I see a list of employees sorted by days overdue, filterable by department, with a one-click 'Send Reminder' button."

---

### 3.3 Gap Analysis: AS-IS → TO-BE Mapping

**The most important tool in your discovery arsenal.**

**AS-IS (Current State):**
Ask the client: "Walk me through exactly how this works TODAY, step by step."
Then map it visually:

```
Employee joins company
        ↓
HR emails them a PDF of compliance courses
        ↓
Employee saves PDF to desktop (or ignores it)
        ↓
HR Admin manually follows up by email after 2 weeks
        ↓
Employee fills out a paper form confirming completion
        ↓
HR Admin adds to Excel spreadsheet
        ↓
Audit happens → scramble to find all evidence
        ↓
Print 200 emails as "proof of completion"
```

**TO-BE (Future State with Edrevel):**
```
Employee joins company
        ↓
Auto-provisioned in Edrevel via HRIS integration
        ↓
Receives personalized onboarding path Day 1
        ↓
Automated reminders at Day 3, 7, 14 before deadline
        ↓
Certificate auto-issued on completion
        ↓
Compliance Dashboard shows real-time status
        ↓
Audit happens → 1-click export of full compliance report
```

**The Gap** (what the project delivers):
- HRIS integration to auto-provision users
- Role-based course assignment rules
- Automated reminder configuration
- Certificate templates
- Compliance dashboard setup
- Admin training

---

### 3.4 RACI Matrix — Stakeholder Alignment Tool

**What it means:**
- **R** — Responsible: does the work
- **A** — Accountable: approves the work (only ONE person per row)
- **C** — Consulted: provides input before decision
- **I** — Informed: notified of outcomes

**Example for Edrevel Implementation Project:**

| Task | Client L&D Admin | Client IT | Compliance Officer | Edrevel BA | Edrevel Tech |
|------|-----------------|-----------|-------------------|------------|-------------|
| Define course list | R | I | C | A | I |
| Configure SSO | I | R | I | A | C |
| Upload content | R | I | I | A | I |
| Test platform | R | C | C | A | R |
| User communication | A | I | C | R | I |
| Go-live sign-off | I | C | A | R | I |
| Post-launch review | R | I | I | A | I |

---

### 3.5 Process Modeling — BPMN Basics (know the shapes)

**What BPMN is:** Business Process Model and Notation — a standard way to draw business processes visually.

**Key shapes:**
- **Circle** = Event (start / end / timer trigger)
- **Rectangle** = Activity / Task (what someone does)
- **Diamond** = Gateway / Decision (Yes/No branch)
- **Arrow** = Sequence flow (order of steps)
- **Swimlanes** = Who is responsible for each step

**When to use it:**
In discovery sessions when mapping a complex workflow — draw it on a whiteboard or Miro. Clients understand visual processes better than text lists.

---

## SECTION 4: Data Analytics for this Role

### 4.1 Key L&D Metrics — Full Definitions

**Completion Rate:**
- Formula: (Learners who completed / Learners assigned) × 100
- Industry average: 60-70% for compliance, 40-50% for voluntary learning
- What makes it drop: poor mobile experience, content too long, no reminders, low relevance
- Edrevel target: 85-95% with proper configuration

**Time-to-Competency:**
- Formula: Average number of days from learning assignment to skills assessment pass
- Why it matters: faster = new employees productive sooner, certifications achieved faster
- Benchmark: varies by role — a new sales rep reaching "certified" in 3 weeks vs 8 weeks has direct revenue impact

**Skill Coverage Score:**
- Formula: (Skills with passing assessment / Total required skills for role) × 100
- Tells you: what % of required skills an employee has actually demonstrated

**Knowledge Retention Rate:**
- Formula: (Score on 30-day reassessment / Score on immediate post-training assessment) × 100
- Why it matters: high completion + low retention = your training isn't working
- Target: 80%+ retention at 30 days (with spaced repetition, achievable; without it, ~20-30%)

**Training ROI:**
- Formula: (Financial benefit from training - Cost of training) / Cost of training × 100
- Example: Training reduced onboarding time by 5 days. Average employee costs ₹5,000/day in salary + overhead. 200 new hires per year. Saving: ₹50 lakhs. Platform cost: ₹8 lakhs. ROI: 525%.

**Net Promoter Score (NPS) for Learning:**
- "On a scale of 1-10, how likely are you to recommend this training to a colleague?"
- 9-10 = Promoters, 7-8 = Passives, 0-6 = Detractors
- NPS = % Promoters - % Detractors
- Target: above 30 is good for corporate training

---

### 4.2 How to Analyze Low Engagement — The Diagnostic Framework

When a client says "our platform engagement is low," here's the structured approach:

**Step 1: Quantify and Segment**
- Overall active users: X%
- Active users by department → which departments are high/low?
- Active users by role → is it specific roles?
- Active users by manager → is this a manager championship issue?
- Active users by device → mobile vs desktop

**Step 2: Content Analysis**
- Completion rate by content type (video vs. PDF vs. interactive)
- Average session length (too long = drop-off)
- Drop-off point in courses (where do people stop?)
- Age of content (last updated > 1 year ago? → stale)

**Step 3: Technical Analysis**
- Are push notifications enabled?
- Are there automated reminder sequences?
- Is SSO working properly? (login friction = abandonment)
- Mobile: is the content mobile-responsive?

**Step 4: Change Management Analysis**
- Did managers get trained on the platform?
- Was there a launch communication campaign?
- Is learning time protected in the culture (no time to learn = won't learn)?

**Step 5: Propose Targeted Fixes**
- For each root cause → one specific action → one metric to measure improvement → timeline

---

## SECTION 5: Tools Knowledge — What to Say

### Jira
**What it is:** Agile project management tool — track work items, sprints, bugs, feature requests
**Your usage context:**
- Managing implementation tasks as "tickets" with owners and due dates
- Writing user stories in Jira for product requests from clients
- Tracking platform configuration tasks across the BA team
- Creating epics (large features) and breaking them into stories and sub-tasks

**What to say in interview:**
> "I use Jira to manage implementation backlogs — each client configuration task is a ticket, assigned to the right team member, with acceptance criteria. For product requests from clients, I write them as user stories in Jira so the product team can evaluate them in sprint planning."

---

### Figma
**What it is:** Collaborative design tool — wireframes, prototypes, UI mockups
**Your usage context:**
- Reviewing UI designs with the product team to validate they meet client requirements
- Creating low-fidelity wireframes to show clients what a proposed dashboard or feature could look like before it's built
- Commenting on designs: "The compliance officer persona needs the 'export to PDF' button visible without scrolling"

**What to say in interview:**
> "In Figma, I primarily work as a reviewer and requirement validator — I take client requirements and check that the design team's mockups address them. I'll add comments like 'this filter should be pre-set to the current month by default, based on how the compliance team actually works.' It prevents expensive rework after development."

---

### Confluence
**What it is:** Collaborative wiki / documentation platform (made by Atlassian, integrates with Jira)
**Your usage context:**
- Solution Design Documents: formal write-up of the agreed solution before implementation
- Client onboarding playbooks
- Meeting notes and decision logs (crucial — "decision log" is your evidence when someone later says "we never agreed to that")
- Platform configuration guides for client admins

---

### Miro
**What it is:** Virtual whiteboard for remote collaboration
**Your usage context:**
- Running discovery workshops remotely — map AS-IS processes in real-time with stakeholders
- Stakeholder journey mapping: "show me every step a learner takes from being assigned a course to receiving a certificate"
- Mind mapping requirements
- Running remote retrospectives

---

### SQL (Basic Queries)
**Why a BA needs SQL:**
- When a client says "the dashboard shows 67% but our admin says it should be 72%," you need to query the raw data to find the discrepancy
- Ad-hoc data requests that don't fit the standard dashboard filters
- Validating that HRIS integration is populating user records correctly

**Basic queries you should know:**

```sql
-- How many users completed course X?
SELECT COUNT(*) FROM completions
WHERE course_id = 'GDPR_2024' AND status = 'completed';

-- Completion rate by department
SELECT department, 
       COUNT(*) as total,
       SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
       ROUND(100.0 * SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) / COUNT(*), 1) as rate
FROM user_enrollments
GROUP BY department
ORDER BY rate DESC;

-- Who hasn't completed GDPR training yet?
SELECT u.name, u.email, u.department
FROM users u
LEFT JOIN completions c ON u.id = c.user_id AND c.course_id = 'GDPR_2024'
WHERE c.user_id IS NULL;
```

**What to say if asked:**
> "I'm comfortable with basic to intermediate SQL — SELECT, GROUP BY, JOINs, WHERE clauses, aggregate functions. I use it for ad-hoc data validation and to cross-check dashboard numbers when a client questions the data. I'm not a data engineer, but I can pull and interpret the data I need without waiting for technical support."
