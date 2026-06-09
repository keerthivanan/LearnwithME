# Master Cheat Sheet — Read This the Night Before

---

## Edrevel in 3 Sentences (Memorize This)
Edrevel is an AI-powered Learning Experience Platform (LXP) that helps organizations train, upskill, and certify their workforce at scale. It combines AI Course Creator (builds structured courses in hours from documents or prompts), SkillsPro Engine (maps skills and personalizes every employee's learning path), and Insights Dashboards (gives compliance officers, L&D managers, and executives real-time visibility into training outcomes). Unlike a traditional LMS that just tracks completions, Edrevel makes learning intelligent, personalized, and measurable.

---

## Your Role in 10 Words
**Bridge clients and technology. Translate pain into solutions. Deliver outcomes.**

---

## The 3 Products — One-Liners

| Product | One-Line Explanation | Business Value |
|---------|---------------------|----------------|
| **AI Course Creator** | Turns any document or topic into a full course in under 2 hours | Reduces course development from 6-8 weeks to hours |
| **SkillsPro Engine** | Netflix-style AI that personalizes every employee's learning path | Employees learn only what they need — faster time-to-competency |
| **Insights Dashboards** | Real-time visibility for compliance officers, L&D teams, and executives | Turns learning data into decisions and proves ROI |

---

## LXP vs LMS — Know This Cold

**LMS (Learning Management System):** Top-down. Admin assigns. Employees complete. Records tracked. Compliance-focused. Think Moodle, Cornerstone, SAP.
**LXP (Learning Experience Platform):** Bottom-up. Learner-driven. Personalized. Skill-building. Think Netflix. Think Degreed, EdCast.
**Edrevel:** Both — compliance tracking of an LMS PLUS the personalization engine of an LXP. One platform.

**The line to use:** "Edrevel doesn't force clients to choose between compliance management and learning engagement. It delivers both."

---

## Full Jargon Dictionary (Every Term Explained in Detail)

### L&D Fundamentals

**L&D (Learning & Development):**
The team or function within a company responsible for employee training, skill-building, and professional development. They design programs, manage the LMS/LXP, measure effectiveness, and partner with HR. Your primary buyer contact.

**Compliance Training:**
Mandatory training required by law, regulation, or company policy. Examples:
- HIPAA (healthcare privacy, USA)
- GDPR (data protection, EU)
- SOX (financial reporting controls, USA)
- GxP (Good Practice — pharmaceutical manufacturing and R&D)
- OSHA (workplace safety, USA)
- ISO standards (quality management)
- AML/KYC (anti-money laundering, financial services)
Characteristics: mandatory completion, often with deadlines, certificate required as proof, auditors check records. High-stakes for organizations — non-compliance = fines, license revocation, reputational damage.

**Skills Taxonomy:**
A structured, hierarchical map of all skills that exist within an organization. Organized by: skill domain (technical, leadership, functional), skill level (awareness/beginner/practitioner/expert), and relationships between skills (SQL → Data Analysis → Business Intelligence).
Why it matters: Without a skills taxonomy, the SkillsPro engine can't recommend anything — it's the database that powers personalization. Building one is often the first implementation task.

**Competency Framework:**
A set of defined knowledge, skills, and behaviors required for each role in the organization. Similar to skills taxonomy but more HR-oriented — used in performance reviews, promotions, hiring.
Example: "Senior Data Analyst" requires: Python (Practitioner), SQL (Expert), Data Visualization (Practitioner), Statistical Modeling (Awareness), Stakeholder Communication (Practitioner).

**Learner Engagement:**
A measure of how actively and voluntarily employees are using the learning platform. Not just "did they complete the mandatory course" — are they coming back? Are they exploring? Are they completing voluntary learning?
Metrics: active users %, session length, courses started voluntarily, return frequency.
Low engagement root causes: content irrelevant, content too long, no mobile support, no personalization, no manager championing, no time protected for learning.

**Time-to-Competency:**
How long it takes from a learning assignment to the learner demonstrating competency (passing the assessment or receiving certification). A key efficiency metric — faster time-to-competency means faster productivity for new hires, faster certification for compliance.

**Kirkpatrick Model:**
The gold standard 4-level framework for evaluating training effectiveness:
- Level 1 Reaction: Did learners like the training? (survey/rating)
- Level 2 Learning: Did they actually learn? (assessment score improvement)
- Level 3 Behavior: Are they applying it on the job? (manager observation, performance data)
- Level 4 Results: Did it improve business outcomes? (sales numbers, error rates, productivity)
Most platforms only capture L1-L2. Edrevel's Insights layer helps clients approach L3-L4.
Use this in the interview to show you understand that measuring training is not just about completion rates.

**Blended Learning:**
A combination of online self-paced learning AND in-person/live virtual instructor-led sessions. Example: 3-day classroom training + 8 weeks of Edrevel microlearning reinforcement. Best of both worlds: human interaction for complex skills + digital for scale and retention.

**Microlearning:**
Breaking content into very short learning units (2-10 minutes). Based on cognitive load theory — humans can only absorb so much at once. For corporate training: a 45-minute course becomes 9 × 5-minute modules. Better completion rates, better retention, works on mobile.

**Spaced Repetition:**
Learning strategy based on the Ebbinghaus Forgetting Curve — resurface content at calculated intervals (1 day, 1 week, 2 weeks, 1 month) to reinforce memory. Without it: people forget 70% within a week. With it: retention doubles or triples. Edrevel's scheduling engine can automate this.

**Social Learning:**
Learning from peers — discussions, knowledge sharing, commenting on content, mentorship. Based on the 70-20-10 model: 70% of learning happens on the job, 20% from others, 10% from formal training. LXPs support this with features like: peer content sharing, discussion forums, expert directories.

---

### Technical Standards

**SCORM (Sharable Content Object Reference Model):**
The old (2001) e-learning standard. Defines how a course "talks" to an LMS. Tracks: course launched, time spent, completion status (complete/incomplete), score, pass/fail.
Limitation: binary — only records "completed" or not. Can't track rich learning behaviors.
Still widely used: most existing corporate content is SCORM-packaged. Edrevel accepts SCORM 1.2 and SCORM 2004 imports.

**xAPI (Experience API / Tin Can API):**
The modern (2013) learning standard. Format: Actor → Verb → Object. Example: "Keerthi completed Module 3 with score 87%." Tracks ANY learning activity: videos, simulations, on-the-job tasks, mobile learning, external content.
Stores data in a Learning Record Store (LRS) — separate from the LMS, enabling cross-platform tracking.
Why it matters: more data → better analytics → better AI personalization recommendations.

**SSO (Single Sign-On):**
Authentication method where employees log in once (to their company's identity provider — Microsoft Azure AD, Okta, Google Workspace) and are automatically authenticated into all connected applications including Edrevel.
Why clients require it: IT security policy, eliminates password management, reduces login friction (= higher adoption).
Protocols: SAML 2.0 (older, widely supported), OIDC (modern OAuth2-based, used by Google/Okta).
Your line: "Edrevel supports SAML 2.0 and OIDC. We've integrated with Azure AD, Okta, Google Workspace, and Ping Identity. Typically 2-4 hours of client IT time for setup."

**HRIS Integration:**
Connecting Edrevel to the client's Human Resources Information System (Workday, SAP SuccessFactors, BambooHR, Oracle HCM) via API or SFTP file sync. Purpose: automatically provision new users, deactivate leavers, update role/department changes. Essential for large organizations — manually maintaining user accounts in Edrevel is not scalable.

**API (Application Programming Interface):**
A set of rules that allows two software systems to communicate and exchange data. When a client asks "can Edrevel integrate with our [system]?" — they're asking if there's an API. Edrevel exposes APIs for: user management, course data, completion records, analytics export. Clients' IT teams connect to these.

**LRS (Learning Record Store):**
A database specifically designed to store xAPI statements (learning activity records). Separate from the LMS — enables aggregating learning data from multiple platforms. Important for organizations with a complex learning ecosystem (LMS + LXP + external providers).

---

### Business Analysis Terms

**MoSCoW Prioritization:**
Framework for agreeing on requirement priorities with stakeholders. Must Have (non-negotiable), Should Have (important), Could Have (nice-to-have), Won't Have This Phase (out of scope now). Use this every time you're agreeing on project scope. "Won't have this phase" prevents scope creep and sets explicit expectations.

**User Story:**
Requirements written in the format: "As a [role], I want [feature] so that [business outcome]." Keeps focus on WHO benefits and WHY, not just what to build. Technology-agnostic. Example: "As a compliance officer, I want a one-click audit report so that I can produce evidence of completion in under 5 minutes instead of 2 days."

**Acceptance Criteria:**
The conditions that must be met for a user story to be considered "done." Written as: Given [context], When [action], Then [expected outcome]. Prevents misunderstandings between what the BA wrote and what engineering built.

**AS-IS / TO-BE:**
AS-IS = current state (how the client does things TODAY). TO-BE = future state (how it works after implementation). The Gap = what the project delivers. Always map both in discovery — it makes the value of the implementation visible and explicit.

**RACI Matrix:**
Responsibility assignment: Responsible (does the work), Accountable (approves it — only ONE), Consulted (provides input), Informed (notified). Prevents confusion about who owns what, especially in multi-stakeholder projects.

**Gap Analysis:**
Structured comparison of AS-IS vs TO-BE to identify what needs to change. Use in every implementation plan — it's the scope of work, made visible.

**Stakeholder Mapping:**
Identifying everyone who is affected by or has influence over a project. Plot on a 2×2 grid: Influence (high/low) × Interest (high/low). High influence, high interest = actively manage and engage. High influence, low interest = keep satisfied. Low influence, high interest = keep informed. Low influence, low interest = minimal effort.

**Change Management:**
The structured approach to helping people adopt new ways of working. The best platform in the world fails if users don't adopt it. Key elements: executive sponsorship (top-down mandate), manager activation (middle management cascade), communication campaign (WHY we're doing this), training (how to use it), feedback loops (what's not working).

---

### EdTech Industry Terms

**Deskless Workers:**
Employees who don't work at a desk — manufacturing workers, retail associates, delivery drivers, healthcare frontline staff. Special training requirements: must be mobile-first, offline-capable, short sessions (microlearning), simplified UX, possibly multi-language. 80% of the global workforce is deskless — huge market opportunity.

**Blended Learning vs. eLearning:**
eLearning = 100% digital/self-paced. Blended = mix of digital and in-person/live. Modern L&D best practice is blended — digital for knowledge transfer at scale, human interaction for complex skill application.

**Rapid Content Development:**
Creating training content quickly — typically using tools like AI Course Creator, Articulate Rise, Adobe Captivate. Important when: regulations change, new products launch, onboarding needs to scale fast. AI Course Creator is Edrevel's answer to rapid content development.

**NPS (Net Promoter Score) for Learning:**
Survey metric: "On a scale of 1-10, how likely are you to recommend this training to a colleague?" Score = % Promoters (9-10) - % Detractors (0-6). Target: above 30 for corporate training. Below 0 = learners actively resent it.

**ROI of Training:**
The business case calculation: (Value created by training - Cost of training) / Cost of training × 100%.
Hard to measure but essential to show to CFOs. Value created examples: faster onboarding (days to productivity), reduced errors (quality improvement), higher sales close rates (product knowledge), reduced attrition (engagement impact).

---

## The 5 Discovery Questions (Memorize Word-for-Word)
1. "Walk me through how this works today — from the moment an employee needs to complete training, all the way through to the evidence you have that they're compliant."
2. "What's the biggest frustration your team has with the current process?"
3. "How do you currently measure whether training is actually working?"
4. "Who else in the organization needs to see value from this platform — not just L&D, but compliance, HR, finance, operations?"
5. "If we implement Edrevel and everything goes perfectly — what does your week look like 12 months from now that it doesn't look like today?"

---

## Industry-Specific ROI Numbers (Use in Proposals)

| Metric | Industry Benchmark | With Edrevel (Target) |
|--------|-------------------|----------------------|
| Compliance completion rate | 55-65% | 88-95% |
| Course development time | 6-8 weeks per course | 1-3 days with AI Creator |
| Time to certificate issuance | 3-7 days (manual) | Instant (automated) |
| New hire onboarding time | 3-4 weeks to productivity | 1.5-2 weeks |
| HR admin time on compliance tracking | 40-60% of their time | Under 10% |
| Employee knowledge retention at 30 days | 20-30% (no reinforcement) | 70-80% (with spaced repetition) |
| Training ROI | Measured by very few | Visible in dashboard |

---

## Competitive Quick Reference

| Competitor | Their Strength | Your Response (Edrevel Advantage) |
|-----------|---------------|----------------------------------|
| Degreed | Curates external content (Coursera, LinkedIn) | Edrevel CREATES custom content — AI Course Creator. Degreed can't build your proprietary compliance course. |
| Cornerstone | Deep enterprise LMS, compliance | Edrevel is more modern UX, better LXP personalization, AI-native |
| LinkedIn Learning | Huge library, trusted brand | Only off-the-shelf content. Can't build custom courses. No compliance tracking. |
| Moodle | Free, open-source | Self-hosted = heavy IT burden. No AI features. No SkillsPro. |
| Workday Learning | Integrated with HR | Workday's learning is secondary to their HRIS. Edrevel's learning is primary. |

---

## Day-Before Preparation Checklist

**Research:**
- [ ] Read Edrevel's entire website — every product page, every case study
- [ ] Google "Edrevel AI news" — any recent announcements, funding rounds, new features
- [ ] Look up your interviewers on LinkedIn — note their background, how long they've been at Edrevel, what they did before
- [ ] Read 2-3 EdTech industry articles from the last 6 months (TechCrunch, EdSurge, Learning & Development Today)

**Prepare Your Stories:**
- [ ] "Tell me about a time you turned data into a decision" — know your specific story cold
- [ ] "Tell me about a time you influenced a difficult stakeholder" — have the details ready
- [ ] "Tell me about a time you worked under pressure" — outcome-focused
- [ ] "Why Edrevel? Why this role, why now?" — authentic, specific

**Logistics:**
- [ ] Confirm interview format, time zone, and link
- [ ] Test audio/video if virtual
- [ ] Have a notepad and pen — shows preparation
- [ ] Print or have open your copy of this cheat sheet for a final read 30 minutes before

**Mental:**
- [ ] Review your key numbers (past project metrics, client counts, outcomes)
- [ ] Reread this file one more time the morning of
- [ ] Sleep by 11pm the night before — tired and overprepped is worse than rested and normally prepped
