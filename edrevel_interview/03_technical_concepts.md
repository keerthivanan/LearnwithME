# Technical Concepts to Know for Edrevel Interview

## 1. AI in Learning — Core Concepts

### Personalized Learning Paths
- **What it is:** AI analyzes learner's role, skill gaps, past performance → recommends next best content
- **How Edrevel does it:** SkillsPro engine maps skills → identifies gaps → routes learner to right content
- **Business value:** Instead of "everyone takes the same 20 courses", each person sees their most relevant next step
- **Interview angle:** "This reduces training time by 40% because learners aren't wasting time on content they already know"

### AI Course Creator
- **What it is:** Inputs a topic/document/job role → AI generates structured course content (objectives, modules, quizzes, assessments)
- **Why it matters:** L&D teams traditionally take 4-8 weeks to build one course. AI does it in hours.
- **Business value:** Speed to deployment, especially for compliance where content needs frequent updates (regulations change)

### Adaptive Learning
- **What it is:** Course difficulty adjusts in real-time based on learner performance
- **Example:** If learner fails 3 quiz questions on GDPR data retention → system shows simpler explanation, more examples
- **Business value:** Better retention, faster certification

### Skills Taxonomy
- **What it is:** A structured map of all skills in an organization — categorized by domain, level, and relationship
- **Why BA needs to know this:** Clients often ask "how do we map our competency framework into Edrevel?" — this is a key implementation task

---

## 2. Learning & Development (L&D) Domain Knowledge

### Key Metrics L&D Teams Care About
| Metric | What it means | Why it matters |
|--------|--------------|----------------|
| **Completion Rate** | % of learners who finish a course | Low completion = engagement problem |
| **Time-to-Competency** | Days from start to skill certification | Impacts onboarding speed and productivity |
| **Skill Gap Score** | Delta between required vs current skills | Drives learning prioritization |
| **Training ROI** | Business outcome / training cost | What CEOs and CFOs ask about |
| **Learner Satisfaction (NPS)** | Would you recommend this to a colleague? | Proxy for content quality |

### Kirkpatrick Model (L&D Evaluation Framework — impress them with this!)
Level 1 — **Reaction**: Did learners like the training?
Level 2 — **Learning**: Did they actually learn something? (pre/post assessment)
Level 3 — **Behavior**: Are they applying it on the job?
Level 4 — **Results**: Did it move business metrics?

Most platforms only do L1 and L2. Edrevel's insights dashboard helps reach L3-L4.

### SCORM / xAPI — E-learning Standards
- **SCORM**: older standard — course tells LMS "completed" or "not completed"
- **xAPI** (Tin Can): modern — tracks ANY learning activity ("Keerthi watched 3 minutes of video X", "practiced skill Y in simulator")
- **Why this matters:** xAPI feeds richer data into analytics dashboards

---

## 3. Data Analytics Concepts for this Role

### What the Insights Dashboard Likely Shows
- Active learners vs. inactive (engagement segmentation)
- Completion rates by department / role / manager
- Skill coverage heatmap (which skills are strong vs. gaps)
- Time spent learning (hours per user)
- Correlation: learning activity → performance outcomes

### How to Talk About Data Analysis in Interview
> "I look at data through the lens of: **what action does this insight drive?**
> A 23% completion rate is just a number. But if I slice it by device, by manager, by content format — now I can tell the client EXACTLY what to fix.
> The goal is always to turn data into a decision."

---

## 4. Business Analysis Frameworks

### Requirements Gathering: MoSCoW Prioritization
- **M**ust have: non-negotiable
- **S**hould have: important but not critical
- **C**ould have: nice to have
- **W**on't have (now): out of scope

### Process Documentation: User Stories
Format: **As a [user], I want [feature] so that [outcome]**
Example: "As a compliance officer, I want to see which employees are overdue for GDPR training so that I can send targeted reminders before the audit."

### Gap Analysis: AS-IS → TO-BE
- **AS-IS**: how the client does it TODAY (manual, email-based, spreadsheet-tracked)
- **TO-BE**: how it works after Edrevel (automated, personalized, real-time dashboard)
- **Gap**: what needs to change (data migration, change management, training their admins)

### RACI Matrix (for stakeholder mapping)
| | Client L&D Admin | IT Team | Executive Sponsor | Edrevel BA |
|-|-----------------|---------|------------------|------------|
| Configure platform | R | C | I | A |
| Migrate content | C | R | I | A |
| Sign-off requirements | A | I | R | C |

---

## 5. Tools to Know

| Tool | Purpose | Your angle |
|------|---------|-----------|
| **Jira** | Agile project management, backlog, bug tracking | "I've used Jira to manage requirement backlogs, write user stories, and track implementation status" |
| **Figma** | UI/UX design and prototyping | "I've worked with design teams in Figma to review wireframes and validate against client requirements" |
| **Confluence** | Documentation wiki | "I document solution designs, meeting notes, and client specifications in Confluence" |
| **Miro / FigJam** | Virtual whiteboard for workshops | "I facilitate discovery workshops using Miro — stakeholder journey mapping, process flows" |
| **Excel / Google Sheets** | Data analysis | "Quick pivot tables, cohort analysis, building ROI models for client proposals" |
| **SQL (basic)** | Query databases for ad-hoc analysis | "I can write basic SQL to pull engagement data and validate what the dashboard is showing" |
| **Power BI / Tableau** | Data visualization | "I build simple dashboards to show clients their learning analytics in a compelling way" |
