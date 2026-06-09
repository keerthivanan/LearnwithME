# 08 — Simple Things You Must Know for This Role

No jargon. No theory. Just the basics — explained plainly so they stick.

---

## 1. What Edrevel Actually Does (Explain It to Anyone)

Edrevel is a software platform that companies use to train their employees online.

It does three things:
1. **Creates training courses** — using AI, a company can upload a document and get a full course built in hours
2. **Personalizes learning** — each employee gets a different learning path based on their role and skill gaps (like Netflix recommendations, but for training)
3. **Tracks and reports** — managers and compliance officers can see who completed what, download certificates, and export reports for audits

**Simple version for an interview:** "Edrevel helps companies train their people faster, smarter, and with proof that it actually worked."

---

## 2. What Your Job Actually Is

You are the person who sits between the client (the company buying Edrevel) and the Edrevel team (product, engineering, implementation).

**Your 3 main jobs:**

| Job | What it means in plain English |
|-----|-------------------------------|
| Understand the client's problem | Ask questions. Listen. Map out how they do things today and what's broken. |
| Design the solution | Match their pain points to Edrevel's features. Build a plan. |
| Make sure it gets delivered | Write the requirements, track progress, manage expectations. |

**You are NOT a salesperson.** You are NOT a developer. You are the person who makes sure the right thing gets built for the right reason.

---

## 3. The 5 Most Common Client Problems You'll Hear

These are the real-world problems clients call Edrevel about. Know them cold.

**Problem 1: "We can't track who has done their training."**
→ They're using Excel or email. HR is drowning in admin work.
→ Edrevel solution: automated assignment, completion tracking, one dashboard.

**Problem 2: "Our employees don't engage with training."**
→ Courses are too long, too boring, not relevant to their actual job.
→ Edrevel solution: personalized paths (SkillsPro), microlearning, mobile access.

**Problem 3: "We have an audit coming and we're not prepared."**
→ No certificates, no timestamps, no proof of who completed what.
→ Edrevel solution: auto-certificates, compliance dashboard, one-click audit report.

**Problem 4: "It takes our L&D team months to build a course."**
→ They use vendors, PowerPoint, and manual processes.
→ Edrevel solution: AI Course Creator — 2 hours instead of 6 weeks.

**Problem 5: "We don't know if training is actually making a difference."**
→ They can see completion rates but not business impact.
→ Edrevel solution: Insights Dashboards — skill scores, assessment performance, engagement trends.

---

## 4. The 3 Types of People You'll Deal With at Every Client

**The L&D Manager / Training Manager:**
- Runs the day-to-day training programs
- Cares about: content quality, ease of use, learner engagement, less admin work
- Talk to them about: course creation, learning paths, analytics

**The Compliance Officer / HR Manager:**
- Responsible for regulatory training (GDPR, ISO, GxP, etc.)
- Cares about: completion rates, certificates, audit readiness, risk
- Talk to them about: compliance dashboard, auto-certificates, audit reports

**The IT Manager:**
- Approves all new software going into the company
- Cares about: security, data privacy, SSO, integrations with existing systems
- Talk to them about: SAML/SSO setup, HRIS integration, data residency, SOC 2 compliance

**Golden rule:** Know who you're talking to and adjust your language. What matters to compliance is not what matters to L&D. What matters to IT is not what matters to HR.

---

## 5. The Basic Flow of Every Implementation Project

Every Edrevel implementation follows roughly this sequence:

```
Step 1: Discovery
  → Meet the client. Understand their current process. Ask questions.
  → Document AS-IS (how things work today)

Step 2: Solution Design
  → Map their pain points to Edrevel features
  → Prioritize what to build first (MoSCoW)
  → Build a phased roadmap

Step 3: Setup
  → Platform configured for the client
  → SSO connected to their IT system
  → HR system integrated (users auto-populated)
  → Courses uploaded or created

Step 4: Go-Live
  → Users receive login and assignments
  → L&D team trained on admin panel
  → Communication sent to all employees

Step 5: Review
  → 30-day check-in: what's working, what's not
  → Engagement data reviewed
  → Adjustments made
```

As a BA, you own Steps 1 and 2 and support Steps 3-5.

---

## 6. Simple Frameworks You'll Use Every Week

### MoSCoW (for prioritizing requirements)
- **Must** = if this isn't in scope, the project fails
- **Should** = important but not blocking
- **Could** = nice to have
- **Won't** = out of scope for now (say this clearly to avoid scope creep)

### User Story (for writing requirements)
Format: "As a [person], I want [thing] so that [reason]."
Example: "As a compliance officer, I want automated reminders sent to employees who haven't completed their training so that I don't have to chase them manually."

### AS-IS → TO-BE (for showing value)
- AS-IS = how the client does it today (usually painful and manual)
- TO-BE = how it works after Edrevel (automated, tracked, simple)
- The gap between them = the value of the implementation

### STAR (for behavioral interview answers)
- **S**ituation — set the scene
- **T**ask — what was your responsibility
- **A**ction — what you specifically did
- **R**esult — the outcome (with numbers if possible)

---

## 7. Numbers and Facts to Have Ready

| Thing | Number/Fact |
|-------|-------------|
| Average course development time (traditional) | 6-8 weeks |
| With AI Course Creator | 1-3 days |
| Typical compliance completion rate without good tooling | 55-65% |
| Target compliance completion with Edrevel | 88-95% |
| How long humans retain information without reinforcement | 70% forgotten within 1 week (Ebbinghaus) |
| What LXP stands for | Learning Experience Platform |
| What LMS stands for | Learning Management System |
| SCORM stands for | Sharable Content Object Reference Model |
| xAPI tracks | Any learning activity (video, simulation, mobile, on-the-job) |
| Kirkpatrick Level 1 | Reaction — did they like the training? |
| Kirkpatrick Level 2 | Learning — did they actually learn? |
| Kirkpatrick Level 3 | Behavior — are they applying it at work? |
| Kirkpatrick Level 4 | Results — did it improve business outcomes? |

---

## 8. Simple Ways to Explain the AI Features

**AI Course Creator — simplest explanation:**
"You upload a document — a policy, a product manual, a regulation. The AI reads it, understands it, and builds a structured course with modules, quizzes, and a final assessment. Your team reviews it and publishes. What used to take 6 weeks takes 2 hours."

**SkillsPro Engine — simplest explanation:**
"Every employee has a job role and skill gaps. SkillsPro figures out what each person needs to learn and recommends it to them. Like how Netflix knows you like thrillers — SkillsPro knows that Priya in Finance needs to learn GDPR compliance and is 70% of the way through her certification path."

**Insights Dashboards — simplest explanation:**
"Instead of asking 'did people do the training?', you can ask 'are people getting better at their jobs?' The dashboard shows completion, scores, skill growth over time, which departments are behind, and whether training is working."

---

## 9. Things That Will Go Wrong — and How to Handle Them

**"Employees aren't logging in."**
→ Check if SSO is working. Check if they received the enrollment email. Check if their manager has been activated (manager behaviour drives team behaviour).

**"The course content is wrong / outdated."**
→ With AI Course Creator, rebuilding it from the updated source document takes hours. Make this a selling point: "This is why AI-generated content pays off — when regulations change, you update in a day, not 6 weeks."

**"The client wants a feature Edrevel doesn't have."**
→ First, understand the REAL need behind the request. Second, check if a workaround exists. Third, document it as a product feedback request. Fourth, be honest: "This isn't on the roadmap for Q1. Here's what we can do today."

**"The data looks bad — low completion, low scores."**
→ Don't panic. Pull the data. Find the root cause (device type? specific department? content length? no push notifications?). Come back with a diagnosis and a 30-day sprint plan.

**"The client is unhappy."**
→ First call should be listening, not defending. Say: "Thank you for telling me directly. Give me 48 hours to pull the data so I give you an accurate diagnosis rather than a guess." Then come back with a plan.

---

## 10. How to Sound Smart in the Interview Without Over-Preparing

Say these phrases naturally — they signal you understand the domain:

- "The real question isn't whether employees completed the training — it's whether they retained it and applied it."
- "Completion rate is a vanity metric. What matters is whether the skill gap actually closed."
- "Discovery is 80% of the job. If you design the wrong solution confidently, you waste everyone's time."
- "The best platform in the world fails without change management — you need manager buy-in before Day 1."
- "I always ask: what does success look like for you in 12 months? That's what I design toward."
- "I don't promise what I can't deliver. A realistic timeline with honest tradeoffs builds more trust than an optimistic pitch."
- "When a client asks for a specific feature, I first ask what problem they're trying to solve. 3 out of 5 times, there's a better solution already available."

---

## 11. What Makes a Good BA vs a Great BA

| Good BA | Great BA |
|---------|----------|
| Writes down what the client said | Hears what the client meant |
| Delivers what was asked | Flags when what was asked isn't what's needed |
| Sends requirements documents | Makes complex things easy to understand |
| Reports problems | Arrives with a diagnosis AND a solution |
| Manages one stakeholder | Manages the whole ecosystem (L&D, IT, Compliance, Finance) |
| Reactive | Proactive — spots issues before they become client calls |
| Thinks about features | Thinks about outcomes |

---

## 12. The Day-One Mindset

When you start this role:
- **First 2 weeks:** learn the product inside out. Use it as if you're a client. Try to build a course with AI Course Creator. Set up a learning path. Run a completion report. You can't help clients if you haven't lived it yourself.
- **First month:** shadow every client call. Don't talk — listen. Notice what questions clients ask repeatedly. Those become your expertise.
- **First 3 months:** own one small client engagement end-to-end. Be the person who knows their situation better than they do.

The fastest way to add value in a BA role: become the person no client wants to lose.
