# Test Management — Design, Agile, Jira/ALM, Lifecycle

Process knowledge — asked in BOTH tracks. Know this cold.

---

## 1. STLC — Software Testing Life Cycle (the backbone)
```
1. Requirement Analysis   — understand what to test
2. Test Planning          — strategy, scope, resources, schedule (test plan doc)
3. Test Case Design       — write test cases + test data
4. Test Environment Setup — prepare the test environment
5. Test Execution         — run tests, log results
6. Defect Reporting       — raise + track bugs
7. Test Closure           — reports, metrics, lessons learned
```
> "Test Plan" is the document; STLC is the process.

---

## 2. Test Case Design — write good ones

A **test case** has: ID, Title, Preconditions, Test Steps, Test Data, **Expected Result**, Actual Result, Status, Priority.

**Example:**
| Field | Value |
|-------|-------|
| ID | TC_LOGIN_01 |
| Title | Valid login |
| Steps | 1. Open login page 2. Enter valid user/pass 3. Click Login |
| Expected | User lands on Dashboard |

### Test design TECHNIQUES (know these — common question)
- **Equivalence Partitioning:** group inputs into classes that behave the same; test one per class (e.g., age 0-17 invalid, 18-60 valid → test one of each)
- **Boundary Value Analysis (BVA):** test the edges (min, min-1, max, max+1) — bugs hide at boundaries (if valid is 18-60, test 17,18,60,61)
- **Decision Table:** combinations of conditions → actions
- **State Transition:** test states and transitions (order: placed→shipped→delivered)
- **Error Guessing:** experience-based, guess where bugs hide

> **BVA + Equivalence Partitioning** are the two you MUST be able to explain with an example.

---

## 3. Defect / Bug Life Cycle (guaranteed question)
```
New → Assigned → Open → Fixed → Retest → Closed
                          ↘ (if still failing) Reopened
                          ↘ Rejected / Duplicate / Deferred
```
**A good defect report has:** ID, Summary, Steps to reproduce, Expected vs Actual, Severity, Priority, Environment, Screenshots/logs.

### Severity vs Priority (classic question)
- **Severity** = how bad the impact is (technical). Critical / Major / Minor / Cosmetic.
- **Priority** = how soon to fix (business). High / Medium / Low.
- They're independent! Examples:
  - **High severity, low priority:** app crashes on a screen no one uses
  - **Low severity, high priority:** company logo misspelled on the homepage (looks bad, must fix now)

---

## 4. Agile / Scrum (the JD says Agile)
**Agile** = iterative development in short cycles (**sprints**, usually 2 weeks), with continuous feedback. **Scrum** is the most common Agile framework.

**Scrum roles:** Product Owner (owns the backlog/priorities), Scrum Master (facilitates, removes blockers), Development Team (incl. testers).

**Scrum ceremonies:**
- **Sprint Planning** — pick work for the sprint
- **Daily Standup** — 15 min: what I did / will do / blockers
- **Sprint Review** — demo to stakeholders
- **Sprint Retrospective** — what went well / improve

**Artifacts:** Product Backlog, Sprint Backlog, Increment.

**User Story:** "As a [role], I want [feature], so that [benefit]." With **acceptance criteria** (the conditions that make it "done"). Testers write test cases from acceptance criteria.

**Definition of Done (DoD):** the checklist a story must meet (coded, tested, reviewed, no critical bugs).

### The tester's role in Agile
- Involved from the start (shift-left testing)
- Write test cases from user stories/acceptance criteria
- Test within the sprint, automate regression
- Participate in all ceremonies

---

## 5. Jira & ALM (test management tools)

### Jira (Atlassian)
The most common Agile project + issue tracking tool.
- **Issues:** Epic → Story → Task → Sub-task; and **Bug** type
- **Boards:** Scrum board / Kanban board (To Do → In Progress → Done)
- **Backlog:** prioritized list of stories
- **Sprints:** create, start, close sprints
- **Workflow:** the status path an issue follows (e.g., Open → In Progress → In Review → Done)
- **JQL (Jira Query Language):** search issues (`project = DHL AND status = Open AND assignee = me`)
- **For testing:** log bugs, link them to stories, track in the sprint; with **Xray / Zephyr** plugins for test case management inside Jira

### ALM (Application Lifecycle Management — OpenText/Micro Focus, formerly HP QC)
An enterprise test management tool. Modules:
- **Requirements** — store requirements
- **Test Plan** — write/store test cases
- **Test Lab** — execute test sets, record pass/fail
- **Defects** — log and track bugs
- **Dashboard/Reports** — coverage and status
- **Traceability:** link Requirements → Test Cases → Defects (so you can prove coverage)

> **Requirement Traceability Matrix (RTM):** maps each requirement to its test cases (and defects), proving every requirement is tested. Common interview term.

---

## 6. Types of testing (broad — know the map)
- **Functional:** does it do what it should (smoke, sanity, regression, integration, system, UAT)
- **Non-functional:** performance, load, security, usability, compatibility
- **Smoke vs Sanity:** Smoke = quick check the build is stable enough to test; Sanity = focused check on specific fixed functionality
- **Regression:** re-test existing features after a change (prime automation candidate)
- **Retesting:** re-test the specific fixed bug
- **Integration:** test modules working together
- **UAT (User Acceptance Testing):** business users confirm it meets needs

---

## 7. Metrics you might mention
- Test coverage %, pass/fail rate, defect density, defect leakage (bugs found in prod vs testing), defect removal efficiency

---

## The process sentence that sounds senior
> "I work in Scrum — writing test cases from user-story acceptance criteria, managing them and logging defects in Jira (with traceability), and automating the regression suite. I design tests using equivalence partitioning and boundary value analysis, track severity vs priority carefully, and maintain a requirement traceability matrix so every requirement has coverage."
