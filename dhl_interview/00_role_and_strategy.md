# DHL QA Role — Overview & Interview Strategy

This role has **TWO tracks**. Know which one you're interviewing for (or be ready for both).

---

## The two tracks

### Track A — ETL / Data Warehouse Testing
You test **data pipelines**: data moving from source systems → staging → data warehouse → reports. You make sure the data is **correct, complete, and not corrupted** along the way.
- **Core skills:** SQL (heavy), Teradata, Data Warehousing concepts, data validation, BI tools (Power BI / QlikSense)
- **You'll be grilled on:** writing SQL queries, comparing source vs target data, finding data integrity issues

### Track B — Automation Testing
You test **applications** (web/UI) by writing automated scripts instead of clicking manually.
- **Core skills:** Selenium / UFT / UiPath, Java / VBScript, test frameworks, Agile, Jira/ALM
- **You'll be grilled on:** Selenium locators, frameworks (POM, data-driven), test case design, the test lifecycle

---

## What "QA / Testing" actually means (the mindset)

A tester's job is to **find problems before customers do.** You think: "How could this break? What's the edge case? Is the data right?"

For DHL (logistics): bad data or a broken app = wrong shipments, wrong billing, lost packages. So **data integrity and reliability** are everything. Tie your answers back to that.

---

## How the interview usually flows
1. **Intro** — "tell me about yourself" (have a 30-sec pitch)
2. **Concept questions** — SQL, DW, ETL testing OR Selenium, frameworks
3. **Hands-on / scenario** — "write a query to find duplicates" or "how would you automate this login page?"
4. **Process questions** — Agile, defect lifecycle, test case design
5. **Behavioral** — teamwork, a tough bug you found, stakeholder handling
6. **Your questions** — always have 2-3 ready

---

## Your file roadmap (read in order)
| File | Track | Topic |
|------|-------|-------|
| 01_sql_for_testing | A | SQL — the #1 skill, deep |
| 02_data_warehousing | A | Star schema, facts, dimensions, SCD |
| 03_teradata_essentials | A | Teradata specifics |
| 04_etl_testing_complete | A | ETL testing types, validation, integrity |
| 05_bi_tools | A | Power BI / QlikSense |
| 06_automation_selenium | B | Selenium + Java + frameworks |
| 07_uft_uipath | B | UFT (VBScript) + UiPath |
| 08_test_management | B | Test design, Agile, Jira/ALM, lifecycle |
| 09_interview_qa | A+B | 80+ Q&A both tracks |
| 10_cheatsheet | A+B | Night-before |

---

## 30-second intro (adapt with your real experience)
> "I'm a QA professional with hands-on experience in both ETL/data-warehouse testing and automation testing. On the data side, I write SQL to validate data moving through ETL pipelines — checking completeness, integrity, and transformations against business rules, with Teradata and data warehouse concepts. On the automation side, I build test frameworks using Selenium with Java, design test cases, and manage the test lifecycle in Agile using Jira. I focus on catching defects early and protecting data integrity — which in a logistics business like DHL directly impacts shipments and billing accuracy."

---

## The golden rule for this interview
**Be specific, not vague.** Don't say "I tested data." Say "I wrote a SQL query joining source and target on the primary key, then used MINUS/EXCEPT to find rows that didn't match, which caught a transformation bug where dates were off by timezone."

Specific = credible. Let's make every answer specific.
