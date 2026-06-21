# Night-Before Cheat Sheet — Read This Last

The highest-yield facts. Skim the morning of the interview.

---

## SQL — the must-knows
- **Duplicates:** `GROUP BY key HAVING COUNT(*) > 1`
- **Source vs target:** `SELECT ... FROM source MINUS SELECT ... FROM target` (both directions)
- **WHERE vs HAVING:** WHERE filters rows (before group), HAVING filters groups (after)
- **Orphans:** `LEFT JOIN ... WHERE right.key IS NULL`
- **Nulls:** `IS NULL` (never `= NULL`); `COALESCE(col,'x')`
- **Nth highest:** `DENSE_RANK() = N` or `ORDER BY ... OFFSET N-1 FETCH NEXT 1`
- **DELETE (rows, rollback) vs TRUNCATE (all rows, fast) vs DROP (table gone)**
- **UNION (dedup) vs UNION ALL (keep dups)**
- **Run order:** FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY

## Data Warehousing
- **OLTP** (run business, normalized) vs **OLAP** (analyze, denormalized, historical)
- **Star** = denormalized dims (fast); **Snowflake** = normalized dims (more joins)
- **Fact** = measures + FKs; **Dimension** = descriptive context
- **SCD Type 1** overwrite, **Type 2** new row + history (effective/end date + current flag), **Type 3** previous-value column
- **Surrogate key** = warehouse-generated integer key

## ETL Testing
- Start from the **Source-to-Target Mapping** document
- Core checks: **counts**, **two-way MINUS**, **transformation rules**, **nulls**, **duplicates**, **referential integrity**
- **Data integrity:** entity (PK), referential (FK/orphans), domain (valid values)

## Teradata
- **MPP** database; **Primary Index** hashes rows across **AMPs** → watch for **data skew**
- **MINUS** for set difference, **QUALIFY** for window-function filtering (dedup)
- Load utilities: **FastLoad** (empty tables), **MultiLoad** (populated), **TPump** (continuous), **BTEQ** (CLI)

## BI tools
- Validate every dashboard metric by **reconciling to warehouse SQL**
- Test filters, drill-downs, aggregations, refresh; watch double-counting
- **Power BI** (DAX, Power Query) · **QlikSense** (associative engine, set analysis)

---

## Selenium / Automation
- **Locators:** id > name > cssSelector > xpath (id best, xpath most flexible/slowest)
- **Relative XPath** `//tag[@attr='x']` over absolute
- **Explicit wait** (ExpectedConditions) — never `Thread.sleep`
- **POM** = page = class with locators + methods (maintainable)
- **Data-driven** = logic + data separate (Excel via Apache POI)
- **TestNG**: @Test, @BeforeMethod, @DataProvider; **Assert** (hard) vs soft assert
- Handle: dropdowns (`Select`), alerts (`switchTo().alert()`), frames (`switchTo().frame()`), windows (`getWindowHandles()`)

## UFT / UiPath
- **UFT:** commercial, web+desktop+enterprise, **VBScript**, **Object Repository**
- **UiPath:** **RPA** — Studio (design) / Robot (run) / Orchestrator (manage)
- Selenium tests apps; RPA automates business processes

## Process
- **STLC:** requirements → planning → design → env → execution → defects → closure
- **Severity** (impact) vs **Priority** (urgency) — independent; know both examples
- **Bug life cycle:** New → Assigned → Open → Fixed → Retest → Closed (Reopened/Rejected)
- **EP** (classes) + **BVA** (edges) — design techniques
- **Agile/Scrum:** sprints, PO/SM/Dev, planning/standup/review/retro
- **Jira** (issues, boards, JQL, Xray/Zephyr) · **ALM** (Requirements/Test Plan/Test Lab/Defects + traceability)
- **RTM** = requirement → test case → defect coverage

---

## The 3 sentences that win each section

**ETL/Data:**
> "I start from the source-to-target mapping, then validate completeness with counts and a two-way MINUS, accuracy per transformation rule, and integrity via foreign-key and orphan checks — automating the regression suite so every load is validated."

**Automation:**
> "I build a Page Object Model framework in Selenium + Java with TestNG, explicit waits for stability, data-driven tests via Apache POI, Extent Reports, and Jenkins CI — preferring id/CSS locators and never Thread.sleep."

**Process:**
> "I work in Scrum, write test cases from acceptance criteria, manage them and defects in Jira with traceability, design with equivalence partitioning and boundary value analysis, and track severity versus priority."

---

## Day-of checklist
- [ ] Reread this sheet + file 09 (Q&A)
- [ ] Practice the "tell me about yourself" pitch out loud 3×
- [ ] Have 2-3 specific stories ready (a bug you found, a data issue, a tough deadline)
- [ ] Prepare 2-3 questions to ask them (their stack, team, what success looks like)
- [ ] Keep a notepad; stay calm; be specific in every answer
- [ ] Connect answers to DHL: data integrity → correct shipments & billing

You've got this. 💪
