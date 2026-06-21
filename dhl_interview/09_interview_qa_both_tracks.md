# Interview Q&A — Both Tracks (80+ Questions)

Practice answers out loud. Grouped by topic.

---

## 🗄️ SQL (15)

**1. Difference between WHERE and HAVING?**
> WHERE filters rows before grouping; HAVING filters groups after GROUP BY. HAVING is used with aggregate functions.

**2. Find duplicate records.**
> `SELECT col, COUNT(*) FROM t GROUP BY col HAVING COUNT(*) > 1;`

**3. Difference between DELETE, TRUNCATE, DROP?**
> DELETE removes rows (can filter with WHERE, can rollback, logs each row). TRUNCATE removes ALL rows fast (no WHERE, minimal logging, resets identity). DROP removes the whole table structure.

**4. Types of JOINs?**
> INNER (matches in both), LEFT/RIGHT OUTER (all of one side + matches), FULL OUTER (everything), CROSS (Cartesian product).

**5. UNION vs UNION ALL?**
> UNION combines and removes duplicates (slower); UNION ALL keeps duplicates (faster).

**6. Find Nth highest salary.**
> `SELECT DISTINCT salary FROM emp ORDER BY salary DESC OFFSET N-1 ROWS FETCH NEXT 1 ROW ONLY;` or use DENSE_RANK() = N.

**7. Primary key vs Unique key?**
> Both enforce uniqueness. PK: one per table, not null. Unique: multiple allowed, can have one null.

**8. What is a foreign key?**
> A column referencing the primary key of another table — enforces referential integrity.

**9. Difference between MINUS/EXCEPT and NOT IN?**
> MINUS/EXCEPT returns rows in the first query not in the second (whole row). NOT IN checks a single column membership (careful: NOT IN with NULLs returns nothing).

**10. ROW_NUMBER vs RANK vs DENSE_RANK?**
> ROW_NUMBER: unique sequential. RANK: ties share rank, gaps after. DENSE_RANK: ties share rank, no gaps.

**11. What is an index? Pros/cons?**
> A structure that speeds up reads (like a book index). Cons: slows writes, uses space.

**12. Find records in source not in target.**
> `SELECT * FROM source MINUS SELECT * FROM target;`

**13. How to handle NULLs?**
> `IS NULL` / `IS NOT NULL`; `COALESCE`/`NVL` to substitute defaults. Never use `= NULL`.

**14. What is a subquery / correlated subquery?**
> A query inside another. Correlated = inner query references the outer query's row (runs per row).

**15. WHERE vs ON in a join?**
> ON defines the join condition; WHERE filters the result. With OUTER joins they behave differently — a filter in WHERE can turn a LEFT JOIN into an INNER.

---

## 🏢 Data Warehousing (12)

**16. OLTP vs OLAP?** → Transactional (run business, normalized) vs Analytical (warehouse, denormalized, historical).
**17. What is a data warehouse?** → Integrated, subject-oriented, time-variant, non-volatile repository for analysis.
**18. Star vs Snowflake schema?** → Star: denormalized dimensions (fewer joins, faster). Snowflake: normalized dimensions (less redundancy, more joins).
**19. Fact vs Dimension table?** → Fact = numeric measures + FKs (center). Dimension = descriptive context (who/what/when/where).
**20. What is grain?** → The level of detail of a fact row (one row per shipment).
**21. Additive/semi-additive/non-additive measures?** → Sum across all dims / some dims / none (ratios).
**22. Surrogate vs natural key?** → Surrogate = warehouse-generated integer (stable, handles history). Natural = source business key.
**23. What is SCD? Types?** → Slowly Changing Dimension. Type 1 overwrite, Type 2 new row + history, Type 3 previous-value column.
**24. How do you test SCD Type 2?** → Change a source attribute, confirm old row closed (end_date, flag=N) and new row added (flag=Y), history preserved.
**25. Full vs incremental load?** → Load all vs only new/changed rows (CDC).
**26. What is a staging area?** → Temp landing zone for raw data before transformation.
**27. What is a conformed dimension?** → A dimension shared/consistent across multiple fact tables.

---

## 🔄 ETL Testing (12)

**28. What is ETL testing?** → Validating data is correctly Extracted, Transformed, Loaded — complete, accurate, consistent.
**29. Where do you start?** → The source-to-target mapping document.
**30. Types of ETL tests?** → Completeness, accuracy/transformation, integrity, quality, metadata, duplicates, nulls, incremental, regression, performance.
**31. How to check completeness?** → Row counts + two-way MINUS between source and target.
**32. How to test a transformation rule?** → Query comparing target value to the expected calculation from source per the mapping.
**33. What is data integrity? Types?** → Trustworthy/consistent data. Entity (PK), referential (FK), domain (valid values).
**34. How to find orphan records?** → LEFT JOIN + WHERE right key IS NULL, or NOT IN the parent.
**35. Common ETL defects?** → Dropped rows, duplicates from bad joins, truncation, type/format issues, nulls, wrong transformation, broken SCD history.
**36. How do you test huge volumes?** → SQL set comparison (MINUS), counts, sampling, automated regression suite — not manual eyeballing.
**37. Manual vs automated ETL testing?** → Manual SQL queries vs tools (QuerySurge, Informatica DVO) or scripted SQL suites for regression.
**38. Source data has bad data — is it a bug?** → If the source is wrong, it's a source/data-quality issue, not an ETL bug — but ETL should handle/cleanse per the mapping rules; I'd raise it and check the rule.
**39. How do you validate a BI report?** → Reconcile each dashboard metric to the warehouse via SQL; test filters, drill-downs, aggregations, and refresh.

---

## 🌐 Selenium / Automation (15)

**40. What is Selenium?** → Open-source web browser automation tool/library.
**41. Selenium components?** → WebDriver, IDE, Grid.
**42. Types of locators?** → id, name, className, tagName, linkText, partialLinkText, cssSelector, xpath.
**43. Best locator? Why?** → id — fast and usually unique; then CSS; XPath when you need to traverse.
**44. Absolute vs relative XPath?** → Absolute from root (brittle); relative `//tag[@attr='x']` (robust).
**45. Implicit vs explicit wait?** → Implicit: global wait for any element. Explicit: wait for a specific condition (preferred).
**46. Why not Thread.sleep?** → Fixed dumb wait — slow and flaky; explicit waits poll for the actual condition.
**47. What is Page Object Model?** → Each page = a class with its locators + methods; maintainable, reusable, fixes in one place.
**48. Data-driven framework?** → Separate test logic from data; run the same test with many data sets (Excel via Apache POI).
**49. How handle dropdowns?** → `Select` class (selectByVisibleText/Value/Index).
**50. How handle alerts?** → `driver.switchTo().alert().accept()/dismiss()/getText()`.
**51. How handle frames?** → `driver.switchTo().frame(...)`, then `defaultContent()` to exit.
**52. How handle multiple windows?** → `getWindowHandles()` + `switchTo().window(handle)`.
**53. What is StaleElementReferenceException?** → The element reference is no longer valid (DOM changed) — re-locate it.
**54. TestNG annotations?** → @BeforeMethod, @Test, @AfterMethod, @DataProvider, @BeforeClass, etc.
**55. Assert vs Verify (soft assert)?** → Assert stops on failure; soft assert continues and reports all at end.

---

## 🛠️ UFT / UiPath (6)

**56. UFT vs Selenium?** → UFT: commercial, web+desktop+enterprise, VBScript, built-in Object Repository. Selenium: free, web-only, multi-language.
**57. What is an Object Repository (UFT)?** → A store of UI objects and their properties that tests reference.
**58. What is descriptive programming?** → Identifying objects by properties in code without the OR.
**59. What is UiPath / RPA?** → RPA automates repetitive business processes by mimicking human actions across apps; UiPath is a leading RPA tool.
**60. UiPath components?** → Studio (design), Robot (execute), Orchestrator (deploy/schedule/monitor).
**61. Selenium vs RPA?** → Selenium tests an application's UI; RPA automates end-to-end business processes (and can also test via UiPath Test Suite).

---

## ⚙️ Process / Agile / Jira (12)

**62. STLC phases?** → Requirement analysis, planning, case design, env setup, execution, defect reporting, closure.
**63. Severity vs Priority?** → Severity = impact (technical); Priority = urgency to fix (business). Independent — give the misspelled-logo (low sev/high pri) and crash-on-unused-screen (high sev/low pri) examples.
**64. Bug life cycle?** → New → Assigned → Open → Fixed → Retest → Closed (Reopened/Rejected/Deferred/Duplicate).
**65. What goes in a defect report?** → ID, summary, steps, expected vs actual, severity, priority, environment, screenshots/logs.
**66. Equivalence Partitioning?** → Group inputs into classes behaving the same; test one per class.
**67. Boundary Value Analysis?** → Test the edges (min-1, min, max, max+1).
**68. Smoke vs Sanity?** → Smoke: build stable enough to test? Sanity: focused check of specific fixed area.
**69. Regression vs Retesting?** → Regression: re-test existing features after change. Retest: re-test the specific fixed bug.
**70. What is Agile/Scrum?** → Iterative delivery in sprints; Scrum has roles (PO, SM, Dev), ceremonies (planning, standup, review, retro), artifacts (backlogs, increment).
**71. Tester's role in Agile?** → Involved from start (shift-left), write cases from acceptance criteria, test in sprint, automate regression.
**72. What is a Requirement Traceability Matrix?** → Maps requirements → test cases → defects to prove coverage.
**73. What is UAT?** → User Acceptance Testing — business users validate it meets needs before go-live.

---

## 🧑 Behavioral (8) — use STAR (Situation, Task, Action, Result)

**74. Tell me about yourself.** → Use the 30-second pitch (file 00), tailored to the track.
**75. A tough bug you found?** → Pick a data-integrity or transformation bug; explain how SQL/automation caught it and the impact.
**76. How do you prioritize when everything's urgent?** → By risk and business impact; severity×priority; communicate trade-offs.
**77. Disagreement with a developer?** → Stay factual — show the failing query/log/repro steps; focus on the data, not the person.
**78. Tight deadline, can't test everything?** → Risk-based testing: cover the highest-risk, highest-impact areas first; communicate what's not covered.
**79. How do you ensure data quality?** → Completeness, accuracy, integrity, and consistency checks via SQL, plus automated regression on every load.
**80. Why testing / why DHL?** → Be authentic: you like protecting quality and data integrity; DHL is data-heavy logistics where correctness directly affects shipments and billing.
**81. How do you keep skills current?** → Practice SQL, build automation frameworks, follow QA communities — tie to what you actually do.

---

## When you don't know an answer
> "I haven't worked with that exact piece, but based on [related thing], I'd approach it by [reasoning]. Is that how your team does it?"

Honest + reasoning beats bluffing.
