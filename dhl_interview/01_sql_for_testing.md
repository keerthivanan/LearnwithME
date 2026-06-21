# SQL for Testing — The #1 Skill (Deep)

For ETL/DW testing, SQL is everything. They WILL ask you to write queries. Master this file.

---

## 1. The query you'll write 100 times: Source vs Target comparison

The heart of ETL testing: did the data move correctly from source to target?

### Find rows in SOURCE but missing in TARGET (data loss)
```sql
SELECT id, name, amount FROM source_table
MINUS                                  -- (Teradata/Oracle); EXCEPT in SQL Server/Postgres
SELECT id, name, amount FROM target_table;
```
If this returns rows → those rows were **lost or changed** during ETL. ❌

### Find rows in TARGET but not in SOURCE (extra/duplicate data)
```sql
SELECT id, name, amount FROM target_table
MINUS
SELECT id, name, amount FROM source_table;
```

> **Interview gold:** "I validate ETL using a two-way MINUS (or EXCEPT) between source and target on the relevant columns. If either direction returns rows, there's a data loss, duplication, or transformation defect."

---

## 2. Row count validation (completeness)
```sql
SELECT COUNT(*) FROM source_table;   -- e.g., 10,000
SELECT COUNT(*) FROM target_table;   -- should match (or match after known filters)
```
Counts differ → records dropped or duplicated. Always check counts first.

---

## 3. Finding DUPLICATES (a classic interview question)
```sql
SELECT id, COUNT(*)
FROM   orders
GROUP BY id
HAVING COUNT(*) > 1;
```
`GROUP BY` the key, `HAVING COUNT(*) > 1` shows keys that appear more than once. **Memorize this.**

---

## 4. Finding NULLs (data integrity)
```sql
SELECT * FROM customers WHERE email IS NULL;          -- missing required field
SELECT COUNT(*) FROM customers WHERE phone IS NULL;   -- how many missing
```
Use `IS NULL` / `IS NOT NULL` — never `= NULL` (that never works).

---

## 5. JOINS — know all four cold

```sql
-- INNER JOIN: only matching rows in both tables
SELECT o.id, c.name
FROM orders o
INNER JOIN customers c ON o.customer_id = c.id;

-- LEFT JOIN: all orders, even those with no matching customer (finds orphans!)
SELECT o.id, c.name
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.id
WHERE c.id IS NULL;          -- orphan orders = referential integrity problem
```

| Join | Returns |
|------|---------|
| INNER | Only rows matching in both |
| LEFT (OUTER) | All left rows + matches (NULLs where no match) |
| RIGHT (OUTER) | All right rows + matches |
| FULL (OUTER) | Everything from both sides |

> **Testing use:** LEFT JOIN + `WHERE right.key IS NULL` finds orphan records (a foreign key pointing to a customer that doesn't exist) — a key data-integrity test.

---

## 6. Aggregations & GROUP BY
```sql
SELECT region, COUNT(*) AS order_count, SUM(amount) AS total
FROM   orders
GROUP BY region
HAVING SUM(amount) > 100000;       -- HAVING filters groups; WHERE filters rows
```
**WHERE vs HAVING:** WHERE filters individual rows (before grouping); HAVING filters groups (after grouping). Common interview question.

---

## 7. The order SQL actually runs (know this!)
You WRITE: SELECT → FROM → WHERE → GROUP BY → HAVING → ORDER BY
It RUNS: **FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY**
(That's why you can't use a SELECT alias in WHERE — WHERE runs before SELECT.)

---

## 8. Window functions (intermediate — impresses)
```sql
-- Rank orders by amount within each region
SELECT id, region, amount,
       ROW_NUMBER() OVER (PARTITION BY region ORDER BY amount DESC) AS rn
FROM orders;
```
- `ROW_NUMBER()` — unique number per row in the partition
- `RANK()` — same rank for ties, skips numbers
- `DENSE_RANK()` — same rank for ties, no skip
**Use case:** find the latest record per key (deduplication): `WHERE rn = 1`.

---

## 9. Set operators
| Operator | Meaning |
|----------|---------|
| UNION | Combine + remove duplicates |
| UNION ALL | Combine + keep duplicates (faster) |
| INTERSECT | Rows in BOTH |
| MINUS / EXCEPT | Rows in first NOT in second (your ETL workhorse) |

---

## 10. Common testing queries to have ready

**Check a transformation rule** (e.g., target amount should be source × 1.1):
```sql
SELECT s.id, s.amount AS src, t.amount AS tgt
FROM source s JOIN target t ON s.id = t.id
WHERE t.amount <> s.amount * 1.1;     -- rows where the rule was broken
```

**Check date format / range:**
```sql
SELECT * FROM orders WHERE order_date > CURRENT_DATE;   -- future dates = bad data
```

**Check referential integrity:**
```sql
SELECT * FROM orders WHERE customer_id NOT IN (SELECT id FROM customers);
```

**Trim / case mismatches (common ETL bug):**
```sql
SELECT * FROM target WHERE name <> TRIM(name);          -- extra spaces
SELECT * FROM target WHERE city <> UPPER(city);         -- case not standardized
```

---

## 11. SQL functions to know
- **String:** `TRIM`, `UPPER`, `LOWER`, `SUBSTRING/SUBSTR`, `LENGTH`, `COALESCE`, `CONCAT`
- **Date:** `CURRENT_DATE`, `EXTRACT`, date diff functions
- **Aggregate:** `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`
- **Null handling:** `COALESCE(col, 'default')`, `NVL` (Oracle/Teradata), `IS NULL`
- **Conditional:** `CASE WHEN ... THEN ... ELSE ... END`

```sql
SELECT id,
  CASE WHEN amount > 1000 THEN 'High'
       WHEN amount > 100  THEN 'Medium'
       ELSE 'Low' END AS tier
FROM orders;
```

---

## 12. Practice these out loud (likely live questions)
1. "Write a query to find duplicate records." → GROUP BY + HAVING COUNT(*)>1
2. "Find records in source not in target." → MINUS / EXCEPT
3. "Find the 2nd highest salary." → see below
4. "Find employees with no department." → LEFT JOIN + IS NULL
5. "Count NULLs in a column." → COUNT(*) WHERE col IS NULL

**2nd highest salary (very common):**
```sql
SELECT MAX(salary) FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);
-- or
SELECT DISTINCT salary FROM employees ORDER BY salary DESC
OFFSET 1 ROW FETCH NEXT 1 ROW ONLY;   -- or use DENSE_RANK()=2
```

---

## The line that proves you're a real data tester
> "My first three ETL checks are always: row counts match, a two-way MINUS finds no mismatches, and no unexpected NULLs or duplicates on the key. Then I validate each transformation rule with a targeted query comparing source and target."
