-- ════════════════════════════════════════════════════════
-- SQL Interview Queries — Definitions + Real Examples
-- ════════════════════════════════════════════════════════
-- WHAT IS SQL?
--   → Structured Query Language — used to talk to relational databases
--   → A relational DB stores data in TABLES (rows + columns)
--   → SQL lets you: CREATE tables, INSERT data, SELECT/filter/aggregate,
--     JOIN tables, and DELETE/UPDATE records
--
-- EXECUTION ORDER (not the order you write them!):
--   1. FROM / JOIN   → which tables?
--   2. WHERE         → filter rows BEFORE grouping
--   3. GROUP BY      → group remaining rows
--   4. HAVING        → filter AFTER grouping (on aggregated values)
--   5. SELECT        → pick columns / compute expressions
--   6. ORDER BY      → sort results
--   7. LIMIT         → take only N rows
-- ════════════════════════════════════════════════════════

-- ── SETUP ────────────────────────────────────────────────
CREATE TABLE employees (
    id          INT PRIMARY KEY,
    name        VARCHAR(100),
    department  VARCHAR(50),
    salary      DECIMAL(10, 2),
    manager_id  INT,           -- references id of another employee (self-join)
    hire_date   DATE,
    city        VARCHAR(50)
);

CREATE TABLE orders (
    id          INT PRIMARY KEY,
    customer_id INT,
    amount      DECIMAL(10, 2),
    status      VARCHAR(20),   -- 'completed', 'pending', 'cancelled'
    created_at  DATETIME
);

CREATE TABLE customers (
    id    INT PRIMARY KEY,
    name  VARCHAR(100),
    email VARCHAR(100)
);


-- ══════════════════════════════════════════════════════════
-- 1. BASIC QUERIES
-- ══════════════════════════════════════════════════════════
-- WHAT IS SELECT?
--   → Retrieves data from one or more tables
--   → SELECT columns FROM table WHERE condition ORDER BY col LIMIT n
--   → * means "all columns" — avoid in production (fragile, slow)
--   → WHERE filters ROWS (before grouping)
--   → ORDER BY sorts: ASC (default, small→big), DESC (big→small)
--   → LIMIT n — only return first n rows

SELECT name, salary
FROM employees
WHERE department = 'Engineering'
  AND salary > 80000
ORDER BY salary DESC
LIMIT 10;


-- ══════════════════════════════════════════════════════════
-- 2. AGGREGATIONS
-- ══════════════════════════════════════════════════════════
-- WHAT ARE AGGREGATE FUNCTIONS?
--   → Compute a SINGLE value from many rows
--   → COUNT(*)   → total row count  (includes NULLs)
--   → COUNT(col) → count non-NULL values in col
--   → SUM(col)   → total
--   → AVG(col)   → average (ignores NULLs)
--   → MAX / MIN  → largest / smallest value
--
-- GROUP BY:
--   → Splits rows into GROUPS and applies aggregate to each group
--   → Every column in SELECT must either be in GROUP BY or aggregated
--
-- HAVING:
--   → Like WHERE but filters AFTER grouping (on aggregate values)
--   → WHERE  → filters individual rows  (before GROUP BY)
--   → HAVING → filters groups           (after  GROUP BY)

SELECT
    department,
    COUNT(*)                AS headcount,        -- how many employees
    ROUND(AVG(salary), 2)  AS avg_salary,
    MAX(salary)            AS max_salary,
    MIN(salary)            AS min_salary,
    SUM(salary)            AS total_payroll
FROM employees
GROUP BY department
HAVING AVG(salary) > 70000      -- only show departments with high avg salary
ORDER BY avg_salary DESC;


-- ══════════════════════════════════════════════════════════
-- 3. JOINS
-- ══════════════════════════════════════════════════════════
-- WHAT IS A JOIN?
--   → Combines rows from TWO tables based on a matching condition
--
-- TYPES:
--   → INNER JOIN : only rows that MATCH in BOTH tables
--   → LEFT JOIN  : ALL rows from LEFT table + matched from right (NULLs if no match)
--   → RIGHT JOIN : ALL rows from RIGHT table + matched from left
--   → FULL JOIN  : ALL rows from BOTH tables (NULLs where no match)
--   → SELF JOIN  : join a table to ITSELF (e.g., employee → their manager)
--   → CROSS JOIN : every row × every row (cartesian product) — rarely used
--
-- KEY RULE:
--   → LEFT JOIN: use when you want ALL records from left table
--     even if they have no matching records in the right table
--     (e.g., customers with NO orders still appear — with NULL order values)

-- LEFT JOIN: all customers, even those with no orders
SELECT
    c.name,
    c.email,
    COUNT(o.id)     AS order_count,     -- 0 if no orders
    SUM(o.amount)   AS total_spent      -- NULL if no orders
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
GROUP BY c.id, c.name, c.email
ORDER BY total_spent DESC;

-- SELF JOIN: employee with their manager's name
-- Same table joined to itself — need aliases to tell them apart
SELECT
    e.name        AS employee,
    m.name        AS manager,           -- m = manager row (also from employees)
    e.salary,
    e.department
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;


-- ══════════════════════════════════════════════════════════
-- 4. WINDOW FUNCTIONS
-- ══════════════════════════════════════════════════════════
-- WHAT ARE WINDOW FUNCTIONS?
--   → Like aggregates BUT they DON'T collapse rows into groups
--   → Each row keeps its identity AND gets a computed value across a "window"
--   → Syntax: FUNCTION() OVER (PARTITION BY col ORDER BY col)
--       PARTITION BY → like GROUP BY but keeps all rows (divides into windows)
--       ORDER BY     → defines the row ordering within each window
--
-- KEY WINDOW FUNCTIONS:
--   → ROW_NUMBER() → 1,2,3,4... unique per window (no ties)
--   → RANK()       → 1,2,2,4... skips numbers on ties
--   → DENSE_RANK() → 1,2,2,3... no gaps on ties ← most interview-friendly
--   → NTILE(n)     → divides rows into n equal buckets (quartiles etc.)
--   → LAG(col, n)  → value from n rows BEFORE current row
--   → LEAD(col, n) → value from n rows AFTER current row
--   → SUM() OVER   → running total
--   → AVG() OVER (ROWS BETWEEN n PRECEDING AND CURRENT ROW) → rolling average

-- Rank employees by salary within each department
SELECT
    name,
    department,
    salary,
    RANK()       OVER (PARTITION BY department ORDER BY salary DESC) AS salary_rank,
    -- 1,2,2,4 → ties get same rank, next rank skips (no rank 3)
    DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dense_rank,
    -- 1,2,2,3 → ties get same rank, no gaps
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS row_num,
    -- 1,2,3,4 → always unique, arbitrary for ties
    NTILE(4)     OVER (ORDER BY salary)                              AS salary_quartile
    -- divides all employees into 4 salary bands (1=bottom, 4=top)
FROM employees;

-- Running total and rolling 3-row average
SELECT
    name,
    hire_date,
    salary,
    SUM(salary) OVER (ORDER BY hire_date)       AS running_total,
    -- cumulative sum as of each hire date
    AVG(salary) OVER (
        ORDER BY hire_date
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW   -- 3-row rolling window
    )                                          AS rolling_avg_3
FROM employees;

-- LAG / LEAD — compare with previous/next row
SELECT
    name,
    hire_date,
    salary,
    LAG(salary, 1)  OVER (ORDER BY hire_date) AS prev_salary,
    -- salary of the person hired just before
    LEAD(salary, 1) OVER (ORDER BY hire_date) AS next_salary,
    -- salary of the person hired just after
    salary - LAG(salary, 1) OVER (ORDER BY hire_date) AS salary_diff
FROM employees;


-- ══════════════════════════════════════════════════════════
-- 5. CTEs (Common Table Expressions)
-- ══════════════════════════════════════════════════════════
-- WHAT IS A CTE?
--   → A named temporary result set defined with WITH keyword
--   → Makes complex queries READABLE — like naming a sub-query
--   → Only lives for the duration of the query (not stored)
--   → You can have MULTIPLE CTEs in one query (chain them)
--
-- SYNTAX:
--   WITH cte_name AS (
--       SELECT ...
--   )
--   SELECT * FROM cte_name;
--
-- RECURSIVE CTE:
--   → References itself to traverse HIERARCHICAL data (org trees, graphs)
--   → Must have a BASE CASE (anchor) + RECURSIVE part joined with UNION ALL

-- CTE: employees earning above their department's average
WITH dept_avg AS (
    SELECT department, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY department
)
SELECT
    e.name,
    e.department,
    e.salary,
    ROUND(da.avg_sal, 0)                                        AS dept_avg,
    ROUND(100.0 * (e.salary - da.avg_sal) / da.avg_sal, 1)    AS pct_above_avg
FROM employees e
JOIN dept_avg da ON e.department = da.department
WHERE e.salary > da.avg_sal
ORDER BY pct_above_avg DESC;

-- RECURSIVE CTE: org chart with depth + full path
WITH RECURSIVE org_tree AS (
    -- BASE CASE: employees with no manager (top of hierarchy)
    SELECT id, name, manager_id, 0 AS depth,
           CAST(name AS CHAR(1000)) AS path
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- RECURSIVE PART: each employee connected to their manager
    SELECT e.id, e.name, e.manager_id,
           ot.depth + 1,
           CONCAT(ot.path, ' > ', e.name)
    FROM employees e
    JOIN org_tree ot ON e.manager_id = ot.id
)
SELECT * FROM org_tree ORDER BY path;
-- Result shows: "CEO > VP Eng > Alice" style paths


-- ══════════════════════════════════════════════════════════
-- 6. COMMON INTERVIEW QUESTIONS
-- ══════════════════════════════════════════════════════════
-- These are the most frequently asked SQL questions in interviews.
-- Learn the PATTERN not just the answer.

-- Q1: Second highest salary
-- PATTERN: use subquery to exclude max, then find max of remainder
SELECT MAX(salary) AS second_highest
FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);

-- BETTER with DENSE_RANK (handles ties correctly):
SELECT salary
FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk
    FROM employees
) ranked
WHERE rnk = 2;

-- Q2: Nth highest salary (generalized — works for any N)
-- Change @N to get any rank
SET @N = 3;
SELECT salary FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk
    FROM employees
) t
WHERE rnk = @N;

-- Q3: Find duplicate emails
-- PATTERN: GROUP BY the column + HAVING COUNT > 1
SELECT email, COUNT(*) AS occurrences
FROM customers
GROUP BY email
HAVING COUNT(*) > 1;

-- Q4: Delete duplicates, keep the one with lowest ID
-- PATTERN: keep MIN(id) per group, delete everything else
DELETE FROM customers
WHERE id NOT IN (
    SELECT MIN(id) FROM customers GROUP BY email
);

-- Q5: Customers with NO orders (find records in A with no match in B)
-- PATTERN: LEFT JOIN + WHERE right side IS NULL
SELECT c.id, c.name, c.email
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
WHERE o.id IS NULL;         -- o.id is NULL when there was no matching order

-- Q6: Month-over-month revenue growth
WITH monthly AS (
    SELECT
        DATE_FORMAT(created_at, '%Y-%m') AS month,
        SUM(amount)                      AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY 1
)
SELECT
    month,
    revenue,
    LAG(revenue) OVER (ORDER BY month)        AS prev_revenue,
    ROUND(
        100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
               / LAG(revenue) OVER (ORDER BY month),
        2
    )                                         AS mom_growth_pct
FROM monthly
ORDER BY month;

-- Q7: Top 3 by revenue per category (top-N per group)
-- PATTERN: RANK inside CTE, then filter rnk <= N
WITH ranked AS (
    SELECT
        category,
        product,
        SUM(amount)                                          AS revenue,
        RANK() OVER (PARTITION BY category ORDER BY SUM(amount) DESC) AS rnk
    FROM sales
    GROUP BY category, product
)
SELECT * FROM ranked WHERE rnk <= 3;

-- Q8: Users who logged in EVERY day for the last 7 days
WITH daily_logins AS (
    SELECT user_id, DATE(login_time) AS login_date
    FROM logins
    WHERE login_time >= CURRENT_DATE - INTERVAL 7 DAY
    GROUP BY user_id, DATE(login_time)   -- deduplicate multiple logins same day
)
SELECT user_id
FROM daily_logins
GROUP BY user_id
HAVING COUNT(DISTINCT login_date) = 7;   -- exactly 7 distinct days

-- Q9: Rolling 7-day average
SELECT
    created_at,
    amount,
    AVG(amount) OVER (
        ORDER BY created_at
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW   -- current + 6 before = 7 rows
    ) AS rolling_7_avg
FROM orders;

-- Q10: PIVOT — rows to columns using CASE WHEN
-- PATTERN: SUM(CASE WHEN condition THEN 1 ELSE 0 END) AS alias
SELECT
    department,
    SUM(CASE WHEN YEAR(hire_date) = 2022 THEN 1 ELSE 0 END) AS hired_2022,
    SUM(CASE WHEN YEAR(hire_date) = 2023 THEN 1 ELSE 0 END) AS hired_2023,
    SUM(CASE WHEN YEAR(hire_date) = 2024 THEN 1 ELSE 0 END) AS hired_2024
FROM employees
GROUP BY department;


-- ══════════════════════════════════════════════════════════
-- 7. STRING & DATE FUNCTIONS
-- ══════════════════════════════════════════════════════════
-- WHAT ARE STRING FUNCTIONS?
--   → UPPER(s) / LOWER(s)    → change case
--   → TRIM(s)                → remove leading/trailing spaces
--   → LENGTH(s)              → number of characters
--   → SUBSTRING(s, start, len) → extract part of string (1-indexed!)
--   → CONCAT(a, b, c)        → join strings together
--   → REPLACE(s, old, new)   → replace all occurrences
--   → s LIKE 'pattern'       → pattern match: % = any chars, _ = one char
--   → REGEXP_LIKE(s, pattern)→ regex match
--
-- WHAT ARE DATE FUNCTIONS?
--   → YEAR/MONTH/DAY(date)   → extract parts
--   → DATEDIFF(d1, d2)       → difference in days
--   → DATE_ADD(d, INTERVAL n DAY/MONTH/YEAR) → add time
--   → DATE_FORMAT(d, fmt)    → format as string (e.g. '%Y-%m-%d')
--   → NOW() / CURRENT_DATE   → current timestamp / date

-- String functions
SELECT
    UPPER(name),                                    -- ALICE
    LOWER(name),                                    -- alice
    TRIM('  Alice  '),                              -- 'Alice'
    LENGTH(name),                                   -- number of chars
    SUBSTRING(name, 1, 3),                          -- first 3 chars
    CONCAT(name, ' — ', department),               -- "Alice — Engineering"
    REPLACE(name, 'Alice', 'Alicia'),               -- rename
    name LIKE '%son',                               -- TRUE if ends with 'son'
    name LIKE 'A%',                                 -- TRUE if starts with 'A'
    REGEXP_LIKE(email, '^[a-z]+@.*\\.com')         -- regex email check
FROM employees;

-- Date functions
SELECT
    hire_date,
    YEAR(hire_date)                                AS hire_year,
    MONTH(hire_date)                               AS hire_month,
    DAYOFWEEK(hire_date)                           AS day_of_week,  -- 1=Sun, 7=Sat
    DATEDIFF(CURRENT_DATE, hire_date)              AS days_employed,
    DATE_ADD(hire_date, INTERVAL 90 DAY)           AS probation_end,
    DATE_FORMAT(hire_date, '%d %M %Y')             AS formatted     -- "15 June 2023"
FROM employees;


-- ══════════════════════════════════════════════════════════
-- 8. SUBQUERIES
-- ══════════════════════════════════════════════════════════
-- WHAT IS A SUBQUERY?
--   → A SELECT inside another SELECT (nested query)
--   → Can appear in: WHERE, FROM, SELECT, HAVING
--   → Correlated subquery: inner query references the OUTER query's row
--     (runs once per outer row — can be slow on large tables)
--
-- SUBQUERY vs JOIN vs CTE:
--   → Use CTE   for readability (complex queries)
--   → Use JOIN  for performance (DB can optimize joins better)
--   → Use subquery for simple lookups or when JOIN would be awkward

-- Employees earning MORE than the company average
SELECT name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- Correlated subquery: employees with max salary IN THEIR department
SELECT name, department, salary
FROM employees e
WHERE salary = (
    SELECT MAX(salary)
    FROM employees
    WHERE department = e.department   -- references outer query's row
);

-- EXISTS: departments that HAVE at least one employee earning > 100k
SELECT DISTINCT department
FROM employees e
WHERE EXISTS (
    SELECT 1
    FROM employees
    WHERE department = e.department
      AND salary > 100000
);


-- ══════════════════════════════════════════════════════════
-- 9. INDEXES & PERFORMANCE
-- ══════════════════════════════════════════════════════════
-- WHAT IS AN INDEX?
--   → A data structure that speeds up lookups on a column
--   → Without index: DB scans EVERY row (full table scan) → slow
--   → With index:    DB jumps directly to matching rows   → fast
--   → Tradeoff: faster reads, slower writes (index must update on INSERT/UPDATE)
--
-- WHEN TO ADD INDEX:
--   → Columns used in WHERE clauses frequently
--   → Columns used in JOIN conditions
--   → Columns used in ORDER BY / GROUP BY
--   → Foreign key columns
--
-- WHEN NOT TO ADD INDEX:
--   → Columns with very few distinct values (e.g., boolean, status with 2 values)
--   → Tables that are written to very frequently (index maintenance overhead)
--   → Small tables (full scan is fine)
--
-- USE EXPLAIN to see query execution plan:
--   → EXPLAIN SELECT ... → shows whether index is used, row count scanned

CREATE INDEX idx_dept    ON employees(department);    -- speed up WHERE department = ?
CREATE INDEX idx_salary  ON employees(salary);        -- speed up ORDER BY salary
CREATE INDEX idx_cust_id ON orders(customer_id);      -- speed up JOIN on customer_id

-- Composite index — covers queries filtering on both columns
CREATE INDEX idx_dept_salary ON employees(department, salary);
-- Useful for: WHERE department = 'Eng' AND salary > 80000

EXPLAIN SELECT name FROM employees WHERE department = 'Engineering';
-- Look for: type=ref (uses index) vs type=ALL (full scan = bad)
