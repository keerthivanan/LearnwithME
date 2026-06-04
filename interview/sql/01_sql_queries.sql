-- SQL Interview Queries — Real Examples

-- ── SETUP ────────────────────────────────────────────
CREATE TABLE employees (
    id          INT PRIMARY KEY,
    name        VARCHAR(100),
    department  VARCHAR(50),
    salary      DECIMAL(10, 2),
    manager_id  INT,
    hire_date   DATE,
    city        VARCHAR(50)
);

CREATE TABLE orders (
    id          INT PRIMARY KEY,
    customer_id INT,
    amount      DECIMAL(10, 2),
    status      VARCHAR(20),
    created_at  DATETIME
);

CREATE TABLE customers (
    id    INT PRIMARY KEY,
    name  VARCHAR(100),
    email VARCHAR(100)
);


-- ── BASIC QUERIES ─────────────────────────────────────
SELECT name, salary
FROM employees
WHERE department = 'Engineering'
  AND salary > 80000
ORDER BY salary DESC
LIMIT 10;


-- ── AGGREGATIONS ──────────────────────────────────────
SELECT
    department,
    COUNT(*)            AS headcount,
    ROUND(AVG(salary), 2)  AS avg_salary,
    MAX(salary)         AS max_salary,
    MIN(salary)         AS min_salary,
    SUM(salary)         AS total_payroll
FROM employees
GROUP BY department
HAVING AVG(salary) > 70000
ORDER BY avg_salary DESC;


-- ── JOINS ─────────────────────────────────────────────
-- Customers with their orders
SELECT
    c.name,
    c.email,
    COUNT(o.id)     AS order_count,
    SUM(o.amount)   AS total_spent
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
GROUP BY c.id, c.name, c.email
ORDER BY total_spent DESC;

-- Self join: employees with their manager's name
SELECT
    e.name        AS employee,
    m.name        AS manager,
    e.salary,
    e.department
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;


-- ── WINDOW FUNCTIONS ──────────────────────────────────
-- Rank employees by salary within department
SELECT
    name,
    department,
    salary,
    RANK()       OVER (PARTITION BY department ORDER BY salary DESC) AS salary_rank,
    DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dense_rank,
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS row_num,
    NTILE(4)     OVER (ORDER BY salary)                              AS salary_quartile
FROM employees;

-- Running total of salary by hire date
SELECT
    name,
    hire_date,
    salary,
    SUM(salary) OVER (ORDER BY hire_date)                 AS running_total,
    AVG(salary) OVER (ORDER BY hire_date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS rolling_avg_3
FROM employees;

-- Lag / Lead (compare with previous/next row)
SELECT
    name,
    hire_date,
    salary,
    LAG(salary, 1)  OVER (ORDER BY hire_date) AS prev_salary,
    LEAD(salary, 1) OVER (ORDER BY hire_date) AS next_salary,
    salary - LAG(salary, 1) OVER (ORDER BY hire_date) AS salary_diff
FROM employees;


-- ── CTEs ──────────────────────────────────────────────
-- Above average salary per department
WITH dept_avg AS (
    SELECT department, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY department
)
SELECT
    e.name,
    e.department,
    e.salary,
    da.avg_sal,
    ROUND(100.0 * (e.salary - da.avg_sal) / da.avg_sal, 1) AS pct_above_avg
FROM employees e
JOIN dept_avg da ON e.department = da.department
WHERE e.salary > da.avg_sal
ORDER BY pct_above_avg DESC;

-- Recursive CTE: org hierarchy depth
WITH RECURSIVE org_tree AS (
    -- Base case: top-level (no manager)
    SELECT id, name, manager_id, 0 AS depth, CAST(name AS CHAR(1000)) AS path
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive: each employee and their level
    SELECT e.id, e.name, e.manager_id, ot.depth + 1, CONCAT(ot.path, ' > ', e.name)
    FROM employees e
    JOIN org_tree ot ON e.manager_id = ot.id
)
SELECT * FROM org_tree ORDER BY path;


-- ── COMMON INTERVIEW QUESTIONS ────────────────────────

-- Q1: Second highest salary
SELECT MAX(salary)
FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);

-- Using DENSE_RANK (better — handles ties)
SELECT salary
FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk
    FROM employees
) ranked
WHERE rnk = 2
LIMIT 1;


-- Q2: Nth highest salary (generalized)
-- Change @N to any rank
SET @N = 3;
SELECT salary FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk
    FROM employees
) t
WHERE rnk = @N;


-- Q3: Find duplicate emails
SELECT email, COUNT(*) AS count
FROM customers
GROUP BY email
HAVING COUNT(*) > 1;


-- Q4: Delete duplicates, keep lowest ID
DELETE FROM customers
WHERE id NOT IN (
    SELECT MIN(id) FROM customers GROUP BY email
);


-- Q5: Customers with NO orders
SELECT c.*
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
WHERE o.id IS NULL;


-- Q6: Month-over-month revenue growth
WITH monthly AS (
    SELECT
        DATE_FORMAT(created_at, '%Y-%m') AS month,
        SUM(amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY 1
)
SELECT
    month,
    revenue,
    LAG(revenue) OVER (ORDER BY month) AS prev_revenue,
    ROUND(
        100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
               / LAG(revenue) OVER (ORDER BY month),
        2
    ) AS mom_growth_pct
FROM monthly
ORDER BY month;


-- Q7: Top 3 products by revenue per category
WITH ranked AS (
    SELECT
        category,
        product,
        SUM(amount) AS revenue,
        RANK() OVER (PARTITION BY category ORDER BY SUM(amount) DESC) AS rnk
    FROM sales
    GROUP BY category, product
)
SELECT * FROM ranked WHERE rnk <= 3;


-- Q8: Users who logged in every day for the last 7 days
WITH daily_logins AS (
    SELECT user_id, DATE(login_time) AS login_date
    FROM logins
    WHERE login_time >= CURRENT_DATE - INTERVAL 7 DAY
    GROUP BY user_id, DATE(login_time)
)
SELECT user_id
FROM daily_logins
GROUP BY user_id
HAVING COUNT(DISTINCT login_date) = 7;


-- Q9: Rolling 7-day average
SELECT
    created_at,
    amount,
    AVG(amount) OVER (
        ORDER BY created_at
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7_avg
FROM orders;


-- Q10: Pivot (rows to columns)
SELECT
    department,
    SUM(CASE WHEN YEAR(hire_date) = 2022 THEN 1 ELSE 0 END) AS hired_2022,
    SUM(CASE WHEN YEAR(hire_date) = 2023 THEN 1 ELSE 0 END) AS hired_2023,
    SUM(CASE WHEN YEAR(hire_date) = 2024 THEN 1 ELSE 0 END) AS hired_2024
FROM employees
GROUP BY department;


-- ── STRING & DATE FUNCTIONS ───────────────────────────
SELECT
    UPPER(name),
    LOWER(name),
    TRIM(name),
    LENGTH(name),
    SUBSTRING(name, 1, 5),
    CONCAT(name, ' - ', department),
    REPLACE(name, 'Alice', 'Alicia'),
    name LIKE '%son',                     -- ends with 'son'
    REGEXP_LIKE(email, '^[a-z]+@.*\\.com')

FROM employees;

SELECT
    hire_date,
    YEAR(hire_date),
    MONTH(hire_date),
    DAYOFWEEK(hire_date),
    DATEDIFF(CURRENT_DATE, hire_date)  AS days_employed,
    DATE_ADD(hire_date, INTERVAL 90 DAY) AS probation_end,
    DATE_FORMAT(hire_date, '%d %M %Y')  AS formatted
FROM employees;
