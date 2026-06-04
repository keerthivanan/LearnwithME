# SQL & Pandas Cheatsheet

---

## SQL — Interview Essentials

### Basic SELECT
```sql
SELECT name, age, salary
FROM employees
WHERE department = 'Engineering' AND salary > 80000
ORDER BY salary DESC
LIMIT 10;
```

### Aggregations
```sql
SELECT department,
       COUNT(*)         AS employee_count,
       AVG(salary)      AS avg_salary,
       MAX(salary)      AS max_salary,
       MIN(salary)      AS min_salary,
       SUM(salary)      AS total_salary
FROM employees
GROUP BY department
HAVING AVG(salary) > 70000   -- filter AFTER aggregation
ORDER BY avg_salary DESC;
```

### JOINs
```sql
-- INNER JOIN: only matching rows
SELECT e.name, d.dept_name
FROM employees e
INNER JOIN departments d ON e.dept_id = d.id;

-- LEFT JOIN: all from left, matching from right (nulls if no match)
SELECT e.name, d.dept_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.id;

-- Self join (find manager name)
SELECT e.name, m.name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;
```

### Window Functions (Important!)
```sql
-- Rank within group
SELECT name, department, salary,
       RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rank,
       DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dense_rank,
       ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS row_num
FROM employees;

-- Running total
SELECT name, salary,
       SUM(salary) OVER (PARTITION BY dept ORDER BY hire_date) AS running_total
FROM employees;

-- Previous/next row
SELECT name, salary,
       LAG(salary, 1)  OVER (ORDER BY hire_date) AS prev_salary,
       LEAD(salary, 1) OVER (ORDER BY hire_date) AS next_salary
FROM employees;

-- Percentile rank
SELECT name, salary,
       PERCENT_RANK() OVER (ORDER BY salary) AS pct_rank,
       NTILE(4) OVER (ORDER BY salary) AS quartile
FROM employees;
```

### Subqueries & CTEs
```sql
-- Subquery
SELECT * FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- CTE (cleaner, reusable)
WITH avg_by_dept AS (
    SELECT department, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY department
)
SELECT e.name, e.salary, a.avg_sal
FROM employees e
JOIN avg_by_dept a ON e.department = a.department
WHERE e.salary > a.avg_sal;

-- Recursive CTE (org hierarchy)
WITH RECURSIVE org AS (
    SELECT id, name, manager_id, 0 AS level
    FROM employees WHERE manager_id IS NULL
    UNION ALL
    SELECT e.id, e.name, e.manager_id, o.level + 1
    FROM employees e
    JOIN org o ON e.manager_id = o.id
)
SELECT * FROM org;
```

### String Functions
```sql
UPPER(name), LOWER(name)
TRIM(name), LTRIM(name), RTRIM(name)
LENGTH(name)
SUBSTRING(name, 1, 3)          -- first 3 chars
CONCAT(first_name, ' ', last_name)
REPLACE(text, 'old', 'new')
LIKE '%pattern%'               -- wildcard search
REGEXP_LIKE(text, '^[A-Z]')   -- regex
```

### Date Functions
```sql
NOW(), CURRENT_DATE, CURRENT_TIMESTAMP
DATE_ADD(date, INTERVAL 7 DAY)
DATEDIFF(end_date, start_date)
DATE_FORMAT(date, '%Y-%m')    -- format date
YEAR(date), MONTH(date), DAY(date), DAYOFWEEK(date)
```

### Common Interview Questions (SQL)

**Q: Find the second highest salary**
```sql
SELECT MAX(salary) FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);

-- OR using window function
SELECT DISTINCT salary
FROM (SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk FROM employees)
WHERE rnk = 2;
```

**Q: Find duplicates**
```sql
SELECT email, COUNT(*) FROM users
GROUP BY email
HAVING COUNT(*) > 1;
```

**Q: Delete duplicates, keep one**
```sql
DELETE FROM users
WHERE id NOT IN (
    SELECT MIN(id) FROM users GROUP BY email
);
```

**Q: Customers with no orders**
```sql
SELECT c.* FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
WHERE o.customer_id IS NULL;
```

**Q: Month-over-month growth**
```sql
WITH monthly AS (
    SELECT DATE_FORMAT(order_date, '%Y-%m') AS month,
           SUM(revenue) AS total
    FROM orders
    GROUP BY 1
)
SELECT month, total,
       LAG(total) OVER (ORDER BY month) AS prev_month,
       ROUND(100.0 * (total - LAG(total) OVER (ORDER BY month)) /
             LAG(total) OVER (ORDER BY month), 2) AS growth_pct
FROM monthly;
```

---

## Pandas Cheatsheet

### Load & Inspect
```python
df = pd.read_csv('data.csv')
df.head(), df.tail(), df.sample(5)
df.shape, df.dtypes, df.describe()
df.isnull().sum(), df.duplicated().sum()
```

### Select & Filter
```python
df['col']                          # column (Series)
df[['col1', 'col2']]              # multiple columns
df.iloc[0]                         # row by position
df.loc[df['age'] > 25]            # filter rows
df[(df['age'] > 25) & (df['city'] == 'Mumbai')]  # multiple conditions
df[df['city'].isin(['Mumbai', 'Delhi'])]
df[~df['city'].isin(['Mumbai'])]   # NOT IN
```

### Clean
```python
df.dropna()                         # drop rows with any null
df.fillna(df.median())              # fill with median
df.drop_duplicates()                # remove duplicates
df['col'].astype(int)               # convert type
df['date'] = pd.to_datetime(df['date'])
df['cat'] = df['cat'].astype('category')
df['col'] = df['col'].str.strip().str.lower()   # clean strings
```

### Transform
```python
df['new'] = df['a'] + df['b']
df['log_price'] = np.log1p(df['price'])
df['label'] = df['score'].apply(lambda x: 'high' if x > 80 else 'low')
df['city'].map({'Mumbai': 0, 'Delhi': 1})
pd.get_dummies(df, columns=['city'])
df.rename(columns={'old': 'new'})
df.drop(columns=['col1', 'col2'])
```

### GroupBy (SQL Equivalent)
```python
# SELECT dept, AVG(salary) FROM emp GROUP BY dept
df.groupby('dept')['salary'].mean()

# Multiple aggregations
df.groupby('dept').agg({'salary': ['mean', 'max', 'count'], 'age': 'mean'})

# Multiple group keys
df.groupby(['dept', 'city'])['salary'].mean()

# Transform (keep original DataFrame shape)
df['dept_avg'] = df.groupby('dept')['salary'].transform('mean')
```

### Merge / Join
```python
# INNER JOIN
pd.merge(df1, df2, on='id', how='inner')

# LEFT JOIN
pd.merge(df1, df2, on='id', how='left')

# Different column names
pd.merge(df1, df2, left_on='user_id', right_on='id')

# Concat (stack)
pd.concat([df1, df2], axis=0, ignore_index=True)
```

### Window Functions
```python
df['rolling_7d']   = df['sales'].rolling(7).mean()
df['cumsum']       = df['sales'].cumsum()
df['rank']         = df['score'].rank(ascending=False)
df['pct_rank']     = df['score'].rank(pct=True)
df['prev']         = df.groupby('user')['sales'].shift(1)  # LAG
df['user_rank']    = df.groupby('dept')['salary'].rank(method='dense')
```

### Pivot & Reshape
```python
# Pivot table
pd.pivot_table(df, values='sales', index='dept', columns='month', aggfunc='sum')

# Melt (wide to long)
pd.melt(df, id_vars=['id'], value_vars=['jan', 'feb', 'mar'],
        var_name='month', value_name='sales')

# Crosstab
pd.crosstab(df['dept'], df['city'], margins=True)
```

### SQL → Pandas Reference

| SQL | Pandas |
|-----|--------|
| `SELECT col FROM df` | `df['col']` |
| `WHERE col > 5` | `df[df['col'] > 5]` |
| `GROUP BY col` | `df.groupby('col')` |
| `ORDER BY col DESC` | `df.sort_values('col', ascending=False)` |
| `LIMIT 10` | `df.head(10)` |
| `INNER JOIN` | `pd.merge(df1, df2, how='inner')` |
| `HAVING COUNT > 1` | `.filter(lambda x: len(x) > 1)` |
| `DISTINCT` | `df.drop_duplicates()` |
| `COUNT(*)` | `df.groupby('col').size()` |
| `NULL check` | `df['col'].isnull()` |
| `LIKE '%abc%'` | `df['col'].str.contains('abc')` |
| `CASE WHEN` | `np.where(cond, val1, val2)` |
