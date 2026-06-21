# Teradata Essentials (for Testers)

The JD names Teradata specifically. You don't need to be a Teradata DBA, but know these concepts and talk credibly.

---

## 1. What is Teradata?
A **massively parallel processing (MPP) relational database** built for large-scale data warehousing and analytics. It spreads data and query work across many processing units, so it handles huge data volumes fast. Common in big enterprises (banks, telcos, logistics) for their warehouse.

---

## 2. The killer concept: AMPs and PARALLELISM
Teradata splits work across **AMPs (Access Module Processors)** — each AMP owns a slice of the data and processes its slice in parallel. More AMPs = more parallelism = faster on big data.

```
Query → PARSING ENGINE (plans it) → BYNET (network) → many AMPs work in parallel → results combined
```

- **Parsing Engine (PE):** receives the SQL, makes the plan, manages sessions
- **BYNET:** the network connecting PEs and AMPs
- **AMP:** does the actual work on its portion of the data
- **Vproc:** virtual processor (AMPs and PEs are vprocs)

---

## 3. The MOST IMPORTANT Teradata concept: the Primary Index (PI)
The **Primary Index determines how rows are distributed across AMPs** (via a hash of the PI value). It is NOT the same as a primary key.

- **Good PI = even distribution** across AMPs (no AMP overloaded) + matches how you query/join
- **Bad PI = data skew** — one AMP gets too much data → slow queries, hot AMP

Types:
- **UPI (Unique Primary Index):** unique values → perfectly even distribution
- **NUPI (Non-Unique Primary Index):** duplicates allowed → can cause skew

> **Interview gold:** "In Teradata, the Primary Index controls data distribution across AMPs via hashing. Choosing a high-cardinality, evenly-distributed PI avoids data skew. As a tester, I watch for skew because it signals a design issue and slows ETL loads."

**Data skew** = uneven data distribution across AMPs. A common Teradata performance problem to mention.

---

## 4. Teradata load utilities (name-drop these)
- **FastLoad:** bulk loads huge data into EMPTY tables, very fast
- **MultiLoad (MLoad):** loads/updates into populated tables (INSERT/UPDATE/DELETE)
- **TPump:** continuous, near-real-time small loads (low volume, always on)
- **FastExport:** bulk exports large data out
- **BTEQ (Basic Teradata Query):** the command-line tool to run SQL and scripts
- **TPT (Teradata Parallel Transporter):** modern unified load/export tool

> Testers care about these because the **load method affects how you validate** (e.g., FastLoad needs an empty table; failed loads land in error tables).

---

## 5. Teradata SQL specifics (vs standard SQL)
- Uses **`MINUS`** (not EXCEPT) for set difference — your ETL comparison query
- **`QUALIFY`** clause — filter on window functions directly:
  ```sql
  SELECT id, amount,
         ROW_NUMBER() OVER (PARTITION BY id ORDER BY load_date DESC) AS rn
  FROM stg_orders
  QUALIFY rn = 1;          -- keep the latest row per id (dedup) — Teradata-specific!
  ```
- **`SAMPLE`** — get a random sample of rows for spot-checks
- **`HELP TABLE tablename`** and **`SHOW TABLE tablename`** — inspect structure
- **Volatile / Global Temporary tables** — for staging intermediate results in a session

---

## 6. Teradata error/recovery tables (testers should know)
When load utilities hit bad rows, they go to:
- **Error Table 1 (ET):** constraint/conversion errors
- **Error Table 2 (UV):** uniqueness violations
- A tester checks these to find rejected records.

---

## 7. Performance terms to mention
- **Data skew** — uneven distribution (bad PI)
- **Collect Statistics** — Teradata gathers stats so the optimizer plans well; stale stats = bad plans
- **Explain plan** — `EXPLAIN <query>` shows how Teradata will run it (spot full table scans, skew)
- **Spool space** — temporary space for intermediate results; "spool space error" = query used too much

---

## 8. If asked "have you used Teradata deeply?"
Be honest, frame transferable knowledge:
> "I've worked with Teradata as a tester — writing validation SQL with MINUS and QUALIFY, understanding that the Primary Index drives data distribution across AMPs, and checking load error tables when FastLoad/MultiLoad jobs reject rows. I'm comfortable picking up deeper Teradata administration, but my focus is using it to validate warehouse data."

---

## The sentence that sounds like a real Teradata tester
> "Teradata is an MPP database — the Primary Index hashes rows across AMPs for parallel processing, so I always check for data skew. I validate loads done via FastLoad or MultiLoad, use QUALIFY for deduplication checks, MINUS for source-to-target comparison, and inspect the ET/UV error tables for rejected records."
