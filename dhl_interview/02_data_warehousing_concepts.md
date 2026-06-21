# Data Warehousing Concepts (Deep)

You MUST know these cold for ETL/DW testing. They come up in every interview.

---

## 1. OLTP vs OLAP (the foundation)

| | OLTP (Transactional) | OLAP (Analytical / Warehouse) |
|--|----------------------|-------------------------------|
| Purpose | Run the business (orders, payments) | Analyze the business (reports, trends) |
| Operations | Many small INSERT/UPDATE/DELETE | Large complex SELECT/aggregations |
| Data | Current, detailed | Historical, summarized |
| Design | Normalized (3NF) | Denormalized (star schema) |
| Example | The DHL booking app DB | The DHL reporting warehouse |

> A **data warehouse is OLAP** — built for reporting and analysis, fed from OLTP systems via ETL.

---

## 2. What is a Data Warehouse?
A central repository that **integrates data from many source systems** into one place, optimized for reporting and analysis. It's:
- **Subject-oriented** (organized by subject: sales, shipments, customers)
- **Integrated** (consistent formats from different sources)
- **Time-variant** (keeps history)
- **Non-volatile** (data isn't changed/deleted, only added)

*(That's the famous Inmon definition — good to quote.)*

---

## 3. ETL — the pipeline you test
**ETL = Extract, Transform, Load**
```
SOURCE SYSTEMS → [EXTRACT] → STAGING → [TRANSFORM] → [LOAD] → DATA WAREHOUSE → BI Reports
```
- **Extract:** pull data from sources (databases, files, APIs)
- **Transform:** clean, standardize, apply business rules, join, aggregate (this is where most bugs hide!)
- **Load:** insert into the warehouse (dimension and fact tables)

**ELT** is a variant: load raw first, transform inside the warehouse (modern cloud approach).

---

## 4. The Star Schema (know this diagram)

```
          [Dim_Date]
              |
[Dim_Customer] — [FACT_SHIPMENTS] — [Dim_Product]
              |
         [Dim_Location]
```

- **Fact table** (center): the measurements/numbers — shipment_count, revenue, weight. Has foreign keys to dimensions + the numeric "measures".
- **Dimension tables** (points of the star): the descriptive context — who, what, when, where. Customer name, date, product, location.

**Star schema = one fact table surrounded by dimension tables.** Denormalized, fast for queries.

### Snowflake schema
Like a star, but dimensions are **normalized** (split into sub-tables). E.g., Dim_Product → Dim_Category. More joins, less redundancy.

| | Star | Snowflake |
|--|------|-----------|
| Dimensions | Denormalized (flat) | Normalized (split) |
| Joins | Fewer (faster) | More (slower) |
| Storage | More redundancy | Less |

---

## 5. Facts and Dimensions (deeper)

**Fact table types:**
- **Transaction fact** — one row per event (one shipment)
- **Snapshot fact** — state at a point in time (daily inventory)
- **Accumulating snapshot** — one row updated through a process (order → ship → deliver)

**Measure types:**
- **Additive** — can sum across all dimensions (revenue)
- **Semi-additive** — sum across some, not time (account balance)
- **Non-additive** — can't sum (ratios, percentages)

**Dimension keys:**
- **Natural key** — the business key from the source (customer_id from the app)
- **Surrogate key** — a warehouse-generated integer key (best practice; stable, handles history)

---

## 6. Slowly Changing Dimensions (SCD) — GUARANTEED question

When a dimension attribute changes over time (a customer moves city), how do you handle it?

| Type | What it does | Example |
|------|-------------|---------|
| **SCD Type 0** | Never changes (ignore updates) | Date of birth |
| **SCD Type 1** | Overwrite — no history | Fix a typo in a name |
| **SCD Type 2** | Add a NEW row, keep old — full history | Customer moves city; old + new rows with effective dates |
| **SCD Type 3** | Add a column for previous value — limited history | "current_city" + "previous_city" |

**SCD Type 2 is the most important.** It adds rows with:
- A surrogate key (new for each version)
- `effective_date` / `end_date`
- A `current_flag` (Y/N) to mark the active row

> **Testing SCD Type 2:** verify that when a source attribute changes, the old row is closed (end_date set, flag=N) and a new row is inserted (flag=Y) — and history isn't lost.

---

## 7. Key warehouse terms
- **Staging area:** temporary landing zone for raw extracted data before transformation
- **Data mart:** a subset of the warehouse for one department (e.g., a Finance data mart)
- **Grain:** the level of detail of a fact table (one row per shipment vs per day)
- **Conformed dimension:** a dimension shared across multiple fact tables (same Dim_Date everywhere)
- **Surrogate key:** warehouse-generated unique key (not the source's business key)
- **CDC (Change Data Capture):** detecting only changed rows in the source to load incrementally
- **Full load vs Incremental load:** load everything vs only new/changed data

---

## 8. The data flow you'll test (memorize)
```
Source (OLTP) → Staging → Transformations (business rules) → Dimension & Fact tables → BI reports
```
At EACH arrow you test: did the data move correctly, completely, and accurately?

---

## The sentence that shows you understand warehousing
> "A data warehouse integrates data from multiple OLTP sources via ETL into a star schema — fact tables with numeric measures surrounded by descriptive dimensions. My testing validates each ETL stage: extraction completeness, transformation business rules, and load accuracy — and I pay special attention to Slowly Changing Dimensions, especially Type 2, to confirm history is preserved correctly."
