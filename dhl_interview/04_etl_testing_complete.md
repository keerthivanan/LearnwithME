# ETL Testing — Complete Guide

This is the core of Track A. Know every type of ETL test and how you'd do it.

---

## 1. What is ETL Testing?
Verifying that data is **correctly Extracted, Transformed, and Loaded** from source systems into the data warehouse — that it's **complete, accurate, consistent, and not corrupted**. You mostly test with **SQL** comparing source vs target.

**Key difference from app testing:** you're testing **data**, not screens. The "bug" is wrong/missing/duplicated data.

---

## 2. The types of ETL testing (know all of these)

| Test type | What it checks | How |
|-----------|---------------|-----|
| **Data Completeness** | All expected records loaded, none lost | Row counts source vs target; MINUS both ways |
| **Data Accuracy / Transformation** | Business rules applied correctly | Compare transformed target vs expected calculation |
| **Data Integrity** | Relationships valid, no orphans | Foreign key checks, referential integrity |
| **Data Quality** | No nulls/duplicates/invalid formats | IS NULL, GROUP BY HAVING, format checks |
| **Metadata testing** | Column names, data types, lengths, constraints match spec | Compare DDL / schema |
| **Duplicate check** | No unexpected duplicates | GROUP BY key HAVING COUNT(*)>1 |
| **Null check** | Required fields not null | WHERE col IS NULL |
| **Data transformation** | Each mapping rule correct | Query per business rule |
| **Incremental load testing** | Only new/changed rows loaded; no re-load of old | Compare before/after, check CDC |
| **Regression testing** | New ETL changes didn't break existing data | Re-run validation suite |
| **Performance testing** | ETL loads within the time window (SLA) | Measure load duration |

---

## 3. The Source-to-Target Mapping (S2T / mapping document)
The **most important document** in ETL testing. It says, for every target column:
- Which source column it comes from
- What transformation rule applies (e.g., "TRIM and UPPERCASE", "amount × 1.1", "concatenate first+last name")
- Data type, length, nullability

> **You test against the mapping document.** For each row in it, you write a query proving the rule worked. If asked "where do you start ETL testing?" → "from the source-to-target mapping document."

---

## 4. The ETL testing process (steps)
1. **Understand requirements** + the **S2T mapping** + business rules
2. **Review the data model** (source schema, target star schema)
3. **Write test cases** (one per mapping rule + completeness + integrity)
4. **Set up test data** in source
5. **Run the ETL job**
6. **Validate** with SQL: counts, MINUS, transformation rules, nulls, duplicates, integrity
7. **Log defects** (with the failing query + expected vs actual)
8. **Retest** after fix, **regression** test the rest

---

## 5. The core validation queries (your toolkit)

**Completeness — counts:**
```sql
SELECT COUNT(*) FROM source;   SELECT COUNT(*) FROM target;
```

**Completeness — actual rows (two-way):**
```sql
SELECT key, col1, col2 FROM source
MINUS
SELECT key, col1, col2 FROM target;   -- + the reverse direction
```

**Transformation rule (e.g., target.full_name = source.first + ' ' + last):**
```sql
SELECT s.id FROM source s JOIN target t ON s.id = t.id
WHERE t.full_name <> TRIM(s.first_name) || ' ' || TRIM(s.last_name);
```

**Duplicates:**
```sql
SELECT key, COUNT(*) FROM target GROUP BY key HAVING COUNT(*) > 1;
```

**Nulls in required column:**
```sql
SELECT COUNT(*) FROM target WHERE required_col IS NULL;
```

**Referential integrity (orphans):**
```sql
SELECT * FROM fact_shipments
WHERE customer_key NOT IN (SELECT customer_key FROM dim_customer);
```

---

## 6. Data integrity — explain it well
Data integrity = the data is **accurate, consistent, and trustworthy** throughout its lifecycle. Types:
- **Entity integrity:** primary key is unique and not null
- **Referential integrity:** foreign keys point to existing parent rows (no orphans)
- **Domain integrity:** values are valid for the column (date is a real date, status is in the allowed list)

> For DHL: if a shipment row references a customer that doesn't exist (broken referential integrity), reports and billing break. That's why integrity testing matters.

---

## 7. Common ETL defects you find (have examples ready)
- Records dropped during transformation (counts don't match)
- Duplicates from a bad join (one-to-many blew up the row count)
- Truncated data (target column too short)
- Wrong data type / format (date as string, timezone shift)
- NULLs where defaults expected
- Transformation rule applied wrong (rounding, currency conversion)
- SCD Type 2 not preserving history (old row overwritten instead of new row added)
- Trailing spaces / case mismatches not standardized

---

## 8. ETL testing challenges (good to mention)
- **Huge data volumes** → can't eyeball; use SQL + sampling + automated comparison
- **Complex transformations** → need the mapping doc + business SME
- **Source data quality** → garbage in; decide what's a source issue vs ETL bug
- **Heterogeneous sources** → different formats to reconcile

---

## 9. Manual vs automated ETL testing
- **Manual:** write and run SQL queries by hand (most common)
- **Automated:** tools/scripts that run the comparison suite automatically — e.g., **QuerySurge**, **Informatica DVO**, custom Python/SQL frameworks, or a set of saved SQL with a runner. Mention you can automate the regression suite so every ETL run is validated.

---

## The sentence that wins the ETL section
> "I start from the source-to-target mapping document and write a test case per rule. My validation always covers completeness — row counts and a two-way MINUS — accuracy of each transformation, integrity via foreign-key and orphan checks, and quality via null and duplicate checks. For SCD Type 2 dimensions I confirm history is preserved. I automate the regression suite so every load is validated, not just the first one."
