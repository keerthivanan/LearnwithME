# BI Tools — Power BI & QlikSense (for Testers)

The JD says "exposure to BI tools." You don't build dashboards — you **test** that reports show correct data. Know enough to talk credibly.

---

## 1. What BI tools do
**Business Intelligence (BI)** tools turn warehouse data into **dashboards, charts, and reports** for decision-makers. The data warehouse is the source; the BI tool is the front-end.

```
Data Warehouse → BI tool (Power BI / QlikSense) → Dashboards for business users
```

As a tester, you verify: **does the number on the dashboard match the data in the warehouse?**

---

## 2. Power BI (Microsoft)
- **Power BI Desktop:** build reports; **Power BI Service:** publish/share online
- Connects to data sources (SQL, Teradata, files), builds a **data model** (relationships between tables — like a star schema)
- **DAX (Data Analysis Expressions):** the formula language for measures/calculations
  ```
  Total Revenue = SUM(Sales[Amount])
  ```
- **Power Query (M language):** data transformation/cleaning before loading
- Visuals: bar, line, pie, KPI cards, matrices, slicers (filters)

---

## 3. QlikSense (Qlik)
- Similar idea: connect data, build interactive dashboards
- Famous for its **associative engine** (in-memory) — click any value and everything filters in context
- **Load Script** to bring in and transform data
- Uses **set analysis** and expressions for calculations
- Green/white/gray selection model (selected / associated / excluded)

---

## 4. How you TEST a BI report (the key part)

**1. Data accuracy (most important):**
Take a KPI on the dashboard (e.g., "Total Shipments = 12,540") and run the equivalent SQL against the warehouse:
```sql
SELECT COUNT(*) FROM fact_shipments WHERE ship_date BETWEEN ... AND ...;
```
The dashboard number MUST match the SQL number. Mismatch = a bug in the report's logic or data model.

**2. Filter / slicer testing:**
Apply a filter (region = "Asia") and confirm all visuals update correctly and consistently.

**3. Drill-down testing:**
Click a bar → does it drill to the right detail? Do totals still add up?

**4. Aggregation testing:**
Does the sum of the parts equal the displayed total? (Watch for double-counting from bad joins in the model.)

**5. Cross-visual consistency:**
Two charts using the same measure must show the same number.

**6. Formatting / UI:**
Dates, currency, decimals, labels correct. Numbers not truncated.

**7. Security / row-level security:**
A user only sees data they're allowed to (e.g., a regional manager sees only their region).

**8. Performance:**
Report loads within acceptable time.

---

## 5. Common BI report defects
- Dashboard total ≠ warehouse SQL total (wrong measure/filter logic)
- Double-counting from a many-to-many relationship in the model
- Filters not applying to all visuals
- Wrong date range / timezone
- Null/blank showing as 0 (or vice versa)
- Stale data (report not refreshed after ETL load)

---

## 6. The tester's golden BI technique
> "I validate every key metric on a dashboard by writing the equivalent SQL against the warehouse and confirming the numbers match exactly. Then I test filters, drill-downs, and aggregations for consistency, and check that the report refreshed after the latest ETL load. A dashboard is only as trustworthy as its numbers, so SQL reconciliation is my main BI test."

---

## 7. If asked "Power BI or QlikSense — which do you know?"
Be honest:
> "I've had exposure to [the one you've used] for testing — mainly validating that dashboard metrics reconcile with the warehouse data via SQL. The testing approach is the same across BI tools: reconcile the displayed numbers to source-of-truth SQL, then test filters, drill-downs, and refresh."

---

## Quick comparison
| | Power BI | QlikSense |
|--|----------|-----------|
| Vendor | Microsoft | Qlik |
| Formula language | DAX | Set analysis / expressions |
| Transform layer | Power Query (M) | Load script |
| Signature feature | Tight MS ecosystem, DAX | Associative in-memory engine |
| Both | Connect to warehouse, build interactive dashboards | Same |
