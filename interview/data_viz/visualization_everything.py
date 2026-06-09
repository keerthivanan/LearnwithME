"""
Matplotlib + Seaborn + Plotly — Definitions + Code + Outputs
=============================================================
pip install matplotlib seaborn plotly pandas numpy

LIBRARY COMPARISON:
  → Matplotlib : low-level, full control, static images (.png/.pdf)
                 verbose but flexible — you control everything
  → Seaborn    : built ON TOP of Matplotlib, high-level statistical plots
                 beautiful defaults, designed for DataFrames
  → Plotly     : interactive charts in the browser (.html)
                 hover tooltips, zoom, pan — great for dashboards

WHEN TO USE WHICH:
  → Matplotlib : scientific papers, custom layouts, publication plots
  → Seaborn    : quick exploratory data analysis (EDA), statistical plots
  → Plotly     : web dashboards, presentations, interactive exploration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

np.random.seed(42)
n = 300
df = pd.DataFrame({
    "age":       np.random.randint(20, 65, n),
    "salary":    np.random.normal(60000, 20000, n),
    "score":     np.random.uniform(0, 100, n),
    "dept":      np.random.choice(["Engineering", "Sales", "HR", "Marketing"], n),
    "gender":    np.random.choice(["M", "F"], n),
    "years_exp": np.random.randint(0, 20, n),
    "promoted":  np.random.randint(0, 2, n),
})
df["salary"] = df["salary"].clip(20000, 150000)


# ══════════════════════════════════════════════════════
# 1. MATPLOTLIB — Low-Level, Full Control
# ══════════════════════════════════════════════════════
# WHAT IS MATPLOTLIB?
#   → The foundational Python plotting library
#   → fig, ax = plt.subplots() → creates Figure (canvas) and Axes (plot area)
#   → ax.plot() / ax.scatter() / ax.bar() → draw on the axes
#
# KEY CONCEPTS:
#   → Figure : the whole image (canvas) — set size with figsize=(width, height)
#   → Axes   : one plot area within the figure — can have multiple
#   → plt.subplots(rows, cols) → grid of multiple axes
#   → plt.tight_layout() → auto-fix spacing between subplots
#   → plt.savefig("file.png", dpi=150) → save to file
#   → plt.close() → release memory (important in loops!)
#
# PLOT TYPES:
#   → ax.plot()      : line chart
#   → ax.scatter()   : scatter plot
#   → ax.bar()       : vertical bar chart
#   → ax.barh()      : horizontal bar chart
#   → ax.hist()      : histogram
#   → ax.boxplot()   : box plot
#   → ax.pie()       : pie chart
#   → ax.errorbar()  : line with error bars
#   → ax.fill_between(): shade area between two lines

print("=" * 55)
print("1. MATPLOTLIB")
print("=" * 55)

# Line Plot
fig, ax = plt.subplots(figsize=(10, 5))
x = np.linspace(0, 4*np.pi, 200)
ax.plot(x, np.sin(x), label="sin(x)", color="blue",  linewidth=2)
ax.plot(x, np.cos(x), label="cos(x)", color="red",   linewidth=2, linestyle="--")
ax.fill_between(x, np.sin(x), alpha=0.1, color="blue")  # shade under sin
ax.set_title("Sine and Cosine Waves", fontsize=14, fontweight="bold")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("01_line_plot.png", dpi=150)
plt.close()
print("  Saved: 01_line_plot.png")

# Multiple subplots — 2 rows × 3 columns
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Scatter — two continuous variables, color by a third
axes[0,0].scatter(df["age"], df["salary"], c=df["promoted"], cmap="coolwarm", alpha=0.6)
axes[0,0].set_title("Age vs Salary"); axes[0,0].set_xlabel("Age"); axes[0,0].set_ylabel("Salary")

# Histogram — distribution of one variable
# density=True → normalize to probability density (area under curve = 1)
axes[0,1].hist(df["salary"], bins=30, edgecolor="black", color="steelblue", density=True)
axes[0,1].set_title("Salary Distribution")

# Horizontal bar — good for long category names
dept_avg = df.groupby("dept")["salary"].mean().sort_values()
axes[0,2].barh(dept_avg.index, dept_avg.values, color="coral")
axes[0,2].set_title("Avg Salary by Dept")

# Box plot — shows median, quartiles, outliers
data_groups = [df[df["dept"]==d]["salary"].values for d in df["dept"].unique()]
axes[1,0].boxplot(data_groups, labels=df["dept"].unique())
axes[1,0].set_title("Salary Boxplot by Dept")
axes[1,0].tick_params(axis="x", rotation=30)

# Pie chart — proportions of categories
pie_data = df["dept"].value_counts()
axes[1,1].pie(pie_data, labels=pie_data.index, autopct="%1.1f%%", startangle=90)
axes[1,1].set_title("Dept Distribution")

# Error bars — show uncertainty/variance
x_vals = np.arange(5)
means  = [60000, 65000, 70000, 62000, 58000]
stds   = [5000, 6000, 4000, 5500, 4500]
axes[1,2].errorbar(x_vals, means, yerr=stds, fmt="o-", capsize=5, color="green")
axes[1,2].set_title("Salary with Error Bars")

plt.suptitle("All Matplotlib Plot Types", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("02_subplots.png", dpi=150)
plt.close()
print("  Saved: 02_subplots.png")

# Correlation Heatmap (manual with Matplotlib)
corr = df[["age", "salary", "score", "years_exp"]].corr()
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im)
ax.set_xticks(range(len(corr)));  ax.set_xticklabels(corr.columns, rotation=45)
ax.set_yticks(range(len(corr)));  ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=10)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("03_heatmap.png", dpi=150)
plt.close()
print("  Saved: 03_heatmap.png")


# ══════════════════════════════════════════════════════
# 2. SEABORN — Statistical Visualization
# ══════════════════════════════════════════════════════
# WHAT IS SEABORN?
#   → High-level wrapper around Matplotlib
#   → Works directly with DataFrames — just pass column names as strings
#   → Beautiful default themes and color palettes
#
# KEY FUNCTIONS:
#   → sns.histplot(data, x, kde=True) : histogram + optional KDE curve
#   → sns.boxplot(x, y, hue, data)    : box plots grouped by category
#   → sns.violinplot(x, y, data)      : violin = box + KDE distribution shape
#   → sns.regplot(x, y, data)         : scatter + linear regression line
#   → sns.heatmap(corr_matrix, annot=True) : correlation heatmap (best option!)
#   → sns.countplot(x, hue, data)     : count of categories
#   → sns.pairplot(df, hue)           : all pairs of features as scatter matrix
#
# hue:
#   → Color/split by a categorical column
#   → e.g., hue="gender" → separate boxplot for M and F
#
# WHEN TO USE VIOLIN vs BOX:
#   → Box plot   : shows median, quartiles, outliers — compact
#   → Violin plot: shows full distribution SHAPE — richer info but wider

print("\n" + "=" * 55)
print("2. SEABORN")
print("=" * 55)

sns.set_theme(style="whitegrid", palette="husl")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# histplot with KDE — distribution + density curve overlay
sns.histplot(df["salary"], kde=True, ax=axes[0,0], color="steelblue")
axes[0,0].set_title("Salary Distribution (KDE)")

# boxplot with hue — grouped by dept, colored by gender
sns.boxplot(x="dept", y="salary", hue="gender", data=df, ax=axes[0,1])
axes[0,1].set_title("Salary by Dept + Gender")
axes[0,1].tick_params(axis="x", rotation=30)

# violinplot — shows distribution shape (fatter = more data there)
sns.violinplot(x="dept", y="salary", data=df, ax=axes[0,2], palette="muted")
axes[0,2].set_title("Violin Plot — Distribution Shape")
axes[0,2].tick_params(axis="x", rotation=30)

# regplot — scatter + regression line + confidence interval
sns.regplot(x="age", y="salary", data=df, ax=axes[1,0], scatter_kws={"alpha":0.4})
axes[1,0].set_title("Age vs Salary + Regression Line")

# heatmap — correlation matrix visualization (best seaborn feature!)
sns.heatmap(df[["age","salary","score","years_exp"]].corr(),
            annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, ax=axes[1,1])
axes[1,1].set_title("Correlation Heatmap")

# countplot — count of categorical variable
sns.countplot(x="dept", hue="promoted", data=df, ax=axes[1,2])
axes[1,2].set_title("Promotion Count by Dept")
axes[1,2].tick_params(axis="x", rotation=30)

plt.suptitle("Seaborn Statistical Plots", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("04_seaborn.png", dpi=150)
plt.close()
print("  Saved: 04_seaborn.png")


# ══════════════════════════════════════════════════════
# 3. PLOTLY — Interactive Charts
# ══════════════════════════════════════════════════════
# WHAT IS PLOTLY?
#   → Creates INTERACTIVE charts rendered in the browser (HTML/JS)
#   → Hover over data points → see values
#   → Click/drag to zoom, pan, select, filter
#   → Save as .html → share with anyone
#
# PLOTLY EXPRESS (px) vs GRAPH_OBJECTS (go):
#   → px : high-level, one-liner charts — best for 90% of cases
#   → go : low-level, full control — needed for custom/complex charts
#
# KEY FUNCTIONS:
#   → px.scatter()   : interactive scatter
#   → px.bar()       : interactive bar chart
#   → px.box()       : interactive box plot
#   → px.histogram() : interactive histogram
#   → px.imshow()    : heatmap (great for correlation matrices)
#   → px.scatter_3d(): 3D scatter plot
#   → make_subplots(): multiple charts in one layout
#
# fig.write_html("file.html")  → save as interactive HTML
# fig.show()                   → open in browser (Jupyter: renders inline)

print("\n" + "=" * 55)
print("3. PLOTLY — INTERACTIVE CHARTS")
print("=" * 55)

# Scatter — hover shows age, salary, dept, years_exp
fig = px.scatter(
    df, x="age", y="salary",
    color="dept", size="score",
    hover_data=["years_exp", "gender"],
    title="Interactive: Age vs Salary (hover for details!)",
    template="plotly_white"
)
fig.write_html("06_scatter_plotly.html")
print("  Saved: 06_scatter_plotly.html")

# Bar chart with error bars
dept_stats = df.groupby("dept")["salary"].agg(["mean","std"]).reset_index()
fig = px.bar(dept_stats, x="dept", y="mean", error_y="std",
             color="dept", title="Avg Salary by Department (with std deviation)",
             labels={"mean": "Average Salary"})
fig.write_html("07_bar_plotly.html")
print("  Saved: 07_bar_plotly.html")

# Box plot — interactive with outlier points
fig = px.box(df, x="dept", y="salary", color="gender",
             title="Salary Distribution by Dept & Gender", points="outliers")
fig.write_html("08_box_plotly.html")
print("  Saved: 08_box_plotly.html")

# Correlation heatmap — easiest way with plotly
corr = df[["age","salary","score","years_exp"]].corr()
fig  = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu",
                 title="Correlation Heatmap (interactive)")
fig.write_html("09_heatmap_plotly.html")
print("  Saved: 09_heatmap_plotly.html")

# Histogram with marginal box plot
fig = px.histogram(df, x="salary", nbins=30, color="dept",
                   marginal="box",   # adds box plot above the histogram
                   title="Salary Histogram with Marginal Box")
fig.write_html("10_histogram_plotly.html")
print("  Saved: 10_histogram_plotly.html")

# Multi-subplot dashboard — make_subplots for complex layouts
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=["Scatter: Age vs Salary", "Histogram: Salary", "Box: Salary by Dept", "Line: Random Walk"]
)
fig.add_trace(go.Scatter(x=df["age"], y=df["salary"], mode="markers",
                          marker=dict(color=df["promoted"], colorscale="Viridis"),
                          name="Scatter"), row=1, col=1)
fig.add_trace(go.Histogram(x=df["salary"], name="Salary", nbinsx=30), row=1, col=2)
fig.add_trace(go.Box(y=df["salary"], x=df["dept"], name="Box"), row=2, col=1)
fig.add_trace(go.Scatter(x=np.arange(100), y=np.cumsum(np.random.randn(100)),
                          mode="lines", name="Random Walk"), row=2, col=2)
fig.update_layout(height=700, title_text="Interactive Multi-Panel Dashboard")
fig.write_html("11_dashboard_plotly.html")
print("  Saved: 11_dashboard_plotly.html  ← open this in browser!")

# 3D Scatter — great for visualizing 3 features simultaneously
fig = px.scatter_3d(df, x="age", y="years_exp", z="salary",
                    color="dept", title="3D Scatter: Age, Experience, Salary")
fig.write_html("12_3d_scatter_plotly.html")
print("  Saved: 12_3d_scatter_plotly.html")

print("\nAll plots saved!")
print("  .png files → view as images")
print("  .html files → open in browser for interactive charts")
print("\nAll done! ✓")
