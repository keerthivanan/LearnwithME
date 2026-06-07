"""
Matplotlib + Seaborn + Plotly — Everything
============================================
pip install matplotlib seaborn plotly pandas numpy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Sample data
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

# ════════════════════════════════════════════
# 1. MATPLOTLIB
# ════════════════════════════════════════════
print("1. MATPLOTLIB")

# ── Line Plot ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x = np.linspace(0, 4*np.pi, 200)
ax.plot(x, np.sin(x), label="sin(x)", color="blue",  linewidth=2)
ax.plot(x, np.cos(x), label="cos(x)", color="red",   linewidth=2, linestyle="--")
ax.fill_between(x, np.sin(x), alpha=0.1, color="blue")
ax.set_title("Sine and Cosine Waves", fontsize=14, fontweight="bold")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("01_line_plot.png", dpi=150)
plt.close(); print("  Saved: 01_line_plot.png")

# ── Subplots ──────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Scatter
axes[0,0].scatter(df["age"], df["salary"], c=df["promoted"], cmap="coolwarm", alpha=0.6)
axes[0,0].set_title("Age vs Salary"); axes[0,0].set_xlabel("Age"); axes[0,0].set_ylabel("Salary")

# Histogram
axes[0,1].hist(df["salary"], bins=30, edgecolor="black", color="steelblue", density=True)
axes[0,1].set_title("Salary Distribution")

# Bar
dept_avg = df.groupby("dept")["salary"].mean().sort_values()
axes[0,2].barh(dept_avg.index, dept_avg.values, color="coral")
axes[0,2].set_title("Avg Salary by Dept")

# Box
data_groups = [df[df["dept"]==d]["salary"].values for d in df["dept"].unique()]
axes[1,0].boxplot(data_groups, labels=df["dept"].unique())
axes[1,0].set_title("Salary Boxplot by Dept")
axes[1,0].tick_params(axis="x", rotation=30)

# Pie
pie_data = df["dept"].value_counts()
axes[1,1].pie(pie_data, labels=pie_data.index, autopct="%1.1f%%", startangle=90)
axes[1,1].set_title("Dept Distribution")

# Line with error bars
x_vals = np.arange(5)
means  = [60000, 65000, 70000, 62000, 58000]
stds   = [5000, 6000, 4000, 5500, 4500]
axes[1,2].errorbar(x_vals, means, yerr=stds, fmt="o-", capsize=5, color="green")
axes[1,2].set_title("Salary with Error Bars")

plt.suptitle("All Plot Types", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("02_subplots.png", dpi=150)
plt.close(); print("  Saved: 02_subplots.png")

# ── Heatmap (Matplotlib) ─────────────────
corr = df[["age", "salary", "score", "years_exp"]].corr()
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im)
ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=45)
ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=10)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("03_heatmap.png", dpi=150)
plt.close(); print("  Saved: 03_heatmap.png")

# ════════════════════════════════════════════
# 2. SEABORN
# ════════════════════════════════════════════
print("\n2. SEABORN")
sns.set_theme(style="whitegrid", palette="husl")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Distribution
sns.histplot(df["salary"], kde=True, ax=axes[0,0], color="steelblue")
axes[0,0].set_title("Salary Distribution (KDE)")

# Box plot
sns.boxplot(x="dept", y="salary", hue="gender", data=df, ax=axes[0,1])
axes[0,1].set_title("Salary by Dept + Gender")
axes[0,1].tick_params(axis="x", rotation=30)

# Violin
sns.violinplot(x="dept", y="salary", data=df, ax=axes[0,2], palette="muted")
axes[0,2].set_title("Violin Plot")
axes[0,2].tick_params(axis="x", rotation=30)

# Scatter with regression
sns.regplot(x="age", y="salary", data=df, ax=axes[1,0], scatter_kws={"alpha":0.4})
axes[1,0].set_title("Age vs Salary with Regression")

# Heatmap
sns.heatmap(df[["age","salary","score","years_exp"]].corr(),
            annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, ax=axes[1,1])
axes[1,1].set_title("Correlation Heatmap")

# Count plot
sns.countplot(x="dept", hue="promoted", data=df, ax=axes[1,2])
axes[1,2].set_title("Promotion by Dept")
axes[1,2].tick_params(axis="x", rotation=30)

plt.suptitle("Seaborn Plots", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("04_seaborn.png", dpi=150)
plt.close(); print("  Saved: 04_seaborn.png")

# Pair Plot
pair_df = df[["age", "salary", "score", "years_exp", "promoted"]].copy()
pair_df["promoted"] = pair_df["promoted"].map({0:"No", 1:"Yes"})
# sns.pairplot(pair_df, hue="promoted", diag_kind="kde")  # saves time — uncomment to run
# plt.savefig("05_pairplot.png", dpi=100); plt.close()

# ════════════════════════════════════════════
# 3. PLOTLY (Interactive)
# ════════════════════════════════════════════
print("\n3. PLOTLY")

# Scatter with hover info
fig = px.scatter(
    df, x="age", y="salary",
    color="dept", size="score",
    hover_data=["years_exp", "gender"],
    title="Interactive: Age vs Salary",
    template="plotly_white"
)
fig.write_html("06_scatter_plotly.html")
print("  Saved: 06_scatter_plotly.html")

# Bar chart
dept_stats = df.groupby("dept")["salary"].agg(["mean","std"]).reset_index()
fig = px.bar(dept_stats, x="dept", y="mean", error_y="std",
             color="dept", title="Avg Salary by Department",
             labels={"mean":"Average Salary"})
fig.write_html("07_bar_plotly.html")
print("  Saved: 07_bar_plotly.html")

# Box plot
fig = px.box(df, x="dept", y="salary", color="gender",
             title="Salary Distribution", points="outliers")
fig.write_html("08_box_plotly.html")

# Heatmap
corr = df[["age","salary","score","years_exp"]].corr()
fig  = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu",
                 title="Correlation Heatmap")
fig.write_html("09_heatmap_plotly.html")

# Histogram with KDE
fig = px.histogram(df, x="salary", nbins=30, color="dept",
                   marginal="box", title="Salary Histogram")
fig.write_html("10_histogram_plotly.html")

# Multi-subplot
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=["Scatter", "Histogram", "Box", "Line"]
)
fig.add_trace(go.Scatter(x=df["age"], y=df["salary"], mode="markers",
                          marker=dict(color=df["promoted"], colorscale="Viridis"),
                          name="Scatter"), row=1, col=1)
fig.add_trace(go.Histogram(x=df["salary"], name="Salary", nbinsx=30), row=1, col=2)
fig.add_trace(go.Box(y=df["salary"], x=df["dept"], name="Box"), row=2, col=1)
fig.add_trace(go.Scatter(x=np.arange(100), y=np.cumsum(np.random.randn(100)),
                          mode="lines", name="Random Walk"), row=2, col=2)
fig.update_layout(height=700, title_text="Multi-Panel Dashboard")
fig.write_html("11_dashboard_plotly.html")
print("  Saved: 11_dashboard_plotly.html (interactive dashboard!)")

# 3D Scatter
fig = px.scatter_3d(df, x="age", y="years_exp", z="salary",
                    color="dept", title="3D Scatter Plot")
fig.write_html("12_3d_scatter_plotly.html")
print("  Saved: 12_3d_scatter_plotly.html")

print("\nAll plots saved! Open the .html files in browser for interactive charts.")
print("All done! ✓")
