"""
Streamlit — Definitions + Full ML App
=======================================
pip install streamlit pandas numpy scikit-learn plotly
Run: streamlit run streamlit_everything.py
Opens at: http://localhost:8501

WHAT IS STREAMLIT?
  → Python library to build data apps and ML dashboards WITHOUT frontend skills
  → Just write Python — Streamlit creates a web UI automatically
  → Every time user changes a widget, the ENTIRE script reruns from top
  → No HTML/CSS/JavaScript needed
  → Use for: ML model demos, dashboards, data exploration tools

STREAMLIT CORE CONCEPTS:
  → Script reruns on every interaction (widget change, button click)
  → @st.cache_data    : cache function results — don't re-load data on every rerun
  → @st.cache_resource: cache heavy objects (models, DB connections)
  → st.session_state  : persist state across reruns (like a global dict)

QUICK REFERENCE:
  st.title() / st.header() / st.subheader()  → headings
  st.write() / st.markdown()                 → text / markdown
  st.dataframe() / st.table()                → show DataFrames
  st.plotly_chart() / st.pyplot()            → show charts
  st.sidebar.xxx                             → sidebar widget
  st.columns([1, 2])                         → side-by-side layout
  st.tabs(["A", "B"])                        → tab layout
  st.button() / st.slider() / st.selectbox() → input widgets
  st.metric("label", value)                  → KPI card (big number)
  st.spinner() / st.success() / st.error()   → status messages
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════
# PAGE SETUP
# ══════════════════════════════════════════════════════
# WHAT IS page_config?
#   → Configures the browser tab and overall page layout
#   → Must be the FIRST Streamlit call in your script
#   → layout="wide"       : use full browser width
#   → layout="centered"   : narrow centered column (default)
#   → initial_sidebar_state: "expanded" or "collapsed"

st.set_page_config(
    page_title="ML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ══════════════════════════════════════════════════════
# SIDEBAR — Controls Panel
# ══════════════════════════════════════════════════════
# WHAT IS THE SIDEBAR?
#   → Left panel that's always visible — perfect for controls/filters
#   → Use: st.sidebar.xxx for any widget — appears in sidebar
#   → st.sidebar.selectbox, .slider, .button, etc.
#
# DYNAMIC WIDGETS:
#   → Widget values are returned directly as Python variables
#   → if algo == "Random Forest": show RF-specific sliders
#   → Each rerun reflects the current widget state

st.sidebar.title("🤖 ML Dashboard")
st.sidebar.markdown("---")

dataset_name = st.sidebar.selectbox("Choose Dataset", ["Iris", "Breast Cancer"])
algo         = st.sidebar.selectbox("Choose Algorithm", ["Random Forest", "Logistic Regression", "SVM"])

st.sidebar.markdown("### Hyperparameters")

if algo == "Random Forest":
    n_estimators = st.sidebar.slider("n_estimators", 10, 300, 100)
    max_depth    = st.sidebar.slider("max_depth", 1, 20, 5)
    params = {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": 42}

elif algo == "Logistic Regression":
    C_val    = st.sidebar.slider("C (regularization)", 0.01, 10.0, 1.0)
    max_iter = st.sidebar.slider("max_iter", 100, 1000, 200)
    params   = {"C": C_val, "max_iter": max_iter}

else:  # SVM
    C_svm  = st.sidebar.slider("C", 0.1, 10.0, 1.0)
    kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly"])
    params = {"C": C_svm, "kernel": kernel, "probability": True}

test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit")


# ══════════════════════════════════════════════════════
# DATA LOADING WITH CACHING
# ══════════════════════════════════════════════════════
# WHAT IS @st.cache_data?
#   → Cache the return value of a function
#   → First call: runs the function, caches the result
#   → Subsequent calls with SAME args: returns cached result instantly
#   → Critical for performance — don't reload large datasets on every rerun!
#
# @st.cache_resource:
#   → For objects that should be shared (ML models, DB connections)
#   → Created once, reused across all users and reruns

@st.cache_data
def load_data(name):
    data = load_iris() if name == "Iris" else load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"]      = data.target
    df["target_name"] = [data.target_names[t] for t in data.target]
    return df, data.feature_names, data.target_names

df, feature_names, target_names = load_data(dataset_name)


# ══════════════════════════════════════════════════════
# MAIN CONTENT — TABS
# ══════════════════════════════════════════════════════
# WHAT IS st.tabs()?
#   → Creates a tabbed layout — each tab shows different content
#   → Use with 'with' context manager: with tab1: st.write(...)
#
# WHAT IS st.columns()?
#   → Creates a column layout — put widgets/charts side by side
#   → st.columns(3) → three equal-width columns
#   → st.columns([2,1]) → first column twice as wide

st.title(f"🤖 ML Model: {algo} on {dataset_name}")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🔍 EDA", "🚀 Model", "🎯 Predict"])

# ── TAB 1: Data Overview ─────────────────────────────
# st.metric() → shows a big KPI number (label, value, delta)
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows",     df.shape[0])
    col2.metric("Features", df.shape[1] - 2)
    col3.metric("Classes",  len(target_names))
    col4.metric("Missing",  df.isnull().sum().sum())

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Class Distribution")
    class_counts = df["target_name"].value_counts().reset_index()
    class_counts.columns = ["Class", "Count"]
    fig = px.bar(class_counts, x="Class", y="Count", color="Class", title="Class Distribution")
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 2: EDA (Exploratory Data Analysis) ───────────
with tab2:
    st.subheader("Correlation Heatmap")
    corr = df[list(feature_names)].corr()
    fig  = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu",
                     title="Feature Correlations")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        x_feat = st.selectbox("X Feature", list(feature_names), index=0)
    with col2:
        y_feat = st.selectbox("Y Feature", list(feature_names), index=1)

    fig = px.scatter(df, x=x_feat, y=y_feat, color="target_name",
                     title=f"{x_feat} vs {y_feat}", symbol="target_name")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Distributions")
    feat_sel = st.multiselect("Select Features", list(feature_names), default=list(feature_names)[:3])
    if feat_sel:
        fig = px.histogram(df[feat_sel + ["target_name"]].melt(id_vars="target_name"),
                           x="value", color="target_name", facet_col="variable",
                           title="Feature Distributions by Class")
        st.plotly_chart(fig, use_container_width=True)

# ── TAB 3: Model Training & Evaluation ───────────────
# @st.cache_resource: cache the trained model
# st.spinner(): show loading message while training
with tab3:
    @st.cache_resource
    def train_model(algo_name, params_dict, ts, ds_name):
        df_tr, feat_names, _ = load_data(ds_name)
        X = df_tr[list(feat_names)].values
        y = df_tr["target"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=ts, random_state=42, stratify=y
        )
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test  = sc.transform(X_test)

        if algo_name == "Random Forest":
            mdl = RandomForestClassifier(**params_dict)
        elif algo_name == "Logistic Regression":
            mdl = LogisticRegression(**params_dict)
        else:
            mdl = SVC(**params_dict)

        mdl.fit(X_train, y_train)
        y_pred  = mdl.predict(X_test)
        y_proba = mdl.predict_proba(X_test) if hasattr(mdl, "predict_proba") else None
        return mdl, sc, X_test, y_test, y_pred, y_proba, feat_names

    with st.spinner("Training model..."):
        model, scaler, X_test, y_test, y_pred, y_proba, feat_names = train_model(
            algo, params, test_size, dataset_name
        )

    st.success("✅ Model trained!")

    acc = accuracy_score(y_test, y_pred)
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy",     f"{acc:.4f}")
    col2.metric("Test Samples", len(y_test))
    col3.metric("Classes",      len(set(y_test)))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm  = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Classification Report")
        report    = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(report_df, use_container_width=True)

    if hasattr(model, "feature_importances_"):
        st.subheader("Feature Importance")
        imp_df = pd.DataFrame({
            "Feature":    feat_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        fig = px.bar(imp_df, x="Importance", y="Feature",
                     orientation="h", title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)

# ── TAB 4: Live Prediction ───────────────────────────
# st.button(): returns True only on the click that triggered the rerun
with tab4:
    st.subheader("Make a Prediction")
    st.info("Adjust sliders to set feature values, then click Predict!")

    input_vals = {}
    cols = st.columns(3)
    for i, feat in enumerate(feature_names):
        with cols[i % 3]:
            min_v  = float(df[feat].min())
            max_v  = float(df[feat].max())
            mean_v = float(df[feat].mean())
            input_vals[feat] = st.slider(
                feat, min_value=min_v, max_value=max_v,
                value=mean_v, step=(max_v - min_v) / 100
            )

    if st.button("🎯 Predict!", type="primary"):
        X_input = scaler.transform(np.array(list(input_vals.values())).reshape(1, -1))
        pred    = model.predict(X_input)[0]
        _, _, tnames = load_data(dataset_name)
        st.success(f"### Prediction: **{tnames[pred]}** (class {pred})")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0]
            fig   = px.bar(x=tnames, y=proba,
                           labels={"x": "Class", "y": "Probability"},
                           title="Prediction Confidence", color=tnames)
            st.plotly_chart(fig, use_container_width=True)
