"""
Streamlit — Full ML App
========================
pip install streamlit pandas numpy scikit-learn plotly
Run: streamlit run streamlit_everything.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)
import warnings
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════
# PAGE SETUP
# ════════════════════════════════════════════
st.set_page_config(
    page_title="ML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════
st.sidebar.title("🤖 ML Dashboard")
st.sidebar.markdown("---")

# Dataset selection
dataset_name = st.sidebar.selectbox(
    "Choose Dataset",
    ["Iris", "Breast Cancer"]
)

# Algorithm selection
algo = st.sidebar.selectbox(
    "Choose Algorithm",
    ["Random Forest", "Logistic Regression", "SVM"]
)

st.sidebar.markdown("### Hyperparameters")

# Dynamic hyperparameters
if algo == "Random Forest":
    n_estimators = st.sidebar.slider("n_estimators", 10, 300, 100)
    max_depth    = st.sidebar.slider("max_depth", 1, 20, 5)
    params = {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": 42}

elif algo == "Logistic Regression":
    C_val    = st.sidebar.slider("C (regularization)", 0.01, 10.0, 1.0)
    max_iter = st.sidebar.slider("max_iter", 100, 1000, 200)
    params = {"C": C_val, "max_iter": max_iter}

elif algo == "SVM":
    C_svm = st.sidebar.slider("C", 0.1, 10.0, 1.0)
    kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly"])
    params = {"C": C_svm, "kernel": kernel, "probability": True}

test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit ❤️")

# ════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════
@st.cache_data   # cache so it doesn't reload every time
def load_data(name):
    if name == "Iris":
        data = load_iris()
    else:
        data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    df["target_name"] = [data.target_names[t] for t in data.target]
    return df, data.feature_names, data.target_names

df, feature_names, target_names = load_data(dataset_name)

# ════════════════════════════════════════════
# MAIN CONTENT
# ════════════════════════════════════════════
st.title(f"🤖 ML Model: {algo} on {dataset_name}")
st.markdown("---")

# ── Tabs ──────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🔍 EDA", "🚀 Model", "🎯 Predict"])

# ── TAB 1: Data Overview ──────────────────
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
    fig = px.bar(class_counts, x="Class", y="Count",
                 color="Class", title="Class Distribution")
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 2: EDA ────────────────────────────
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

# ── TAB 3: Model ──────────────────────────
with tab3:
    @st.cache_resource
    def train_model(algo, params, test_size, dataset_name):
        df, feature_names, _ = load_data(dataset_name)
        X = df[list(feature_names)].values
        y = df["target"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        if algo == "Random Forest":
            model = RandomForestClassifier(**params)
        elif algo == "Logistic Regression":
            model = LogisticRegression(**params)
        else:
            model = SVC(**params)

        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        return model, scaler, X_test, y_test, y_pred, y_proba, feature_names

    with st.spinner("Training model..."):
        model, scaler, X_test, y_test, y_pred, y_proba, feat_names = train_model(
            algo, params, test_size, dataset_name
        )

    st.success("✅ Model trained!")

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy",    f"{acc:.4f}")
    col2.metric("Test Samples", len(y_test))
    col3.metric("Classes",      len(set(y_test)))

    # Confusion matrix
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(report_df, use_container_width=True)

    # Feature importance
    if hasattr(model, "feature_importances_"):
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            "Feature":    feat_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        fig = px.bar(importance_df, x="Importance", y="Feature",
                     orientation="h", title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)

# ── TAB 4: Predict ────────────────────────
with tab4:
    st.subheader("Make a Prediction")
    st.info("Adjust sliders to set feature values, then click Predict!")

    input_vals = {}
    cols = st.columns(3)
    for i, feat in enumerate(feature_names):
        with cols[i % 3]:
            min_v = float(df[feat].min())
            max_v = float(df[feat].max())
            mean_v = float(df[feat].mean())
            input_vals[feat] = st.slider(
                feat, min_value=min_v, max_value=max_v,
                value=mean_v, step=(max_v-min_v)/100
            )

    if st.button("🎯 Predict!", type="primary"):
        X_input = scaler.transform(np.array(list(input_vals.values())).reshape(1, -1))
        pred = model.predict(X_input)[0]
        df_data, _, target_names = load_data(dataset_name)

        st.success(f"### Prediction: **{target_names[pred]}** (class {pred})")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0]
            fig = px.bar(
                x=target_names, y=proba,
                labels={"x": "Class", "y": "Probability"},
                title="Prediction Confidence",
                color=target_names,
            )
            st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════
# HOW TO RUN
# ════════════════════════════════════════════
# streamlit run streamlit_everything.py
# Opens at: http://localhost:8501
#
# Streamlit Key Concepts:
# st.title(), st.header(), st.subheader() → text
# st.write(), st.markdown()               → markdown
# st.dataframe(), st.table()              → tables
# st.plotly_chart(), st.pyplot()          → charts
# st.sidebar.xxx                          → sidebar widget
# st.columns([1,2])                       → layout
# st.tabs(["A","B"])                      → tabs
# st.button(), st.slider(), st.selectbox()→ inputs
# st.metric("label", value)               → KPI card
# st.spinner(), st.success(), st.error()  → status
# @st.cache_data                          → cache data
# @st.cache_resource                      → cache models
# st.session_state                        → persist state across reruns
