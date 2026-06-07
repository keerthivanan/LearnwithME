"""
Scikit-learn + XGBoost + LightGBM — Everything
================================================
pip install scikit-learn xgboost lightgbm optuna
"""

import numpy as np
import pandas as pd
from sklearn.datasets import (
    make_classification, make_regression,
    load_iris, load_breast_cancer, load_boston
)
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    GridSearchCV, RandomizedSearchCV, StratifiedKFold
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, LabelEncoder,
    OneHotEncoder, PolynomialFeatures
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════
# 1. DATA PREP
# ════════════════════════════════════════════
print("=" * 50)
print("1. DATA PREPARATION")
print("=" * 50)

# Classification dataset
X_clf, y_clf = make_classification(
    n_samples=1000, n_features=20,
    n_informative=10, random_state=42
)

# Regression dataset
X_reg, y_reg = make_regression(
    n_samples=1000, n_features=10,
    noise=0.1, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
# Train: (800, 20), Test: (200, 20)

# ════════════════════════════════════════════
# 2. LOGISTIC REGRESSION
# ════════════════════════════════════════════
print("\n" + "=" * 50)
print("2. LOGISTIC REGRESSION")
print("=" * 50)

from sklearn.linear_model import LogisticRegression, Ridge, Lasso

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)
preds = lr.predict(X_test_sc)
proba = lr.predict_proba(X_test_sc)[:, 1]

print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, preds))

# ════════════════════════════════════════════
# 3. DECISION TREE
# ════════════════════════════════════════════
print("=" * 50)
print("3. DECISION TREE")
print("=" * 50)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
dt.fit(X_train, y_train)
print(f"Train Acc: {dt.score(X_train, y_train):.4f}")
print(f"Test  Acc: {dt.score(X_test, y_test):.4f}")
# Train Acc: 0.9350  Test Acc: 0.8400  (gap = overfitting)

# ════════════════════════════════════════════
# 4. RANDOM FOREST
# ════════════════════════════════════════════
print("\n" + "=" * 50)
print("4. RANDOM FOREST")
print("=" * 50)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

rf = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    max_depth=None,
    min_samples_leaf=1,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
print(f"OOB Score: {rf.oob_score_:.4f}")    # free validation!
print(f"Test  Acc: {rf.score(X_test, y_test):.4f}")

# Feature importance
importances = pd.Series(rf.feature_importances_).sort_values(ascending=False)
print(f"\nTop 5 features: {importances.head(5).index.tolist()}")

# ════════════════════════════════════════════
# 5. SVM
# ════════════════════════════════════════════
print("\n" + "=" * 50)
print("5. SVM")
print("=" * 50)

from sklearn.svm import SVC, SVR

svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True))
])
svm_pipe.fit(X_train, y_train)
print(f"SVM Accuracy: {svm_pipe.score(X_test, y_test):.4f}")

# ════════════════════════════════════════════
# 6. KNN
# ════════════════════════════════════════════
print("\n" + "=" * 50)
print("6. KNN")
print("=" * 50)

from sklearn.neighbors import KNeighborsClassifier

knn_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=7, weights="distance"))
])
knn_pipe.fit(X_train, y_train)
print(f"KNN Accuracy: {knn_pipe.score(X_test, y_test):.4f}")

# ════════════════════════════════════════════
# 7. NAIVE BAYES
# ════════════════════════════════════════════
print("\n" + "=" * 50)
print("7. NAIVE BAYES")
print("=" * 50)

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train, y_train)
print(f"Naive Bayes Accuracy: {nb.score(X_test, y_test):.4f}")

# ════════════════════════════════════════════
# 8. GRADIENT BOOSTING
# ════════════════════════════════════════════
print("\n" + "=" * 50)
print("8. SKLEARN GRADIENT BOOSTING")
print("=" * 50)

from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1,
    max_depth=3, subsample=0.8, random_state=42
)
gb.fit(X_train, y_train)
print(f"GB Accuracy: {gb.score(X_test, y_test):.4f}")

# ════════════════════════════════════════════
# 9. XGBOOST
# ════════════════════════════════════════════
print("\n" + "=" * 50)
print("9. XGBOOST")
print("=" * 50)

import xgboost as xgb

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

xgb_model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False
)

print(f"Best iteration: {xgb_model.best_iteration}")
print(f"XGB Accuracy:   {xgb_model.score(X_test, y_test):.4f}")
print(f"XGB AUC-ROC:    {roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1]):.4f}")

# XGBoost Regression
xgb_reg = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.05,
    max_depth=5, random_state=42
)
X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
    *make_regression(n_samples=1000, n_features=10, random_state=42),
    test_size=0.2
)
xgb_reg.fit(X_tr2, y_tr2, eval_set=[(X_val2, y_val2)],
            early_stopping_rounds=50, verbose=False)
preds_reg = xgb_reg.predict(X_val2)
print(f"XGB Regression R²: {r2_score(y_val2, preds_reg):.4f}")

# ════════════════════════════════════════════
# 10. LIGHTGBM
# ════════════════════════════════════════════
print("\n" + "=" * 50)
print("10. LIGHTGBM")
print("=" * 50)

import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False),
               lgb.log_evaluation(period=-1)]
)

print(f"LGB Accuracy: {lgb_model.score(X_test, y_test):.4f}")
print(f"LGB AUC-ROC:  {roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:,1]):.4f}")

# ════════════════════════════════════════════
# 11. CROSS-VALIDATION
# ════════════════════════════════════════════
print("\n" + "=" * 50)
print("11. CROSS-VALIDATION")
print("=" * 50)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in [("LogReg", lr), ("RF", rf), ("XGB", xgb_model)]:
    scores = cross_val_score(model, X_clf, y_clf, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"{name:8s} AUC: {scores.mean():.4f} ± {scores.std():.4f}")

# ════════════════════════════════════════════
# 12. HYPERPARAMETER TUNING
# ════════════════════════════════════════════
print("\n" + "=" * 50)
print("12. HYPERPARAMETER TUNING")
print("=" * 50)

# Grid Search
param_grid = {"n_estimators": [100, 200], "max_depth": [3, 5, None]}
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=3, scoring="roc_auc", n_jobs=-1
)
grid.fit(X_train, y_train)
print(f"Grid Search best params: {grid.best_params_}")
print(f"Grid Search best AUC:   {grid.best_score_:.4f}")

# Optuna (Bayesian optimization — best)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        "n_estimators":  trial.suggest_int("n_estimators", 100, 500),
        "max_depth":     trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
    }
    model = xgb.XGBClassifier(**params, use_label_encoder=False,
                               eval_metric="logloss", random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=3,
                            scoring="roc_auc", n_jobs=-1).mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
print(f"\nOptuna best params: {study.best_params}")
print(f"Optuna best AUC:    {study.best_value:.4f}")

# ════════════════════════════════════════════
# 13. FULL PIPELINE (Production Ready)
# ════════════════════════════════════════════
print("\n" + "=" * 50)
print("13. FULL SKLEARN PIPELINE")
print("=" * 50)

# Simulate mixed data
np.random.seed(42)
n = 500
df = pd.DataFrame({
    "age":      np.random.randint(18, 70, n),
    "salary":   np.random.normal(50000, 15000, n),
    "city":     np.random.choice(["Mumbai", "Delhi", "Chennai"], n),
    "edu":      np.random.choice(["UG", "PG", "PhD"], n),
    "target":   np.random.randint(0, 2, n)
})
df.loc[np.random.choice(n, 30), "salary"] = np.nan   # add some nulls

X = df.drop("target", axis=1)
y = df["target"]

num_features = ["age", "salary"]
cat_features = ["city", "edu"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
full_pipeline.fit(X_tr, y_tr)
print(f"Pipeline Accuracy: {full_pipeline.score(X_te, y_te):.4f}")

# Save model
import joblib
joblib.dump(full_pipeline, "model.pkl")
loaded = joblib.load("model.pkl")
print(f"Loaded model Accuracy: {loaded.score(X_te, y_te):.4f}")
print("\nAll done! ✓")
