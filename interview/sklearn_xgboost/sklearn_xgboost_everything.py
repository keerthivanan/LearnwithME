"""
Scikit-learn + XGBoost + LightGBM — Definitions + Code + Outputs
==================================================================
pip install scikit-learn xgboost lightgbm optuna joblib
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    GridSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════
# 1. DATA PREPARATION
# ══════════════════════════════════════════════════════
# WHAT IS DATA PREPARATION?
#   → Before training any model you need to:
#     1. Split data into train / test sets (never train on test data!)
#     2. Scale features so large numbers don't dominate small ones
#     3. Handle missing values (NaN)
#     4. Encode categorical columns (strings → numbers)
#
# train_test_split():
#   → Splits X and y into train/test simultaneously (keeps pairs aligned)
#   → test_size=0.2 → 20% for testing, 80% for training
#   → stratify=y    → ensures class distribution is same in train and test
#   → random_state  → makes the split reproducible
#
# StandardScaler:
#   → Transforms each feature to mean=0, std=1
#   → CRITICAL: fit ONLY on training data, then transform both train and test
#   → fit_transform(X_train) = learn stats + apply
#   → transform(X_test)      = apply learned stats (don't relearn!)
#   → WHY: prevents "data leakage" — test data must be completely unseen

print("=" * 55)
print("1. DATA PREPARATION")
print("=" * 55)

X_clf, y_clf = make_classification(
    n_samples=1000, n_features=20,
    n_informative=10, random_state=42
)
X_reg, y_reg = make_regression(
    n_samples=1000, n_features=10,
    noise=0.1, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)
print(f"Train shape: {X_train.shape}")   # (800, 20)
print(f"Test  shape: {X_test.shape}")    # (200, 20)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit + transform on TRAIN
X_test_sc  = scaler.transform(X_test)        # only transform on TEST (no fit!)


# ══════════════════════════════════════════════════════
# 2. LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════
# WHAT IS LOGISTIC REGRESSION?
#   → A CLASSIFICATION algorithm (despite the name "regression")
#   → Predicts the PROBABILITY that an input belongs to class 1
#   → Uses a sigmoid function: output is always between 0 and 1
#   → If probability > 0.5 → predict class 1, else class 0
#
# WHEN TO USE:
#   → Binary classification (spam/not-spam, fraud/legit)
#   → When you need INTERPRETABLE results (coefficients show feature importance)
#   → As a BASELINE before trying complex models
#
# KEY PARAMETER:
#   → C = inverse of regularization strength
#   → Small C → more regularization (simpler model, less overfitting)
#   → Large C → less regularization (complex model, may overfit)

print("\n" + "=" * 55)
print("2. LOGISTIC REGRESSION")
print("=" * 55)

from sklearn.linear_model import LogisticRegression, Ridge, Lasso

lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)
preds = lr.predict(X_test_sc)
proba = lr.predict_proba(X_test_sc)[:, 1]   # probability of class 1

print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
print(f"AUC-ROC:  {roc_auc_score(y_test, proba):.4f}")
print(classification_report(y_test, preds))

# Ridge (L2) and Lasso (L1) — for regression
# Ridge: penalizes large coefficients → shrinks them (good for multicollinearity)
# Lasso: forces some coefficients to ZERO → automatic feature selection
ridge = Ridge(alpha=1.0)    # alpha = regularization strength (like 1/C)
lasso = Lasso(alpha=0.01)
ridge.fit(X_train_sc, y_train[:800] if len(y_train) > 800 else y_train)


# ══════════════════════════════════════════════════════
# 3. DECISION TREE
# ══════════════════════════════════════════════════════
# WHAT IS A DECISION TREE?
#   → Splits data into branches based on feature thresholds
#   → Like a flowchart of if-else decisions
#   → Root node → split → branches → leaf nodes (final predictions)
#   → Splitting criterion: Gini impurity or Entropy (information gain)
#
# PROS:
#   → Easy to interpret and visualize
#   → No need to scale features
#   → Handles both numerical and categorical data
#
# CONS:
#   → Prone to OVERFITTING (memorizes training data)
#   → Unstable — small data changes create very different trees
#   → Solution: use Random Forest (many trees together)
#
# KEY PARAMS:
#   → max_depth      : max levels of splitting (controls overfitting)
#   → min_samples_leaf: min samples required in a leaf (smooths the tree)

print("\n" + "=" * 55)
print("3. DECISION TREE")
print("=" * 55)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
dt.fit(X_train, y_train)
print(f"Train Acc: {dt.score(X_train, y_train):.4f}")   # high — might overfit
print(f"Test  Acc: {dt.score(X_test, y_test):.4f}")     # if gap is large = overfitting


# ══════════════════════════════════════════════════════
# 4. RANDOM FOREST
# ══════════════════════════════════════════════════════
# WHAT IS A RANDOM FOREST?
#   → An ENSEMBLE of many Decision Trees trained on random subsets
#   → "Wisdom of the crowd" — many weak learners → one strong learner
#   → Each tree: trained on a BOOTSTRAP sample (random rows with replacement)
#   → Each split: considers only a RANDOM SUBSET of features (max_features)
#   → Prediction: majority vote (classification) or average (regression)
#
# WHY BETTER THAN ONE TREE:
#   → Individual trees overfit → but their errors are UNCORRELATED
#   → When you average uncorrelated errors, they cancel out
#   → Result: much lower variance than a single tree
#
# OOB SCORE (Out-Of-Bag):
#   → Each tree is trained on ~63% of data (bootstrap)
#   → The remaining ~37% (out-of-bag) is used as a FREE validation set
#   → oob_score=True gives you a validation score without using test data
#
# KEY PARAMS:
#   → n_estimators  : number of trees (more = better, but slower)
#   → max_features  : "sqrt" for classification, "log2" also common
#   → max_depth     : None = grow fully (Random Forest handles overfitting via ensemble)
#   → n_jobs=-1     : use all CPU cores

print("\n" + "=" * 55)
print("4. RANDOM FOREST")
print("=" * 55)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",   # at each split, try sqrt(n_features) features
    max_depth=None,        # grow full trees — ensemble handles variance
    oob_score=True,        # free validation via out-of-bag samples
    n_jobs=-1,             # parallel — use all CPU cores
    random_state=42
)
rf.fit(X_train, y_train)
print(f"OOB Score: {rf.oob_score_:.4f}")        # free validation score!
print(f"Test  Acc: {rf.score(X_test, y_test):.4f}")

# Feature importance — which features matter most?
importances = pd.Series(rf.feature_importances_).sort_values(ascending=False)
print(f"Top 5 feature indices: {importances.head(5).index.tolist()}")


# ══════════════════════════════════════════════════════
# 5. SVM (Support Vector Machine)
# ══════════════════════════════════════════════════════
# WHAT IS SVM?
#   → Finds the HYPERPLANE that best separates classes with MAXIMUM MARGIN
#   → Margin = distance between the hyperplane and nearest data points (support vectors)
#   → kernel="rbf" (Radial Basis Function): maps data to higher dimension for non-linear separation
#   → Effective in HIGH-DIMENSIONAL spaces (text classification, image classification)
#
# KEY PARAMS:
#   → C       : regularization (low C = wider margin, more misclassifications allowed)
#   → kernel  : "rbf" (most common), "linear", "poly"
#   → gamma   : "scale" = 1/(n_features * X.var()) — controls influence radius
#
# CONS:
#   → Slow on large datasets (O(n²) to O(n³))
#   → Needs feature scaling (use inside Pipeline)

print("\n" + "=" * 55)
print("5. SVM")
print("=" * 55)

from sklearn.svm import SVC

svm_pipe = Pipeline([
    ("scaler", StandardScaler()),       # SVM REQUIRES scaling
    ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True))
])
svm_pipe.fit(X_train, y_train)
print(f"SVM Accuracy: {svm_pipe.score(X_test, y_test):.4f}")


# ══════════════════════════════════════════════════════
# 6. KNN (K-Nearest Neighbors)
# ══════════════════════════════════════════════════════
# WHAT IS KNN?
#   → To predict a new point: find its K nearest neighbors, take majority vote
#   → "Lazy learner" — no training phase, just memorizes all data
#   → Prediction time = O(n * d) where n=samples, d=dimensions
#
# KEY PARAMS:
#   → n_neighbors : K (small K = complex boundary, large K = smoother)
#   → weights     : "uniform" = all neighbors equal, "distance" = closer = more weight
#
# CONS:
#   → SLOW for large datasets (checks every training point at prediction time)
#   → Sensitive to irrelevant features and different scales → MUST scale features

print("\n" + "=" * 55)
print("6. KNN")
print("=" * 55)

from sklearn.neighbors import KNeighborsClassifier

knn_pipe = Pipeline([
    ("scaler", StandardScaler()),      # KNN REQUIRES scaling
    ("knn", KNeighborsClassifier(n_neighbors=7, weights="distance"))
])
knn_pipe.fit(X_train, y_train)
print(f"KNN Accuracy: {knn_pipe.score(X_test, y_test):.4f}")


# ══════════════════════════════════════════════════════
# 7. NAIVE BAYES
# ══════════════════════════════════════════════════════
# WHAT IS NAIVE BAYES?
#   → Probabilistic classifier based on Bayes' Theorem
#   → "Naive" = assumes ALL features are INDEPENDENT (rarely true but works!)
#   → P(class | features) ∝ P(class) × P(feature1|class) × P(feature2|class) × ...
#   → GaussianNB: assumes features follow a normal (Gaussian) distribution
#   → MultinomialNB: for count data (word counts in text classification)
#
# PROS:
#   → Very fast to train and predict
#   → Works well with small data
#   → Great for TEXT classification (spam detection)
#
# CONS:
#   → Independence assumption is often wrong → lower accuracy than ensemble methods

print("\n" + "=" * 55)
print("7. NAIVE BAYES")
print("=" * 55)

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train, y_train)
print(f"Naive Bayes Accuracy: {nb.score(X_test, y_test):.4f}")


# ══════════════════════════════════════════════════════
# 8. GRADIENT BOOSTING
# ══════════════════════════════════════════════════════
# WHAT IS GRADIENT BOOSTING?
#   → Builds trees SEQUENTIALLY — each new tree corrects ERRORS of previous trees
#   → Trains on the RESIDUALS (mistakes) of previous ensemble
#   → Unlike Random Forest (parallel trees), Gradient Boosting is sequential
#
# GRADIENT BOOSTING vs RANDOM FOREST:
#   → Random Forest: parallel trees, reduces VARIANCE (overfitting)
#   → Gradient Boosting: sequential trees, reduces BIAS (underfitting)
#
# XGBoost / LightGBM = faster, optimized implementations of gradient boosting
#   → Add regularization (L1, L2) to prevent overfitting
#   → Early stopping — stop adding trees when validation score stops improving
#
# KEY PARAMS:
#   → n_estimators  : max trees (use with early stopping)
#   → learning_rate : how much each tree contributes (smaller = better, more trees needed)
#   → max_depth     : depth of each tree (3-6 is typical for boosting)
#   → subsample     : fraction of rows used per tree (adds randomness, reduces overfitting)

print("\n" + "=" * 55)
print("8. GRADIENT BOOSTING (sklearn)")
print("=" * 55)

from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1,
    max_depth=3, subsample=0.8, random_state=42
)
gb.fit(X_train, y_train)
print(f"GB Accuracy: {gb.score(X_test, y_test):.4f}")


# ══════════════════════════════════════════════════════
# 9. XGBOOST
# ══════════════════════════════════════════════════════
# WHAT IS XGBOOST?
#   → eXtreme Gradient Boosting — the most popular ML algorithm for tabular data
#   → Faster and more regularized than sklearn's GradientBoosting
#   → Dominated Kaggle competitions for years
#
# EARLY STOPPING:
#   → Monitor validation metric after each tree
#   → Stop adding trees when metric doesn't improve for N rounds
#   → Prevents overfitting AND saves training time
#   → Set n_estimators HIGH, let early stopping decide when to stop
#
# KEY PARAMS:
#   → n_estimators     : max trees (set high, use early_stopping_rounds)
#   → learning_rate    : smaller = more robust (0.01–0.1)
#   → max_depth        : tree depth (4–8 for XGBoost)
#   → subsample        : row sampling per tree (0.6–1.0)
#   → colsample_bytree : feature sampling per tree (0.6–1.0)
#   → reg_alpha (L1)   : sparsity regularization
#   → reg_lambda (L2)  : weight regularization

print("\n" + "=" * 55)
print("9. XGBOOST")
print("=" * 55)

import xgboost as xgb

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

xgb_model = xgb.XGBClassifier(
    n_estimators=1000,        # high — early stopping will find best
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,            # L1 regularization
    reg_lambda=1.0,           # L2 regularization
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,   # stop if no improvement for 50 rounds
    verbose=False
)
print(f"Best iteration: {xgb_model.best_iteration}")
print(f"XGB Accuracy:   {xgb_model.score(X_test, y_test):.4f}")
print(f"XGB AUC-ROC:    {roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1]):.4f}")


# ══════════════════════════════════════════════════════
# 10. LIGHTGBM
# ══════════════════════════════════════════════════════
# WHAT IS LIGHTGBM?
#   → Light Gradient Boosting Machine — by Microsoft
#   → FASTER than XGBoost (especially for large datasets)
#   → Grows trees LEAF-WISE (best leaf first) vs LEVEL-WISE (XGBoost)
#   → Leaf-wise: more accurate but can overfit → use min_child_samples
#
# LIGHTGBM vs XGBOOST:
#   → LightGBM: faster, better for large data, categorical support built-in
#   → XGBoost:  more stable, slightly better for small/medium data
#   → In practice: try both, pick the one with better CV score
#
# KEY PARAMS:
#   → num_leaves   : max leaves per tree (key param! use instead of max_depth)
#   → learning_rate: same as XGBoost
#   → subsample    : row sampling
#   → colsample_bytree: feature sampling

print("\n" + "=" * 55)
print("10. LIGHTGBM")
print("=" * 55)

import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,       # key param for LightGBM (default 31)
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
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
)
print(f"LGB Accuracy: {lgb_model.score(X_test, y_test):.4f}")
print(f"LGB AUC-ROC:  {roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:,1]):.4f}")


# ══════════════════════════════════════════════════════
# 11. CROSS-VALIDATION
# ══════════════════════════════════════════════════════
# WHAT IS CROSS-VALIDATION?
#   → A technique to evaluate model performance more RELIABLY than a single train/test split
#   → K-Fold CV: split data into K folds, train on K-1, test on 1, repeat K times
#   → Report: mean ± std of the K scores
#   → WHY: a single split can get lucky or unlucky with which data ends up in test
#
# STRATIFIED K-FOLD:
#   → Ensures each fold has the SAME class distribution as the full dataset
#   → ALWAYS use for classification to avoid folds with no positive class
#
# WHEN TO USE:
#   → When comparing models (pick one with best CV score)
#   → When data is limited (can't afford to "waste" data on a fixed test set)
#   → DO NOT use CV score as final test score — still need a held-out test set

print("\n" + "=" * 55)
print("11. CROSS-VALIDATION")
print("=" * 55)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in [("LogReg", lr), ("RandomForest", rf)]:
    scores = cross_val_score(model, X_clf, y_clf, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"{name:15s} AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    # ± shows stability — high std means model is sensitive to data split


# ══════════════════════════════════════════════════════
# 12. HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════
# WHAT IS HYPERPARAMETER TUNING?
#   → Finding the BEST configuration of model settings
#   → Hyperparameters = settings you choose BEFORE training (not learned from data)
#   → Examples: n_estimators, max_depth, learning_rate, C, kernel
#
# METHODS:
#   → GridSearchCV     : try ALL combinations — exhaustive but slow
#   → RandomizedSearchCV: try random combinations — faster, often good enough
#   → Optuna (Bayesian): learns from past trials, tries smarter next time ← BEST
#
# IMPORTANT: use CV inside tuning (e.g., cv=3) so each candidate is evaluated fairly

print("\n" + "=" * 55)
print("12. HYPERPARAMETER TUNING")
print("=" * 55)

# GridSearchCV — exhaustive (slow for large grids)
param_grid = {"n_estimators": [100, 200], "max_depth": [3, 5, None]}
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=3, scoring="roc_auc", n_jobs=-1
)
grid.fit(X_train, y_train)
print(f"Grid best params: {grid.best_params_}")
print(f"Grid best AUC:    {grid.best_score_:.4f}")

# Optuna — Bayesian optimization (BEST for expensive models)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        "n_estimators":  trial.suggest_int("n_estimators", 100, 500),
        "max_depth":     trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
    }
    model = xgb.XGBClassifier(**params, eval_metric="logloss", random_state=42)
    return cross_val_score(model, X_train, y_train, cv=3, scoring="roc_auc").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
print(f"Optuna best params: {study.best_params}")
print(f"Optuna best AUC:    {study.best_value:.4f}")


# ══════════════════════════════════════════════════════
# 13. FULL SKLEARN PIPELINE (Production Pattern)
# ══════════════════════════════════════════════════════
# WHAT IS A SKLEARN PIPELINE?
#   → Chains preprocessing steps + model into ONE object
#   → Benefit 1: prevents data leakage (scaler fit only on train, applied to test)
#   → Benefit 2: one object to .fit(), .predict(), .save()
#   → Benefit 3: works seamlessly with cross_val_score and GridSearchCV
#
# ColumnTransformer:
#   → Applies DIFFERENT transformations to DIFFERENT columns
#   → Numeric columns → impute + scale
#   → Categorical columns → impute + one-hot encode
#
# FLOW:
#   raw data → ColumnTransformer → model → predictions
#   (no manual scaling/encoding needed at prediction time!)

print("\n" + "=" * 55)
print("13. FULL SKLEARN PIPELINE")
print("=" * 55)

np.random.seed(42)
n = 500
df = pd.DataFrame({
    "age":    np.random.randint(18, 70, n),
    "salary": np.random.normal(50000, 15000, n),
    "city":   np.random.choice(["Mumbai", "Delhi", "Chennai"], n),
    "edu":    np.random.choice(["UG", "PG", "PhD"], n),
    "target": np.random.randint(0, 2, n)
})
df.loc[np.random.choice(n, 30), "salary"] = np.nan   # introduce missing values

X = df.drop("target", axis=1)
y = df["target"]
num_features = ["age", "salary"]
cat_features = ["city", "edu"]

# Numeric pipeline: fill NaN with median, then standardize
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

# Categorical pipeline: fill NaN with most common, then one-hot encode
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Combine: apply right pipeline to right columns
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# Full pipeline: preprocess → model
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
full_pipeline.fit(X_tr, y_tr)       # preprocessing + training in one call
print(f"Pipeline Accuracy: {full_pipeline.score(X_te, y_te):.4f}")

# Save and load — entire pipeline (preprocessing + model) saved as one file
import joblib
joblib.dump(full_pipeline, "model.pkl")
loaded = joblib.load("model.pkl")
print(f"Loaded  Accuracy: {loaded.score(X_te, y_te):.4f}")   # identical result

print("\nAll done! ✓")
