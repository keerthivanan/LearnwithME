# Model Evaluation & Optimization

---

## Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
```

| Metric | Formula | Use When |
|--------|---------|---------|
| MSE | mean((y - ŷ)²) | Penalizes large errors more |
| RMSE | √MSE | Interpretable (same unit as target) |
| MAE | mean(\|y - ŷ\|) | Outliers present, want robust metric |
| R² | 1 - SS_res/SS_tot | % of variance explained. 1 = perfect, 0 = useless |

---

## Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)

print(classification_report(y_test, y_pred))
```

### Confusion Matrix
```
                Predicted No   Predicted Yes
Actual No:        TN               FP  ← False Alarm
Actual Yes:       FN  ← Missed    TP
```

### Metrics Explained Simply

| Metric | Formula | Use When |
|--------|---------|---------|
| Accuracy | (TP+TN) / all | Balanced classes only |
| Precision | TP / (TP+FP) | FP is costly (spam filter — don't block real emails) |
| Recall | TP / (TP+FN) | FN is costly (cancer — don't miss a case) |
| F1 | 2×P×R / (P+R) | Imbalanced data, need balance of precision & recall |
| AUC-ROC | Area under ROC | Comparing models, threshold-independent |

### The Golden Rule
> **Never use accuracy for imbalanced classes.** If 99% is class 0, predicting always 0 gives 99% accuracy but is useless.  
> Use F1, AUC-ROC, or PR-AUC instead.

```python
# ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])

# Tune threshold (default 0.5 is not always best)
# High recall needed → lower threshold
# High precision needed → raise threshold
y_pred_custom = (y_proba[:, 1] >= 0.3).astype(int)
```

---

## Cross-Validation

### Why?
A single train/test split can be lucky or unlucky. CV gives a more reliable estimate.

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='f1')

print(f"F1: {scores.mean():.3f} ± {scores.std():.3f}")
```

### Types
- **K-Fold**: split data into K folds, train on K-1, test on 1. Repeat K times.
- **Stratified K-Fold**: keeps class ratio in each fold (use for classification)
- **Leave-One-Out (LOO)**: extreme — each sample is the test set once. Expensive.
- **Time Series CV**: always train on past, test on future (never shuffle!)

---

## Overfitting vs Underfitting

```
Underfitting (High Bias):
  Training score low → Test score low
  Fix: More complex model, more features, less regularization

Overfitting (High Variance):
  Training score high → Test score much lower
  Fix: More data, simpler model, regularization, dropout, early stopping
```

```python
# Learning curve to diagnose
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
plt.legend()
# If train >> val → overfitting
# Both low → underfitting
```

---

## Regularization

Adds a penalty to large model weights → prevents overfitting.

| Type | Formula | Effect |
|------|---------|--------|
| L2 (Ridge) | loss + λ × Σw² | Shrinks all weights toward 0 |
| L1 (Lasso) | loss + λ × Σ\|w\| | Some weights become exactly 0 (feature selection!) |
| Elastic Net | L1 + L2 combined | Best of both |

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

ridge = Ridge(alpha=1.0)    # alpha = λ, higher = more regularization
lasso = Lasso(alpha=0.1)    # sparse model, drops unimportant features
```

---

## Hyperparameter Tuning

### Grid Search
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
```

### Random Search (Faster)
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'max_depth': randint(3, 10),
    'n_estimators': randint(100, 1000),
    'learning_rate': uniform(0.01, 0.3)
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,       # number of combinations to try
    cv=5,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1
)
```

### Optuna (Best — Bayesian Optimization)
```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    }
    model = xgb.XGBClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(study.best_params)
```

---

## Full Pipeline (Production Ready)

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define feature groups
num_features = ['age', 'salary']
cat_features = ['city', 'dept']

# Preprocessing pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# Full pipeline with model
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100))
])

full_pipeline.fit(X_train, y_train)
y_pred = full_pipeline.predict(X_test)
# The pipeline applies all transformations automatically on new data!
```

---

## Class Imbalance

```python
# Option 1: class_weight (simplest)
model = LogisticRegression(class_weight='balanced')
model = RandomForestClassifier(class_weight='balanced')

# Option 2: SMOTE (create synthetic minority samples)
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

# Option 3: Change threshold
# predict positive if prob > 0.3 instead of 0.5
y_pred = (model.predict_proba(X_test)[:, 1] >= 0.3).astype(int)

# Always use: F1, AUC, PR-AUC (not accuracy) to evaluate
```
