# ML Algorithms — Interview Ready

> For each algorithm: what it is, how it works simply, when to use, code.

---

## SUPERVISED LEARNING

---

## 1. Linear Regression
**What**: Fit a straight line through data to predict a number.  
**How**: Find weights that minimize the sum of squared errors.  
**Use when**: Target is continuous, relationship is roughly linear.

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(model.coef_)       # weight for each feature
print(model.intercept_)  # bias
```

**Key formula**: `y = w₁x₁ + w₂x₂ + ... + b`

---

## 2. Logistic Regression
**What**: Classification algorithm. Predicts probability using sigmoid function.  
**How**: Linear regression output squeezed into [0,1] via sigmoid.  
**Use when**: Binary classification, need interpretable model, baseline.

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)  # probabilities
preds = model.predict(X_test)        # 0 or 1
```

**Sigmoid**: `P = 1 / (1 + e^(-z))` where `z = w·x + b`  
**Decision boundary**: predict 1 if P > 0.5

---

## 3. Decision Tree
**What**: Makes decisions by splitting data on features — like a flowchart.  
**How**: At each node, pick the feature + threshold that best separates classes (using Gini or entropy).  
**Use when**: Need interpretability, mixed data types, no scaling needed.

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)
model.fit(X_train, y_train)
```

**Pros**: Interpretable, no scaling, handles non-linearity  
**Cons**: Overfits easily → use ensemble (Random Forest, XGBoost)

---

## 4. Random Forest
**What**: Many decision trees trained on random subsets of data and features. Final answer = majority vote.  
**How**: Bagging + random feature selection. Reduces overfitting by averaging many uncorrelated trees.  
**Use when**: Good general-purpose model, need feature importance.

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=200,   # number of trees
    max_features='sqrt', # features per split
    n_jobs=-1,
    random_state=42
)
model.fit(X_train, y_train)

# Feature importance
importances = model.feature_importances_
```

**Key insight**: More trees = less variance. Doesn't overfit with more trees.

---

## 5. XGBoost / LightGBM (Gradient Boosting)
**What**: Trees trained SEQUENTIALLY — each tree fixes the errors of the previous ones.  
**How**: Fit new tree to the residuals (errors) of current model. Add with small learning rate.  
**Use when**: Best for tabular data. Wins most Kaggle competitions.

```python
import xgboost as xgb
model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    early_stopping_rounds=50
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
```

```python
import lightgbm as lgb
model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50)])
```

**XGBoost vs LightGBM**: LightGBM is faster, uses less memory, grows trees leaf-wise instead of level-wise.

---

## 6. Support Vector Machine (SVM)
**What**: Find the hyperplane that maximizes the margin between classes.  
**How**: Only the "support vectors" (closest points to boundary) matter. Kernel trick maps data to higher dimensions.  
**Use when**: Small/medium datasets, high-dimensional data (text), clear margin.

```python
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

model = Pipeline([
    ('scaler', StandardScaler()),   # SVM MUST be scaled
    ('svm', SVC(C=10, kernel='rbf', probability=True))
])
model.fit(X_train, y_train)
```

**C**: Low C = wide margin (underfits), High C = narrow margin (overfits)

---

## 7. K-Nearest Neighbors (KNN)
**What**: Classify by looking at K nearest neighbors and taking majority vote.  
**How**: No training! At prediction, compute distance to all training points.  
**Use when**: Simple baseline, small dataset.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('scaler', StandardScaler()),  # KNN MUST be scaled
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
```

**Cons**: Slow at prediction (O(n)), bad with high dimensions.

---

## 8. Naive Bayes
**What**: Probabilistic classifier using Bayes' theorem, assumes features are independent.  
**How**: P(class | features) ∝ P(features | class) × P(class)  
**Use when**: Text classification (spam detection), very fast, small data.

```python
from sklearn.naive_bayes import MultinomialNB  # for text counts
from sklearn.naive_bayes import GaussianNB     # for numeric features

model = MultinomialNB(alpha=1.0)  # alpha = smoothing
model.fit(X_train, y_train)
```

---

## UNSUPERVISED LEARNING

---

## 9. K-Means Clustering
**What**: Group data into K clusters by their similarity.  
**How**: Place K centroids, assign each point to nearest centroid, update centroids to cluster mean. Repeat.  
**Use when**: Customer segmentation, document clustering, image compression.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)  # scale!

# Find best K using elbow method
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Fit final model
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_scaled)
```

**Cons**: Must choose K, assumes spherical clusters, sensitive to outliers.

---

## 10. DBSCAN
**What**: Find clusters as dense regions. Doesn't need K. Marks outliers.  
**How**: Points with enough neighbors = core. Core points form clusters. Isolated = noise.  
**Use when**: Unknown K, clusters of irregular shapes, need outlier detection.

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)
# label = -1 means outlier
```

---

## 11. PCA (Dimensionality Reduction)
**What**: Compress data while keeping maximum information.  
**How**: Find directions of maximum variance (principal components), project data onto them.  
**Use when**: Too many features, visualization, remove correlated features.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # keep 95% of variance
X_reduced = pca.fit_transform(X_scaled)
print("Components kept:", pca.n_components_)

# For 2D visualization
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)
```

---

## 12. Anomaly Detection — Isolation Forest
**What**: Find outliers. Anomalies are easier to isolate than normal points.  
**How**: Randomly split features. Anomalies need fewer splits to isolate.  
**Use when**: Fraud detection, quality control, network intrusion.

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.05, random_state=42)
labels = model.fit_predict(X)   # 1 = normal, -1 = anomaly
scores = model.decision_function(X)  # lower = more anomalous
```

---

## Summary Table

| Algorithm | Type | Needs Scaling | Handles Non-linear | Speed |
|-----------|------|--------------|-------------------|-------|
| Linear Regression | Regression | Yes | No | Fast |
| Logistic Regression | Classification | Yes | No | Fast |
| Decision Tree | Both | No | Yes | Fast |
| Random Forest | Both | No | Yes | Medium |
| XGBoost/LightGBM | Both | No | Yes | Fast |
| SVM | Both | YES | With kernel | Slow |
| KNN | Both | YES | Yes | Very slow |
| Naive Bayes | Classification | No | No | Very fast |
| K-Means | Clustering | YES | No | Fast |
| PCA | Dim. Reduction | YES | No | Fast |

---

## Bias vs Variance (Must Know)

```
Bias    = model too simple, misses patterns → UNDERFITTING
Variance = model too complex, memorizes noise → OVERFITTING

High Bias:     training error high, test error high
High Variance: training error low, test error high

Fix underfitting: more complex model, more features, less regularization
Fix overfitting:  more data, simpler model, regularization, dropout
```
