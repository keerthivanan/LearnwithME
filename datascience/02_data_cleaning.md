# Data Cleaning & EDA

---

## EDA Checklist (Do This Every Time)

```python
import pandas as pd
import seaborn as sns

df = pd.read_csv('data.csv')

df.shape          # rows, columns
df.dtypes         # data types
df.describe()     # mean, std, min, max per column
df.isnull().sum() # missing values per column
df.duplicated().sum()  # duplicate rows

# Visualize distributions
df.hist(bins=30, figsize=(15, 10))

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
```

---

## Missing Values

### How to Detect
```python
df.isnull().sum()                        # count
df.isnull().sum() / len(df) * 100        # percentage
```

### What to Do

| Missing % | Strategy |
|-----------|---------|
| < 5% | Fill with mean/median/mode |
| 5–30% | Model-based imputation or add "missing" flag |
| > 30% | Consider dropping the column |

```python
# Fill with median (safer than mean — not affected by outliers)
df['age'].fillna(df['age'].median(), inplace=True)

# Fill with mode (for categorical)
df['city'].fillna(df['city'].mode()[0], inplace=True)

# Add "was missing" indicator feature (very useful!)
df['age_missing'] = df['age'].isnull().astype(int)

# Drop rows / columns
df.dropna(subset=['target'])   # drop rows where target is null
df.drop(columns=['col_with_80pct_missing'])

# KNN Imputation (best quality)
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df)
```

---

## Outliers

### Detect
```python
# IQR method
Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['salary'] < Q1 - 1.5*IQR) | (df['salary'] > Q3 + 1.5*IQR)]

# Z-score method (for normal data)
from scipy import stats
z = stats.zscore(df['salary'])
outliers = df[abs(z) > 3]
```

### What to Do
```python
# Option 1: Remove (if clearly wrong data — e.g. age = 999)
df = df[df['age'] < 120]

# Option 2: Cap (clip at percentile — safest)
lower = df['salary'].quantile(0.01)
upper = df['salary'].quantile(0.99)
df['salary'] = df['salary'].clip(lower, upper)

# Option 3: Log transform (for right-skewed data like prices)
import numpy as np
df['salary_log'] = np.log1p(df['salary'])
```

---

## Feature Encoding

### For Numeric Features

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standard Scaling: mean=0, std=1 (use for linear models, SVM, neural nets)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)  # fit on train only!
X_test_scaled = scaler.transform(X_test)

# Min-Max: range [0,1] (use for neural nets, KNN)
scaler = MinMaxScaler()

# ⚠️ NEVER fit scaler on full dataset — fit on train, transform test
```

### For Categorical Features

```python
# One-Hot Encoding (for low cardinality: < 15 unique values)
pd.get_dummies(df, columns=['city', 'color'], drop_first=True)

# Label Encoding (only for tree-based models or ordinal data)
from sklearn.preprocessing import LabelEncoder
df['size'] = LabelEncoder().fit_transform(df['size'])

# Target Encoding (for high cardinality — use carefully, can leak)
mean_salary_by_city = df.groupby('city')['salary'].mean()
df['city_encoded'] = df['city'].map(mean_salary_by_city)

# Frequency Encoding (safe, no leakage)
freq = df['city'].value_counts(normalize=True)
df['city_freq'] = df['city'].map(freq)
```

---

## Feature Engineering

```python
# Create new features from existing ones
df['age_squared'] = df['age'] ** 2
df['price_per_sqft'] = df['price'] / df['area']
df['debt_ratio'] = df['debt'] / df['income']

# Date features
df['date'] = pd.to_datetime(df['date'])
df['year']      = df['date'].dt.year
df['month']     = df['date'].dt.month
df['dayofweek'] = df['date'].dt.dayofweek
df['is_weekend']= df['dayofweek'].isin([5,6]).astype(int)

# Text features
df['text_len']   = df['review'].str.len()
df['word_count'] = df['review'].str.split().str.len()

# Aggregation (group stats)
user_avg = df.groupby('user_id')['spend'].mean().rename('user_avg_spend')
df = df.join(user_avg, on='user_id')
```

---

## Train/Test Split (Critical Rule)

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify for classification!
)

# ALWAYS: fit preprocessing on TRAIN, apply to TEST
# NEVER: look at test data before finalizing model
```

### Data Leakage — The #1 Mistake
```
Leakage = test data "leaks" info into training → model looks great but fails in production

Common leakage:
1. Scaling on full dataset before splitting
2. Feature derived from the target (e.g., "number of purchases" predicts "purchased")
3. Future data used to predict past
4. Duplicate rows in both train and test

Prevention: always split FIRST, then preprocess
```
