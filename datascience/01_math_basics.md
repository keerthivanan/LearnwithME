# Math Basics for Data Science

---

## Statistics

### Mean, Median, Mode
- **Mean** = sum / count → pulled by outliers
- **Median** = middle value → robust to outliers
- **Mode** = most frequent value

### Variance & Standard Deviation
```
Variance  = average of (x - mean)²   → how spread out data is
Std Dev   = √Variance                 → same unit as data
```

### Normal Distribution
- Bell curve, symmetric around mean
- 68% of data within 1 std, 95% within 2 std, 99.7% within 3 std
- Many ML algorithms assume normal distribution

### Correlation
- Pearson r ∈ [-1, 1]
- r = 1 → perfect positive, r = -1 → perfect negative, r = 0 → no linear relationship
- **Correlation ≠ Causation** (always remember this!)

### P-value & Hypothesis Testing
- **Null hypothesis (H₀)**: no effect / no difference
- **p-value**: probability of seeing this data if H₀ is true
- p < 0.05 → reject H₀ → result is statistically significant
- **Type I error**: false positive (said yes, was no)
- **Type II error**: false negative (said no, was yes)

### Central Limit Theorem
> No matter the distribution, sample means will be normally distributed if sample is large enough (n > 30).  
> This is why we can use statistical tests even on non-normal data.

---

## Probability

### Basics
```
P(A) ∈ [0, 1]
P(A') = 1 - P(A)
P(A or B) = P(A) + P(B) - P(A and B)
P(A and B) = P(A) × P(B)   if A, B are independent
```

### Conditional Probability
```
P(A | B) = P(A and B) / P(B)
"Probability of A given B already happened"
```

### Bayes' Theorem
```
P(A | B) = P(B | A) × P(A) / P(B)

Example: P(disease | positive test) = P(positive test | disease) × P(disease) / P(positive test)
```

---

## Linear Algebra (What You Really Need)

### Vectors & Matrices
- Vector = list of numbers → represents a data point
- Matrix = table of numbers → represents entire dataset
- Dataset with 1000 rows, 10 features = matrix of shape (1000, 10)

### Dot Product
```
a = [1, 2, 3], b = [4, 5, 6]
a · b = 1×4 + 2×5 + 3×6 = 32

Used in: linear regression (w · x), neural networks
```

### Matrix Multiplication
- A (m×k) × B (k×n) = C (m×n)
- Used everywhere in ML — neural networks are just stacked matrix multiplications

### Gradient (Most Important)
- Gradient = direction of steepest increase in a function
- We go **opposite** to gradient to minimize loss → **Gradient Descent**
- Backpropagation = chain rule of calculus applied to neural networks

---

## Information Theory

### Entropy
- Measures uncertainty / randomness
- High entropy → unpredictable → lots of information
- Decision trees use entropy to find the best feature to split on

### Cross-Entropy Loss
- Used as loss function for classification
- Penalizes confident wrong predictions heavily
- `loss = -Σ y × log(ŷ)`
