# Top Data Science Interview Q&A

---

## Statistics & Probability

**Q: What is the difference between mean and median? When do you use each?**  
Mean = sum/count, sensitive to outliers. Median = middle value, robust to outliers.  
Use median when data is skewed (salary, house prices). Use mean for symmetric distributions.

**Q: What is p-value?**  
Probability of observing the data (or more extreme) assuming null hypothesis is true.  
p < 0.05 → reject null hypothesis → result is statistically significant.  
Common mistake: p-value is NOT the probability the null hypothesis is true.

**Q: Explain Type I and Type II errors.**  
Type I (α) = False Positive → you said there's an effect, there isn't. (Crying wolf)  
Type II (β) = False Negative → you said no effect, there is one. (Missing cancer)  
Decreasing one increases the other. Adjust based on which is costlier in your domain.

**Q: What is the Central Limit Theorem?**  
Regardless of the original distribution, the distribution of sample means approaches Normal as sample size increases (n > 30). This is why we can use t-tests even for non-normal data.

**Q: What is Bayes' Theorem?**  
P(A|B) = P(B|A) × P(A) / P(B)  
Used in spam filters: P(spam | "free money") = P("free money" | spam) × P(spam) / P("free money")

**Q: What is correlation vs causation?**  
Correlation: two things change together. Causation: one CAUSES the other.  
Ice cream sales and drowning rates are correlated (both increase in summer) but neither causes the other. Use experiments (A/B tests) to establish causation.

---

## Machine Learning

**Q: What is overfitting and how do you fix it?**  
Overfitting: model memorizes training data, fails on new data. Training accuracy >> test accuracy.  
Fix: get more data, simplify model, add regularization (L1/L2), dropout, early stopping, cross-validation.

**Q: What is the bias-variance tradeoff?**  
Bias: error from wrong assumptions (model too simple = underfitting).  
Variance: error from sensitivity to noise (model too complex = overfitting).  
Goal: minimize both. Find the right model complexity via cross-validation.

**Q: How does gradient descent work?**  
Compute the gradient (slope) of the loss function. Move weights in the opposite direction of gradient by a small step (learning rate). Repeat until loss converges to minimum.

**Q: What is the difference between bagging and boosting?**  
Bagging: train models INDEPENDENTLY on random subsets, combine by averaging. Reduces variance. (Random Forest)  
Boosting: train models SEQUENTIALLY, each fixes previous errors. Reduces bias. (XGBoost, LightGBM)

**Q: How does Random Forest differ from a single Decision Tree?**  
Random Forest trains many trees on random subsets of data and features, takes majority vote. Less variance, more robust. A single tree overfits. Random Forest rarely overfits as you add more trees.

**Q: Explain XGBoost.**  
Gradient boosting with extra features: L1+L2 regularization, second-order gradients, column subsampling, parallel tree building, handles missing values automatically. State-of-the-art for tabular data.

**Q: When would you use SVM vs Logistic Regression?**  
Both are linear classifiers. Use SVM when: data is high-dimensional (text), there's a clear margin of separation, small dataset. Use Logistic Regression when: need probability outputs, large dataset, need faster training.

**Q: How do you handle imbalanced classes?**  
1. Class weights (`class_weight='balanced'`)  
2. SMOTE oversampling  
3. Lower decision threshold  
4. Use F1/AUC instead of accuracy to evaluate

**Q: What is regularization and why use it?**  
Adds penalty for large weights to prevent overfitting.  
L1 (Lasso): drives some weights to exactly 0 → feature selection.  
L2 (Ridge): shrinks all weights toward 0 → handles multicollinearity.

**Q: What is cross-validation?**  
Split data into K folds. Train on K-1, test on 1. Repeat K times. Average the scores.  
Gives more reliable estimate than single train/test split. Avoids lucky/unlucky splits.

---

## Feature Engineering & Data

**Q: How do you handle missing values?**  
First understand WHY they're missing. Then:  
- Drop if < 5% and no pattern  
- Impute with mean/median (numeric) or mode (categorical)  
- Add "is_missing" indicator feature (very useful!)  
- KNN imputation for correlated features  
Never fill test set with mean from test set — use training mean only.

**Q: What is data leakage?**  
When information from the future or target leaks into training features.  
Example: scaling the entire dataset before splitting, or a feature derived from the target.  
Fix: always split first, then preprocess. Build a Pipeline.

**Q: What is feature selection? Why does it matter?**  
Remove irrelevant or redundant features.  
Benefits: less overfitting, faster training, more interpretable model.  
Methods: correlation analysis, feature importance (tree models), Lasso (drops weights to 0), SHAP values.

**Q: How do you encode categorical variables?**  
One-hot encoding: for low cardinality (< 15 values) with linear models/NNs.  
Label encoding: only for ordinal data or tree models.  
Target encoding: high cardinality, but risk of leakage — use CV.  
Frequency encoding: safe, no leakage.

---

## Deep Learning

**Q: What is backpropagation?**  
Algorithm to compute gradients of loss w.r.t weights using chain rule of calculus.  
Works backward from output → input, computing how much each weight contributed to the error.

**Q: Why use ReLU instead of sigmoid in hidden layers?**  
Sigmoid has vanishing gradient problem — gradients become tiny in deep networks, training stalls.  
ReLU: f(x) = max(0,x). Gradient is 1 for positive values → no vanishing gradient. Faster training.

**Q: What is the vanishing gradient problem?**  
In deep networks, gradients get multiplied through many layers during backprop. If gradients < 1, they shrink to near zero → early layers learn nothing. Solved by: ReLU, batch norm, residual connections (ResNet), LSTM gates.

**Q: What is dropout?**  
During training, randomly set some neuron outputs to 0 with probability p.  
Forces network to learn redundant representations → reduces overfitting.  
At test time, dropout is turned off (all neurons active, weights scaled by 1-p).

**Q: What is batch normalization?**  
Normalize the inputs of each layer to have mean=0, std=1. Then scale and shift with learned parameters.  
Benefits: faster training, can use higher LR, acts as regularizer, reduces sensitivity to initialization.

**Q: What makes Transformers better than RNNs?**  
RNNs: sequential processing (slow), gradient vanishes over long sequences.  
Transformers: all positions attend to all others simultaneously (parallelizable), no forgetting, scales better. Self-attention handles long-range dependencies directly.

**Q: What is BERT? What is GPT? Difference?**  
BERT: bidirectional encoder. Reads full sentence both ways. Good for understanding tasks (classification, NER, QA). Fine-tune on labeled data.  
GPT: decoder-only, left-to-right. Good for generation tasks. Can do few-shot learning.  
BERT = understanding. GPT = generation.

**Q: What is transfer learning?**  
Take a model pretrained on large dataset, fine-tune on your smaller task-specific dataset.  
Benefits: needs less data, trains faster, better performance. Used heavily in CV (ImageNet → your task) and NLP (BERT → sentiment analysis).

---

## Model Evaluation

**Q: Why is accuracy a bad metric for imbalanced data?**  
If 99% is class 0, predicting always 0 gives 99% accuracy but is completely useless.  
Use F1-score, AUC-ROC, or PR-AUC instead.

**Q: What is AUC-ROC?**  
Area Under the Receiver Operating Characteristic curve.  
ROC plots True Positive Rate vs False Positive Rate at all thresholds.  
AUC = 0.5 → random, AUC = 1.0 → perfect.  
Threshold-independent: tells you overall ranking ability of the model.

**Q: Precision vs Recall — which to optimize?**  
Precision: of things predicted positive, how many actually are? Optimize when FP is costly.  
Recall: of all actual positives, how many did you catch? Optimize when FN is costly.  
Example: cancer detection → optimize recall (don't miss cases).  
Spam filter → optimize precision (don't block real emails).

**Q: What is K-fold cross-validation?**  
Split data into K equal parts. Train on K-1 parts, evaluate on 1. Rotate the test fold K times. Average scores. Use stratified K-fold for classification to maintain class proportions.

---

## Practical

**Q: How do you approach a new ML problem?**  
1. Understand the business problem + success metric  
2. Get and explore the data (EDA)  
3. Baseline model (simple, interpretable)  
4. Feature engineering  
5. Try better models, tune hyperparameters  
6. Evaluate on held-out test set  
7. Interpret results + present to stakeholders

**Q: How do you handle large datasets that don't fit in memory?**  
- `pd.read_csv(chunksize=10000)` — process in chunks  
- Reduce dtypes (float64 → float32, int64 → int32, object → category)  
- Use Dask or Spark for distributed processing  
- LightGBM / XGBoost have out-of-core support

**Q: How do you explain a model to non-technical stakeholders?**  
- SHAP values: show how each feature contributes to each prediction  
- Feature importance: which variables matter most  
- Simple analogies: "The model works like a decision flowchart..."  
- Show business impact: "This model would have caught X% more fraud..."

**Q: What is the difference between parametric and non-parametric models?**  
Parametric: assume a form for the function (e.g., linear regression assumes linearity). Fixed number of parameters. Faster, needs less data.  
Non-parametric: no fixed form. Number of parameters grows with data (KNN, decision trees). More flexible.

**Q: What is a confusion matrix?**  
Table showing actual vs predicted classes. Gives TP, FP, TN, FN.  
From this you derive all classification metrics: accuracy, precision, recall, F1.

**Q: Explain the curse of dimensionality.**  
As dimensions (features) increase, data becomes sparse. Distances between points become similar (everything is "far" from everything else). Models need exponentially more data. Fix: feature selection, PCA, dimensionality reduction.

---

## Tips for Interview Success

```
1. Always clarify the problem before diving in
2. Start with a simple baseline — never jump to complex models
3. Ask about the metric — accuracy vs AUC vs business metric
4. Mention data leakage and how you prevent it
5. Talk about how you'd validate the model in production
6. Show curiosity: "What's the class distribution? Any missing values?"
7. Explain tradeoffs: "This model is accurate but slow; this one is fast but less accurate"
```
