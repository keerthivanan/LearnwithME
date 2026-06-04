# Statistics & A/B Testing

---

## Probability Distributions

| Distribution | Use | Example |
|-------------|-----|---------|
| Normal | Continuous, symmetric | Heights, test scores |
| Binomial | Count of successes in n trials | # heads in 10 coin flips |
| Poisson | Count of events in fixed time | Emails per hour |
| Bernoulli | Single yes/no event | Single coin flip |
| Uniform | All values equally likely | Random number generation |
| Exponential | Time between events | Time between customer arrivals |

```python
from scipy import stats
import numpy as np

# Normal
norm = stats.norm(loc=0, scale=1)
norm.pdf(0)       # probability density
norm.cdf(1.96)    # P(X <= 1.96) = 0.975

# Binomial: P(5 heads in 10 flips)
binom = stats.binom(n=10, p=0.5)
binom.pmf(5)

# Poisson: P(3 emails when avg = 2)
poisson = stats.poisson(mu=2)
poisson.pmf(3)
```

---

## Hypothesis Testing

### Step-by-Step Process
```
1. State H₀ (null) and H₁ (alternative)
2. Choose significance level α (usually 0.05)
3. Choose the right test
4. Compute test statistic and p-value
5. If p < α → reject H₀
6. State conclusion in plain language
```

### Common Tests

**t-test** — compare means
```python
from scipy import stats

# One-sample: is mean different from 30?
t, p = stats.ttest_1samp(data, popmean=30)

# Two-sample: are two groups different?
t, p = stats.ttest_ind(group_a, group_b)          # assumes equal variance
t, p = stats.ttest_ind(group_a, group_b, equal_var=False)  # Welch's (safer)

# Paired: same subjects, before vs after
t, p = stats.ttest_rel(before, after)

if p < 0.05:
    print("Significant difference!")
```

**Chi-square test** — are categorical variables independent?
```python
# Example: Is click rate independent of gender?
observed = np.array([[30, 70],   # male: clicked, not clicked
                     [50, 50]])  # female: clicked, not clicked

chi2, p, dof, expected = stats.chi2_contingency(observed)
print(f"Chi2={chi2:.2f}, p={p:.4f}")
```

**Mann-Whitney U** — non-parametric alternative to t-test (no normality assumption)
```python
u, p = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
```

**ANOVA** — compare means across 3+ groups
```python
f, p = stats.f_oneway(group1, group2, group3)
# Post-hoc: which groups are different?
from statsmodels.stats.multicomp import pairwise_tukeyhsd
result = pairwise_tukeyhsd(data['value'], data['group'])
```

---

## A/B Testing

### What is A/B Testing?
Compare two versions (A = control, B = treatment) to measure which is better.  
Example: Does the new checkout button increase conversion rate?

### A/B Test Process

```
1. Define metric (conversion rate, revenue, click rate)
2. Formulate hypotheses:
   H₀: conversion_B = conversion_A (no difference)
   H₁: conversion_B ≠ conversion_A (there is a difference)
3. Calculate required sample size (power analysis)
4. Run experiment (randomly assign users to A or B)
5. Collect data
6. Statistical test
7. Make decision
```

### Sample Size Calculation (Critical!)
```python
from statsmodels.stats.power import NormalIndPower

power_analysis = NormalIndPower()
sample_size = power_analysis.solve_power(
    effect_size=0.2,   # minimum detectable effect
    alpha=0.05,        # significance level
    power=0.80,        # 80% chance to detect real effect
    ratio=1.0          # equal groups
)
print(f"Required per group: {int(sample_size)}")

# Or using proportions
from statsmodels.stats.proportion import proportion_effectsize, zt_ind_solve_power

# Current conversion: 10%, target: 12%
effect = proportion_effectsize(0.10, 0.12)
n = zt_ind_solve_power(effect_size=effect, alpha=0.05, power=0.8)
print(f"Need {int(n)} users per group")
```

### Running the Test
```python
import pandas as pd
from scipy import stats
import numpy as np

# Your experiment data
df = pd.read_csv('ab_test_results.csv')

control   = df[df['group'] == 'A']['converted']
treatment = df[df['group'] == 'B']['converted']

# Conversion rates
conv_a = control.mean()
conv_b = treatment.mean()
print(f"Control: {conv_a:.3f}, Treatment: {conv_b:.3f}")
print(f"Relative lift: {(conv_b - conv_a)/conv_a * 100:.1f}%")

# Z-test for proportions (for conversion rates)
from statsmodels.stats.proportion import proportions_ztest

count = np.array([treatment.sum(), control.sum()])
nobs  = np.array([len(treatment), len(control)])

stat, p_value = proportions_ztest(count, nobs)
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("SIGNIFICANT — launch treatment!")
else:
    print("Not significant — don't launch yet")

# Confidence interval for difference
from statsmodels.stats.proportion import proportion_confint
ci_a = proportion_confint(control.sum(), len(control), alpha=0.05)
ci_b = proportion_confint(treatment.sum(), len(treatment), alpha=0.05)
print(f"A: {ci_a[0]:.3f} - {ci_a[1]:.3f}")
print(f"B: {ci_b[0]:.3f} - {ci_b[1]:.3f}")
```

### Common A/B Testing Mistakes

```
1. Peeking problem: checking results before reaching required sample size
   → p-value fluctuates early → higher chance of false positive
   Fix: pre-commit to sample size, use sequential testing

2. Multiple comparisons: testing 20 variants → 1 will be significant by chance (5%)
   Fix: Bonferroni correction (α/n), or FDR correction

3. Simpson's Paradox: overall trend reverses when data is segmented
   Example: Treatment wins overall, but control wins in every subgroup
   Fix: always segment your data and check

4. Network effects: users in A affect users in B (e.g., social features)
   Fix: cluster randomization (randomize by group, not individual)

5. Novelty effect: users engage more simply because something is new
   Fix: run experiment long enough (1-2 weeks minimum)

6. Seasonal effects: starting experiment on a Monday ≠ starting on Friday
   Fix: run for full weeks, avoid holidays

7. Imbalanced groups: not 50/50 split
   Fix: check randomization, use stratified randomization
```

### Bayesian A/B Testing
```python
# Alternative to frequentist approach
# Directly answers: "What's P(B > A)?"
import numpy as np

n_simulations = 100000

# Prior: Beta(1,1) = uniform (no prior knowledge)
# Update with data: Beta(1 + successes, 1 + failures)

successes_a, n_a = 500, 5000   # 10% conversion
successes_b, n_b = 540, 5000   # 10.8% conversion

samples_a = np.random.beta(1 + successes_a, 1 + n_a - successes_a, n_simulations)
samples_b = np.random.beta(1 + successes_b, 1 + n_b - successes_b, n_simulations)

prob_b_better = np.mean(samples_b > samples_a)
expected_lift = np.mean((samples_b - samples_a) / samples_a)

print(f"P(B > A): {prob_b_better:.3f}")
print(f"Expected relative lift: {expected_lift:.3f}")
```

---

## Confidence Intervals

```
95% CI = [x̄ - 1.96×SE, x̄ + 1.96×SE]
where SE = std / √n

Interpretation:
"If we repeated this 100 times, 95 of the intervals would contain the true mean"
NOT: "95% chance the true mean is in this interval"
```

```python
import scipy.stats as stats

data = [23, 45, 12, 67, 34, 56, 78, 45, 23, 65]
n = len(data)
mean = np.mean(data)
se = stats.sem(data)

ci_95 = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)
print(f"Mean: {mean:.2f}, 95% CI: ({ci_95[0]:.2f}, {ci_95[1]:.2f})")
```

---

## Effect Size (Often Overlooked!)

Statistical significance ≠ practical significance.  
With 1M users, even a 0.001% difference is "significant" but meaningless.

```python
# Cohen's d (for means)
d = (mean_b - mean_a) / pooled_std
# d < 0.2: tiny, 0.2-0.5: small, 0.5-0.8: medium, > 0.8: large

# For proportions
from statsmodels.stats.proportion import proportion_effectsize
effect = proportion_effectsize(p1, p2)
```

---

## Key Interview Questions

**Q: What is a p-value? Common misconception?**  
p-value = P(seeing this data | H₀ is true). It is NOT P(H₀ is true | this data).

**Q: Can you have a significant result that isn't practically meaningful?**  
Yes! With large samples, tiny differences are significant. Always report effect size.

**Q: How do you decide sample size for A/B test?**  
Power analysis: need to specify minimum detectable effect, α (0.05), and power (0.80).

**Q: What's the difference between one-tailed and two-tailed test?**  
Two-tailed: B ≠ A (is there ANY difference?)  
One-tailed: B > A (is B specifically better?)  
Use two-tailed unless you have strong prior reason to test only one direction.

**Q: What do you do if your A/B test shows significant result but business metric doesn't improve?**  
Investigate: surrogate metric may not correlate with business metric. Look at guardrail metrics. Check for confounds. Consider longer-term effects.
