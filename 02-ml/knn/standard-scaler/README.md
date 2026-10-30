# Feature Scaling: StandardScaler

**StandardScaler** (Z-score standardization) transforms numerical features so that each feature has a mean of zero ($\mu = 0$) and a standard deviation of one ($\sigma = 1$).

It is an indispensable preprocessing step for all distance-based algorithms (KNN, K-Means, SVM) and gradient-based models (Linear & Logistic Regression, Neural Networks).

---

## 1. Mathematical Formula

For each feature column $j$ in a dataset $\mathbf{X} \in \mathbb{R}^{n \times d}$:

$$z_{i, j} = \frac{x_{i, j} - \mu_j}{\sigma_j}$$

where:
- $\mu_j = \frac{1}{n} \sum_{i=1}^n x_{i, j}$ is the sample **mean** of feature $j$.
- $\sigma_j = \sqrt{\frac{1}{n} \sum_{i=1}^n (x_{i, j} - \mu_j)^2}$ is the **standard deviation** of feature $j$.

### Inverse Transformation:
To convert scaled features back to their original human-readable physical units:

$$x_{i, j} = z_{i, j} \cdot \sigma_j + \mu_j$$

---

## 2. StandardScaler vs MinMaxScaler

| Feature | StandardScaler (Z-Score) | MinMaxScaler |
|---|---|---|
| **Formula** | $z = \frac{x - \mu}{\sigma}$ | $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ |
| **Output Range** | Unbounded ($(-\infty, +\infty)$, typically $[-3, +3]$) | Strictly bounded in $[0, 1]$ |
| **Mean & Variance** | Mean $= 0$, Variance $= 1$ | Not fixed |
| **Outlier Handling** | Robust: outliers do not compress the main distribution | Fragile: an extreme outlier compresses normal data into a tiny band |
| **Best Used For** | Normal/Gaussian distributed features, KNN, SVM, Regression | Image pixel intensities ($[0, 255] \to [0, 1]$), bounded neural network inputs |

---

## 3. The Golden Rule of Data Preprocessing: No Data Leakage

> [!CAUTION]
> **Never fit scalers on the test set or the full combined dataset.**

```
CORRECT:
  X_train, X_test = split(X)
  scaler.fit(X_train)                  # Learn μ and σ from TRAIN ONLY
  X_train_scaled = scaler.transform(X_train)
  X_test_scaled  = scaler.transform(X_test)  # Apply TRAIN μ and σ to TEST

WRONG (Data Leakage):
  scaler.fit(entire_dataset)          # Information from test set leaks into training!
```

If you calculate mean and std on the entire dataset before splitting, information about the distribution and variance of the test set leaks into the training pipeline, leading to overly optimistic and invalid evaluation metrics.

---

## 4. Numerical Stability

If a feature column is constant across all samples (e.g. all values equal $5.0$), its standard deviation is $\sigma = 0$. Dividing by zero produces `NaN` or `inf`.

In production implementations, we guard against division by zero by replacing zero variance with $1.0$:

```python
self.std[self.std == 0.0] = 1.0
```

---

## Summary

- StandardScaler centers features at 0 and scales them to unit variance.
- Essential for KNN so that all feature dimensions contribute proportionally to Euclidean distance.
- Must be fitted **only** on training data to prevent data leakage.

**Previous:** [Feature Scaling Impact](../feature-scaling/README.md) · **Next:** [Classification & Regression with KNN](../classification-and-regression/README.md)

