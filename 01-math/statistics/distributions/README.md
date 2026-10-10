# Empirical Distributions & Standardization

In applied statistics, we work with **empirical data distributions** (the actual distribution of collected samples) and standardize them to compare features on a common scale.

## 1. Z-Score Standardization (Standard Score)

Standardization rescales data so that the mean is $0$ and the standard deviation is $1$:

$$z = \frac{x - \bar{x}}{s}$$

- $z > 0$: The observation is above the sample mean.
- $z < 0$: The observation is below the sample mean.
- $z = \pm 3$: Extreme outlier (less than $0.3\%$ chance in a normal distribution).

## 2. Standard Normal Distribution ($Z$) vs Student's $t$-Distribution

- **Z-distribution ($\mathcal{N}(0, 1)$):** Used when the true population standard deviation $\sigma$ is known or when sample size $n$ is large ($n \ge 30$).
- **Student's $t$-Distribution:** Used when sample size is small ($n < 30$) and $\sigma$ is unknown (estimated by $s$). It has **heavier tails** to account for the additional uncertainty in estimating $\sigma$.
  - Parameterized by degrees of freedom: $df = n - 1$.
  - As $df \to \infty$, the $t$-distribution converges to the standard normal distribution $Z$.

## Standardization in ML Pipelines

- **StandardScaler:** Crucial pre-processing step for distance-based algorithms (KNN, SVM, K-Means), gradient-based optimization (Linear/Logistic Regression, Neural Networks), and dimensionality reduction (PCA).
- Prevents features with large units (e.g. salary in thousands) from dominating features with small units (e.g. age).
