# Covariance Matrix

The **covariance matrix** measures the variance of each feature and how features vary together (co-vary).

## Variance and Covariance

For centered data ($x_{\text{centered}} = x - \bar{x}$):

$$\text{Cov}(x, y) = \frac{1}{n - 1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y}) = \frac{1}{n - 1} \sum_{i=1}^{n} x_{\text{centered}, i} \cdot y_{\text{centered}, i}$$

- When $x = y$, $\text{Cov}(x, x) = \text{Var}(x)$ (variance of feature $x$).
- $\text{Cov}(x, y) > 0$: when $x$ increases, $y$ tends to increase.
- $\text{Cov}(x, y) < 0$: when $x$ increases, $y$ tends to decrease.
- $\text{Cov}(x, y) = 0$: no linear relationship.

## 2×2 Covariance Matrix

$$C = \begin{bmatrix} \text{Cov}(x, x) & \text{Cov}(x, y) \\ \text{Cov}(y, x) & \text{Cov}(y, y) \end{bmatrix}$$

- **Diagonal elements:** variances of individual features.
- **Off-diagonal elements:** covariances between different features ($\text{Cov}(x, y) = \text{Cov}(y, x)$).

## Matrix Form

$$C = \frac{1}{n - 1} X_{\text{centered}}^T X_{\text{centered}}$$
