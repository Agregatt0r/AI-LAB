# Covariance

**Covariance** measures the joint variability and direction of the linear relationship between two random variables $X$ and $Y$.

## Formula

For population covariance with $N$ pairs of observations $(x_1, y_1), (x_2, y_2), \dots, (x_N, y_N)$:

$$\text{Cov}(X, Y) = \frac{1}{N} \sum_{i=1}^N (x_i - \mu_X)(y_i - \mu_Y)$$

*(For sample covariance, divide by $N - 1$)*.

## Interpretation of Sign

- **$\text{Cov}(X, Y) > 0$ (Positive Covariance):** When $X$ is above its mean, $Y$ tends to be above its mean (variables increase together).
- **$\text{Cov}(X, Y) < 0$ (Negative Covariance):** When $X$ is above its mean, $Y$ tends to be below its mean (inverse relationship).
- **$\text{Cov}(X, Y) \approx 0$ (Zero Covariance):** No linear relationship between $X$ and $Y$.

## Key Properties

1. **Symmetry:** $\text{Cov}(X, Y) = \text{Cov}(Y, X)$
2. **Variance relation:** $\text{Cov}(X, X) = \text{Var}(X) = \sigma_X^2$

## Example

$X = [1, 2, 3, 4, 5] \implies \mu_X = 3.0$  
$Y = [2, 4, 6, 8, 10] \implies \mu_Y = 6.0$

Deviations $(x_i - \mu_X)(y_i - \mu_Y)$:
- $i=1$: $(1 - 3)(2 - 6) = (-2)(-4) = 8$
- $i=2$: $(2 - 3)(4 - 6) = (-1)(-2) = 2$
- $i=3$: $(3 - 3)(6 - 6) = (0)(0) = 0$
- $i=4$: $(4 - 3)(8 - 6) = (1)(2) = 2$
- $i=5$: $(5 - 3)(10 - 6) = (2)(4) = 8$

$$\text{Sum} = 8 + 2 + 0 + 2 + 8 = 20$$
$$\text{Cov}(X, Y) = \frac{20}{5} = 4.0$$

## Covariance in ML

- **PCA (Principal Component Analysis):** The covariance matrix $\Sigma$ describes how all pairs of features correlate, and its eigenvectors form the principal components.
- **Multivariate Gaussian Distributions:** Covariance matrix determines shape and orientation of the distribution.
