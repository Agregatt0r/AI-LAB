# Regularization: Ridge ($L_2$) and Lasso ($L_1$)

When a model has many features, correlated inputs (multicollinearity), or noisy data, standard linear regression (OLS) often suffers from **overfitting**. The weights grow excessively large to fit every minor variation in the training samples.

**Regularization** adds a penalty on the magnitude of the model weights to the loss function, preventing weights from exploding.

---

## 1. Ridge Regression ($L_2$ Regularization)

Ridge adds the sum of squared weights as a penalty:

$$J_{\text{Ridge}}(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2 + \lambda \sum_{j=1}^d w_j^2$$

where:
- $\lambda \ge 0$ (lambda, or alpha) is the **regularization hyperparameter**.
- When $\lambda = 0$: Reverts to standard unregularized Linear Regression.
- When $\lambda \to \infty$: Forces all weights towards zero (high bias, simple model).
- The bias term $b$ is **not** penalized.

### Gradient for Ridge:
$$\nabla_{\mathbf{w}} J_{\text{Ridge}} = \frac{2}{n} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y}) + 2\lambda \mathbf{w}$$

Updating $\mathbf{w}$ reveals why Ridge is often called **Weight Decay**:
$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \left(\nabla_{\mathbf{w}} \text{MSE} + 2\lambda \mathbf{w}\right) = \mathbf{w}(1 - 2\alpha\lambda) - \alpha \nabla_{\mathbf{w}} \text{MSE}$$

At each step, the weights are multiplied by $(1 - 2\alpha\lambda) < 1$ before subtracting the standard gradient step!

---

## 2. Lasso Regression ($L_1$ Regularization)

Lasso (Least Absolute Shrinkage and Selection Operator) penalizes the sum of absolute values of weights:

$$J_{\text{Lasso}}(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2 + \lambda \sum_{j=1}^d |w_j|$$

### Geometric Difference & Sparsity
Because the $L_1$ penalty has sharp corners on the coordinate axes (diamond-shaped constraint), the loss minimum often intersects the axes directly.
- **Ridge ($L_2$)**: Shrinks weights close to zero, but rarely sets them exactly to zero.
- **Lasso ($L_1$)**: Drives non-informative weights **exactly to zero**, performing automatic **feature selection**.

---

## 3. Comparison Summary

| Property | Ordinary Least Squares (OLS) | Ridge Regression ($L_2$) | Lasso Regression ($L_1$) |
|---|---|---|---|
| **Penalty** | None | $\lambda \sum w_j^2$ | $\lambda \sum \|w_j\|$ |
| **Weight Magnitude** | Can become very large | Shrunk smoothly toward 0 | Shrunk to exactly 0 |
| **Feature Selection** | No | No | **Yes** (creates sparse weights) |
| **Handling Multicollinearity** | Unstable | Excellent | Selects one feature arbitrarily |
| **Use Case** | Few features, low noise | Many correlated features | Many irrelevant features |
