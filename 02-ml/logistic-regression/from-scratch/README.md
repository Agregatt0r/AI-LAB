# Logistic Regression From Scratch

This module assembles every concept from the previous modules into one complete, NumPy-only classifier. There are no machine learning libraries involved: the code implements the sigmoid, the loss, the gradients, the training loop, standardization, and prediction by hand.

---

## 1. Pipeline

1. **Standardize features:** $\mathbf{X}_s = \dfrac{\mathbf{X} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$
2. **Linear score:** $\mathbf{z} = \mathbf{X}_s\mathbf{w} + b$
3. **Probability:** $\mathbf{p} = \sigma(\mathbf{z})$
4. **Loss:** $J = \text{BCE}(\mathbf{y}, \mathbf{p})$
5. **Gradient descent:** $\mathbf{w} \leftarrow \mathbf{w} - \alpha\frac{1}{n}\mathbf{X}_s^\top(\mathbf{p}-\mathbf{y})$, $\;b \leftarrow b - \alpha\,\text{mean}(\mathbf{p}-\mathbf{y})$
6. **Prediction:** $\hat{y} = \mathbb{1}[\,p \geq t\,]$

---

## 2. Standardization and Data Leakage

Means and standard deviations must be computed on the **training set only**, then reused on the test set. Computing them on the full dataset leaks information from test data into training. The class stores $\mu$ and $\sigma$ at `fit` time and applies them later in `predict_proba`.

---

## 3. Class Design

| Method | Role |
|---|---|
| `fit(X, y)` | Learn $\mathbf{w}, b$ by gradient descent; records BCE per epoch |
| `predict_proba(X)` | Return $P(y=1 \mid \mathbf{x})$ |
| `predict(X, threshold)` | Return hard labels |
| `score(X, y)` | Return accuracy |

---

## 4. Sanity Checks

- BCE should start near $\ln 2 \approx 0.693$ (all weights zero) and decrease.
- Test accuracy should be close to training accuracy. A large gap signals overfitting.
- Standardization is essential here: the second feature in the practice data has a mean of 50, which would otherwise slow convergence.

---

## Summary

- A complete logistic regression needs only sigmoid, BCE, gradients, and an update loop.
- Fit statistics on training data only.
- The result matches what scikit-learn computes, as shown in `../scikit-learn`.

**Previous:** [Gradient Descent](../gradient-descent/README.md) · **Next:** [scikit-learn](../scikit-learn/README.md)
