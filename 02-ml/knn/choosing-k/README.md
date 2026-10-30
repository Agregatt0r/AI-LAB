# Choosing $k$ and the Bias-Variance Tradeoff

The hyperparameter $k$ (number of nearest neighbors) controls the complexity of a KNN model. Choosing the right value of $k$ is essential for balancing **overfitting** (high variance) against **underfitting** (high bias).

---

## 1. The Extremes: $k = 1$ vs $k = n$

```
          k = 1 (Overfitting)                  Optimal k                       k = n (Underfitting)
    High Variance, Low Bias              Balanced Decision               High Bias, Low Variance
      y ^   _/\_   /\                      y ^                           y ^
        |  /    \_/  \                       |    /-----\                  |   -------------------
        | | *  ▲  *   |                      |   / *  ▲  \                 |   (Predicts majority
        |  \_________/                       |  /_________\                |    class everywhere)
        +----------------> x                 +----------------> x          +---------------------> x
```

### When $k = 1$:
- Every query point is assigned the class of its single nearest neighbor.
- **Training error is always 0 (100% training accuracy)** because every training point is its own nearest neighbor (distance $= 0$).
- The decision boundary is jagged and fractured, wrapping around individual noisy outliers.
- **Problem:** Severe **overfitting** and sensitivity to noise.

### When $k = n$ (total number of samples):
- Every query point polls all $n$ training samples.
- The prediction is always the most frequent class in the entire training dataset, regardless of the input $\mathbf{x}$.
- The decision boundary is completely flat or non-existent.
- **Problem:** Severe **underfitting** (high bias), ignoring local feature patterns entirely.

---

## 2. The Bias-Variance Tradeoff in KNN

| Metric | Small $k$ (e.g. $k=1$) | Large $k$ (e.g. $k=50$) |
|---|---|---|
| **Model Complexity** | High | Low |
| **Bias** | Low | High |
| **Variance** | High | Low |
| **Decision Boundary** | Jagged, island-like, irregular | Smooth, simple |
| **Noise Sensitivity** | Highly sensitive to outliers | Robust to noise and outliers |
| **Risk** | Overfitting | Underfitting |

---

## 3. Practical Guidelines for Selecting $k$

1. **Use an Odd Value for Binary Classification:**
   If $y \in \{0, 1\}$, choosing an odd $k$ (e.g. $3, 5, 7$) guarantees that a majority vote cannot result in a 50/50 tie.
2. **Rule of Thumb Heuristic:**
   A common starting baseline is:
   $$k \approx \sqrt{n_{\text{train}}}$$
   For $n = 100$, $\sqrt{100} = 10 \implies$ test $k \in \{7, 9, 11\}$.
3. **Hyperparameter Tuning via Validation Set:**
   Split your training data into a sub-train set and a validation set. Evaluate $k$ across a range:
   $$k \in [1, 3, 5, 7, \dots, 25]$$
   Select the $k$ that achieves the highest accuracy / lowest error on the **validation set**.

---

## Summary

- $k=1$ memorizes training noise (100% train accuracy, poor generalization).
- Large $k$ oversmooths the boundary and loses local resolution.
- Select $k$ via validation curve / cross-validation, preferring odd values to prevent voting ties.

**Previous:** [Distance Metrics](../distance-metrics/README.md) · **Next:** [Feature Scaling Impact](../feature-scaling/README.md)

