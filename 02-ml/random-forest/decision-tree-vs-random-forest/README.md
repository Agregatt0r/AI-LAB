# Decision Tree vs Random Forest

While Random Forest is composed entirely of Decision Trees, the two models exhibit fundamentally different statistical properties, failure modes, and practical strengths.

---

## 1. Deep Architectural Comparison

| Dimension | Decision Tree (Single Tree) | Random Forest (Ensemble of Trees) |
|---|---|---|
| **Model Type** | Single base estimator | Bagged ensemble with random feature subsets |
| **Bias / Variance** | **Low bias, High variance** (prone to overfitting) | **Low bias, Low variance** (averaging cancels variance) |
| **Decision Boundary** | Rigid, sharp, orthogonal staircases | **Smooth, gradual, probabilistic transitions** |
| **Resistance to Noise** | Very poor (easily memorizes outliers) | **High** (outliers are outvoted by majority) |
| **Overfitting as $B \to \infty$**| N/A | **Does NOT overfit** (error converges to an upper bound) |
| **Interpretability** | **White box**: Exact human-readable rules | **Grey box**: Interpreted via Feature Importances |
| **Training Speed** | Fast ($O(d \cdot n \log n)$) | $B$ times slower (embarrassingly parallel: `n_jobs=-1`) |
| **Inference Speed** | Instantaneous ($O(\text{depth})$) | Fast, but evaluates $B$ separate trees ($O(B \cdot \text{depth})$) |
| **Hyperparameter Sensitivity**| High (requires careful pruning / `max_depth`) | Low (often works exceptionally well with default parameters) |

---

## 2. Geometric Decision Boundary Comparison

```
        Single Decision Tree                          Random Forest
   (Sharp, brittle staircases)                (Smooth, resilient contours)
  x2 ^                                       x2 ^
     |   +-------+                              |        .------.
     |   |   ▲   |   ■                          |       /   ▲    \    ■
     |   |       +-------+                      |      |          |
     |   |   ▲       ▲   |                      |       \   ▲    /
     +---+---------------+-----> x1             +--------'------'--------> x1
```

- A single tree must split with purely horizontal or vertical cuts. Small shifts in data rotate the entire subsequent subtree.
- A Random Forest averages hundreds of slightly different orthogonal cuts, synthesizing a **smooth, quasi-continuous non-linear contour**.

---

## 3. Does Adding More Trees Cause Overfitting?

> [!NOTE]
> **A common misconception is that adding more trees will eventually cause a Random Forest to overfit. This is mathematically false.**

Leo Breiman proved that as the number of trees $B \to \infty$:
$$\lim_{B \to \infty} \text{PE}^*(\text{Forest}) \leq \frac{\rho(1 - s^2)}{s^2}$$

where $\text{PE}^*$ is the generalization error, $\rho$ is the average tree correlation, and $s$ is tree strength.

**The generalization error converges to a fixed limit.**
- Adding more trees (e.g. increasing from 50 to 500) **never increases test error**.
- It only yields diminishing returns in accuracy while increasing computation time.

---

## 4. When to Use Which?

### Use a Single Decision Tree When:
1. Strict **regulatory compliance or auditability** is required (e.g. healthcare triage protocol, credit underwriting where every exact rule must be justified).
2. Ultra-low latency edge devices or microcontrollers with tiny memory footprints.

### Use a Random Forest When:
1. Maximum **predictive accuracy and robustness** is required on tabular data.
2. The dataset has noise, outliers, or high dimensionality.
3. You want strong out-of-the-box performance without hours of hyperparameter tuning.

---

## Summary

- Single Decision Trees have high variance, low stability, and sharp boundaries.
- Random Forests reduce variance via bagging and feature randomization, generating smooth and highly generalizable predictions.
- Adding more trees stabilizes the forest and never induces overfitting.

**Previous:** [Feature Importance](../feature-importance/README.md) · **Next:** [Scikit-Learn & From Scratch](../scikit-learn/README.md)

