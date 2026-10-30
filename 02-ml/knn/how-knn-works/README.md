# How K-Nearest Neighbors (KNN) Works

**K-Nearest Neighbors (KNN)** is one of the simplest yet most effective non-parametric algorithms in machine learning. It can be used for both **classification** and **regression**.

Unlike parametric models (such as Linear Regression or Logistic Regression), KNN does not learn a fixed set of weights $\mathbf{w}$ and bias $b$. Instead, it makes predictions by looking directly at the **$k$ closest data points** in the training set.

---

## 1. Key Intuition

KNN is based on the neighborhood assumption: **similar data points exist close to each other in feature space**.

- If an unknown fruit is red, spherical, and weighs 150g, and its closest neighbors in the training set are apples, it is probably an apple.
- If a customer shares the exact same age, income, and browsing habits as customers who purchased a product, they will likely purchase it too.

```
       Feature 2 ^
                 |         ▲ Class 1
                 |     ▲       ▲
                 |         ? (Query point)
                 |      ■     ■
                 |         ■   Class 2
                 +------------------------> Feature 1
```

---

## 2. Core Properties

### 1. Lazy Learning (Instance-Based)
KNN has **no explicit training phase**:
- `fit(X, y)` simply memorizes and stores the training dataset in memory.
- All actual computation occurs at **query time** (`predict(X)`).

### 2. Non-Parametric
- Parametric models assume a specific functional form (e.g., a straight line or plane $y = \mathbf{w}^\top\mathbf{x} + b$).
- KNN makes **no assumptions** about the underlying data distribution, allowing it to naturally model complex, non-linear decision boundaries.

---

## 3. The 4 Steps of KNN Prediction

To predict the target for a new query point $\mathbf{x}_{\text{query}}$:

1. **Calculate Distances:** Compute the distance between $\mathbf{x}_{\text{query}}$ and every sample $\mathbf{x}_i$ in the stored training set ($i = 1, \dots, n$).
2. **Sort:** Sort the training samples in ascending order of distance.
3. **Select $k$ Neighbors:** Pick the $k$ training points with the smallest distances ($N_k$).
4. **Aggregate Targets:**
   - **For Classification:** Take the **majority vote** (mode) of the classes in $N_k$.
   - **For Regression:** Take the **mean** (average) of the continuous targets in $N_k$.

---

## 4. Computational Complexity

| Phase | Time Complexity | Space Complexity | Notes |
|---|---|---|---|
| **Training** | $O(1)$ | $O(n \cdot d)$ | Only stores the training matrix $\mathbf{X}$ and targets $\mathbf{y}$. |
| **Inference (Single Sample)** | $O(n \cdot d + n \log k)$ | $O(n)$ | Must compute distance to all $n$ training samples. |
| **Inference ($m$ Samples)** | $O(m \cdot n \cdot d)$ | $O(m \cdot n)$ | Can become slow for large datasets ($n > 100,000$). |

Where:
- $n$ is the number of training samples.
- $d$ is the number of features (dimensions).
- $m$ is the number of query samples.

---

## 5. Strengths & Limitations

### Strengths:
- Very easy to understand and explain.
- Naturally learns non-linear and multi-modal boundaries without complex math.
- Multi-class classification comes out of the box without One-vs-Rest wrappers.
- Zero training time.

### Limitations:
- **Inference is expensive:** predicting on large datasets requires comparing against all stored points.
- **Memory intensive:** the full training dataset must stay in RAM.
- **Curse of Dimensionality:** in high-dimensional spaces ($d \gg 50$), Euclidean distance loses discriminatory power because all points become nearly equidistant.
- **Sensitive to feature scale:** features with large numerical ranges completely dominate the distance calculation (requires standardization).

---

## Summary

- KNN stores the training data and delays computation until prediction time (lazy learning).
- Prediction finds the $k$ closest neighbors and aggregates their labels (vote for classification, average for regression).
- Highly expressive for non-linear patterns, but computationally expensive during inference.

**Next:** [Distance Metrics](../distance-metrics/README.md)

