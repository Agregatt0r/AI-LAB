# K-Nearest Neighbors From Scratch

This module unites every concept from the previous modules into a complete, clean, object-oriented NumPy implementation of K-Nearest Neighbors supporting both **Classification** and **Regression**.

---

## 1. Architecture & Design

Following scikit-learn's standard estimator conventions:

```
           _BaseKNN
         (fit, distance, kneighbors)
           ▲           ▲
           │           │
     KNNClassifier   KNNRegressor
   (predict_proba,   (mean / weighted avg,
    predict, score)   predict, score)
```

### Class Responsibilities:
- **`fit(X, y)`**: Validates shapes and stores training references ($O(1)$ time).
- **`kneighbors(X)`**: Computes vectorized pairwise distances and returns the indices and distances of the $k$ closest neighbors for each query sample.
- **`KNNClassifier`**:
  - `predict_proba(X)`: Computes empirical class distribution across the $k$ neighbors.
  - `predict(X)`: Returns the highest-probability class label.
  - `score(X, y)`: Returns classification accuracy.
- **`KNNRegressor`**:
  - `predict(X)`: Returns arithmetic mean or distance-weighted average of neighbor targets.
  - `score(X, y)`: Returns the $R^2$ coefficient of determination.

---

## 2. Distance Metric Choices

The implementation supports two primary distance functions:
1. **`metric='euclidean'`**:
   $$\|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum (u_j - v_j)^2}$$
2. **`metric='manhattan'`**:
   $$\|\mathbf{u} - \mathbf{v}\|_1 = \sum |u_j - v_j|$$

---

## 3. Weighting Options

1. **`weights='uniform'`**: Each neighbor has equal voting/averaging weight ($w_i = 1$).
2. **`weights='distance'`**: Each neighbor is weighted by the inverse of its distance ($w_i = \frac{1}{d_i + \epsilon}$). Closer neighbors contribute much more than distant neighbors.

---

## 4. End-to-End Pipeline with StandardScaler

Because KNN is distance-based, feature scaling is mandatory. An end-to-end workflow always:
1. Splits raw data into `(X_train, X_test, y_train, y_test)`.
2. Fits `StandardScaler` **only on `X_train`**.
3. Transforms `X_train` and `X_test` with the fitted scaler.
4. Fits and predicts with `KNNClassifier` or `KNNRegressor`.

---

## Summary

- Non-parametric lazy model implemented using pure NumPy.
- Vectorized distance matrix calculation achieves fast inference.
- Predictions match `sklearn.neighbors` estimators.

**Previous:** [Classification & Regression](../classification-and-regression/README.md)

