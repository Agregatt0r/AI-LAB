# Random Forest with scikit-learn & From Scratch

`scikit-learn` provides highly optimized implementations of Random Forests:
- `RandomForestClassifier` for classification
- `RandomForestRegressor` for regression

This module covers the complete `scikit-learn` API, key hyperparameter tuning, out-of-bag evaluation, and verifies a complete pure NumPy implementation from scratch against `scikit-learn`.

---

## 1. The scikit-learn API Overview

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,       # Number of trees in the forest
    criterion="gini",       # "gini", "entropy", or "log_loss"
    max_depth=None,         # Maximum tree depth
    max_features="sqrt",    # Number of features to consider per split
    bootstrap=True,         # Train each tree on a bootstrap sample
    oob_score=True,         # Calculate Out-Of-Bag validation score
    n_jobs=-1,              # Parallelize across all available CPU cores
    random_state=42
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_probs = rf.predict_proba(X_test)
oob_acc = rf.oob_score_
importances = rf.feature_importances_
```

---

## 2. Essential Hyperparameters

| Hyperparameter | Default | Recommended Tuning Guide |
|---|:---:|---|
| **`n_estimators`** | `100` | Start with `100`. Increase to `200-500` for final models. Larger values never hurt accuracy, only increase compute time. |
| **`max_features`** | `"sqrt"` | `"sqrt"` for classification ($\sqrt{d}$), `1.0` or `1.0/3.0` for regression. Decreasing `max_features` reduces tree correlation. |
| **`max_depth`** | `None` | `None` works well in most cases; set to `10-20` if memory is constrained. |
| **`min_samples_leaf`** | `1` | Increase to `2-5` on very noisy datasets to prevent trees from fitting individual outliers. |
| **`oob_score`** | `False` | Set to `True` to get free cross-validation metrics without creating a separate validation set. |
| **`n_jobs`** | `None` | Set to `-1` to utilize all CPU cores in parallel for massive speedups. |

---

## 3. Parallelization: Why Random Forest Scales

Unlike Boosting (which must train trees sequentially: Tree 2 depends on Tree 1), **Random Forest is embarrassingly parallel**:
- Every tree is completely independent of every other tree.
- `n_jobs=-1` distributes trees across all CPU cores simultaneously, achieving linear speedups during training.

---

## Summary

- `RandomForestClassifier` and `RandomForestRegressor` provide fast, parallelized, production-grade ensemble learning.
- `oob_score=True` provides built-in validation accuracy without holding out data.
- Pure NumPy from-scratch implementations mirror the same mechanics and achieve comparable accuracy.

**Previous:** [Decision Tree vs Random Forest](../decision-tree-vs-random-forest/README.md)

