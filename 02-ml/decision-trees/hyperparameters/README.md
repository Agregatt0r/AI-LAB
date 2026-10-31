# Decision Tree Hyperparameters

Left unconstrained, a decision tree will continue splitting until every single training sample is isolated into its own leaf node ($G = 0$), memorizing noisy outliers and resulting in severe **overfitting**.

Hyperparameters in Decision Trees act as **pre-pruning** constraints: stopping rules that halt tree growth early to ensure the model captures genuine patterns rather than noise.

---

## 1. Key Hyperparameters

| Hyperparameter | Meaning | Default (sklearn) | Effect of Increasing |
|---|---|:---:|---|
| **`max_depth`** | Maximum length of path from root to leaf | `None` (unlimited) | **Increases variance** (deeper, more complex tree) |
| **`min_samples_split`** | Minimum samples required in a node to attempt a split | `2` | **Increases bias** (restricts small splits, smoother) |
| **`min_samples_leaf`** | Minimum samples required to exist in any leaf | `1` | **Increases bias** (prevents leaves dedicated to single outliers) |
| **`max_leaf_nodes`** | Maximum number of terminal leaf nodes | `None` (unlimited) | **Increases variance** (allows more partitions) |
| **`min_impurity_decrease`** | Minimum gain ($\Delta G$) required to justify splitting | `0.0` | **Increases bias** (stops splits that offer minimal gain) |
| **`max_features`** | Number of features considered at each split | `None` (all $d$) | Random subset (decorrelates trees in Random Forests) |

---

## 2. Deep Dive: Core Parameters

### 1. `max_depth`
Controls the maximum number of decision levels:
- `max_depth = 1`: Known as a **Decision Stump**. A single split on one feature (very high bias, underfitting).
- `max_depth = 3-5`: Typically ideal for tabular datasets with few features; highly interpretable.
- `max_depth = None`: Tree grows until all leaves are pure or contain fewer than `min_samples_split` samples (extreme risk of overfitting).

```
   max_depth = 1 (Stump)             max_depth = 2
          (Root)                         (Root)
          /    \                         /    \
      [Leaf]  [Leaf]                 (Node)  (Node)
                                     /   \   /   \
                                   [L]  [L] [L]  [L]
```

### 2. `min_samples_split` vs `min_samples_leaf`
- **`min_samples_split`** checks the node **before** splitting:
  - If node has $N = 4$ samples and `min_samples_split = 5`, the node will **not** split; it immediately becomes a leaf.
- **`min_samples_leaf`** checks the candidate children **after** splitting:
  - If a split would create a left child with $1$ sample and a right child with $9$ samples, and `min_samples_leaf = 2`, the split is rejected.
  - Highly effective at suppressing single-sample outlier memorization.

---

## 3. Tuning Hyperparameters: Bias-Variance Control

To combat overfitting, adjust hyperparameters in the direction of **regularization (simplification)**:
- **Decrease:** `max_depth`, `max_leaf_nodes`.
- **Increase:** `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease`.

In practice, `max_depth` and `min_samples_leaf` are the two most impactful hyperparameters to tune via cross-validation.

---

## Summary

- Unconstrained trees grow until leaves are pure, leading to massive overfitting.
- Pre-pruning parameters (`max_depth`, `min_samples_split`, `min_samples_leaf`) enforce early stopping.
- Restricting depth and leaf size prevents the tree from creating specialized rules for noise.

**Previous:** [Split Finding](../split-finding/README.md) · **Next:** [Overfitting & Pruning](../overfitting-and-pruning/README.md)

