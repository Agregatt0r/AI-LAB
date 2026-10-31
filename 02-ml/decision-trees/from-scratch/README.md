# Decision Tree From Scratch

This module brings together every concept from the previous modules into a complete, clean, object-oriented NumPy implementation of a **Decision Tree Classifier**.

---

## 1. Class Design & Architecture

Following the familiar `scikit-learn` API conventions:

```
            DecisionTreeClassifier
  ├── fit(X, y)             # Starts recursive tree growth
  │    └── _grow_tree()     # Evaluates candidate splits & creates Nodes
  ├── predict(X)            # Traverses tree to leaf for each sample
  ├── predict_proba(X)      # Returns empirical class distribution at leaf
  ├── score(X, y)           # Returns classification accuracy
  └── print_tree()          # Prints ASCII hierarchy of decision rules
```

---

## 2. The Recursive Splitting Algorithm

The recursive method `_grow_tree(X, y, depth)` follows these steps:

1. **Check Stopping Conditions (Pre-Pruning):**
   - Has `depth` reached `max_depth`?
   - Does `y` contain fewer than `min_samples_split` samples?
   - Is the node completely pure ($G = 0.0$ or $H = 0.0$)?
   - If **yes**, return a **Leaf Node** holding the majority class and class probabilities.

2. **Find Best Split:**
   - Scan every feature $j \in \{0, \dots, d-1\}$.
   - For every continuous feature, evaluate candidate thresholds $\theta = \frac{v_k + v_{k+1}}{2}$.
   - Compute Gini Gain (or Information Gain).
   - If no split achieves positive gain, return a **Leaf Node**.

3. **Recurse:**
   - Partition samples into left child ($x_j \leq \theta$) and right child ($x_j > \theta$).
   - Recursively call `_grow_tree` on left and right subsets.
   - Return an **Internal Node** referencing left and right children.

---

## 3. Fast Prediction via Traversal

To classify an input $\mathbf{x}$:
- At each internal node, check whether $x[j] \leq \theta$.
- Branch left if True, right if False.
- Once a leaf is reached, return the stored majority class or empirical probability vector $\mathbf{p}$.
- Complexity per query is $O(\text{depth})$.

---

## 4. Comparing with scikit-learn

The custom implementation matches the behavior of `sklearn.tree.DecisionTreeClassifier(criterion='gini', max_depth=...)` because both implement the standard CART binary recursive splitting algorithm.

---

## Summary

- Complete recursive tree implementation using only Python and NumPy.
- Supports both `gini` and `entropy` impurity criteria.
- Provides human-readable tree visualization and exact matching performance with scikit-learn.

**Previous:** [Overfitting & Pruning](../overfitting-and-pruning/README.md)

