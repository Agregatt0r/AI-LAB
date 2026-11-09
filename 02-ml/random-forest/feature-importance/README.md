# Feature Importance in Random Forests

One of the greatest practical strengths of Random Forests is **interpretability**: it provides a built-in metric detailing which features contributed most to predicting the target.

The primary method used by Random Forests (and `scikit-learn`) is **Mean Decrease in Impurity (MDI)**, also known as **Gini Importance**.

---

## 1. How Gini Importance (MDI) is Calculated

Every time a node $t$ in a decision tree is split on feature $j$, it reduces the impurity of the data:

$$\Delta I(t) = N_t \cdot I(t) - \left( N_L \cdot I(t_L) + N_R \cdot I(t_R) \right)$$

where:
- $N_t$ is the number of samples arriving at node $t$.
- $I(t)$ is the impurity (Gini Impurity for classification, MSE variance for regression).
- $N_L, N_R$ are the sample counts of the left and right children.

Notice that the impurity drop is **weighted by the number of samples $N_t$** that pass through node $t$:
- Splits near the top (root) of the tree affect more samples and receive massive credit.
- Splits deep in the tree near leaves affect very few samples and receive small credit.

---

## 2. Aggregating Across the Entire Forest

For a forest of $B$ trees:

1. **Sum across all nodes in Tree $b$:**
   $$\text{Importance}_j(T_b) = \sum_{t \in T_b \text{ split on feature } j} \Delta I(t)$$

2. **Average across all $B$ trees:**
   $$\text{Importance}_j = \frac{1}{B} \sum_{b=1}^B \text{Importance}_j(T_b)$$

3. **Normalize to Sum to 1.0 (100%):**
   $$\overline{\text{Importance}}_j = \frac{\text{Importance}_j}{\sum_{k=1}^d \text{Importance}_k}$$

---

## 3. Interpreting Feature Importance

```
   Feature Importance Bar Chart:
   Feature 0 (Income)     |████████████████████████ 52.4%
   Feature 1 (Age)        |█████████████            28.1%
   Feature 2 (Education)  |██████                   14.5%
   Feature 3 (Random ID)  |██                        5.0%
```

- Features with high importance are frequently selected at high levels of the trees and cause large drops in impurity.
- Uninformative or pure noise features are rarely selected, and when chosen, produce negligible impurity drops.

---

## 4. Caveats & Limitations of MDI

While fast to compute, MDI has a known statistical bias:
- **High Cardinality Bias:** Numerical features with many unique values or high-cardinality categorical variables have more candidate split thresholds, giving them an unfair advantage to be chosen over low-cardinality features.
- **Alternative:** **Permutation Feature Importance** (shuffling column $j$ on out-of-bag data and measuring the drop in accuracy) is slower but completely unbiased.

---

## Summary

- Feature importance sums the sample-weighted impurity reduction ($\Delta I$) across all nodes and trees.
- Top-level splits contribute more importance because they affect more samples.
- Values are normalized to sum to $1.0$, providing an immediate ranking of feature relevance.

**Previous:** [Prediction Aggregation](../prediction-aggregation/README.md) · **Next:** [Decision Tree vs Random Forest](../decision-tree-vs-random-forest/README.md)

