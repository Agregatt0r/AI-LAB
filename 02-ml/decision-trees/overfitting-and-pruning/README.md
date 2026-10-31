# Overfitting and Pruning in Decision Trees

Because Decision Trees make no geometric assumptions about data distribution and can create arbitrary orthogonal splits, they are **among the most prone to overfitting of all machine learning models**.

---

## 1. Why Decision Trees Overfit

When a tree is allowed to grow without depth or leaf size constraints:
1. It creates narrow, highly complex rectangular slices around individual noisy data points.
2. It continues splitting until **every single leaf is completely pure** ($G = 0.0$).
3. **The result:** $100\%$ training accuracy, but very poor generalization to new, unseen test data (high variance, low bias).

```
         Overfitting (Unconstrained Tree)                      Pruned Tree (Good Generalization)
      y ^   +-------------+                                  y ^
        |   |   ▲   ▲   ▲ |                                    |          ▲   ▲   ▲
        |   | +---+       | (Carves a special box              |   -----------------------
        |   | | ■ |  ▲    |  for an outlier)                   |          ■   ■   ■
        |   | +---+       |                                    |
        +---+-------------+--------> x                         +-------------------------> x
```

---

## 2. Techniques to Combat Overfitting: Pruning (Обрізка)

Pruning reduces the complexity of the tree by removing branches that provide little predictive power. There are two primary paradigms:

### A. Pre-Pruning (Early Stopping)
Halts the tree growth during the recursive splitting process before it overfits.
- **Rules:**
  - Cap maximum depth (`max_depth = 3-5`).
  - Require a minimum number of samples to split (`min_samples_split = 10-20`).
  - Require a minimum number of samples per leaf (`min_samples_leaf = 5-10`).
  - Require a minimum gain threshold (`min_impurity_decrease = 0.01`).
- **Pros:** Fast and computationally efficient.
- **Cons:** Risk of premature stopping (the algorithm might stop before discovering useful multi-feature interactions, like XOR problems).

### B. Post-Pruning (Cost-Complexity Pruning)
Grows the complete unconstrained tree first, then prunes branches back from the bottom up.

The standard algorithm is **Minimal Cost-Complexity Pruning** (parameterized by $\alpha \geq 0$ in CART):

$$R_\alpha(T) = R(T) + \alpha |T|$$

where:
- $R(T)$ is the total training error of tree $T$.
- $|T|$ is the number of terminal leaf nodes.
- $\alpha$ (ccp_alpha) is the **complexity penalty**:
  - When $\alpha = 0$: The full, unpruned tree is kept.
  - When $\alpha \to \infty$: The tree is pruned all the way back to a single root leaf node.

---

## 3. Pre-Pruning vs Post-Pruning Comparison

| Feature | Pre-Pruning (Early Stopping) | Post-Pruning (Cost-Complexity) |
|---|---|---|
| **Approach** | Stop growth during construction | Grow full tree, then prune backwards |
| **Speed** | Very fast (avoids building deep branches) | Slower (builds full tree first) |
| **Risk** | May stop too early (horizon effect) | Finds optimal subtrees reliably |
| **Implementation** | `max_depth`, `min_samples_leaf` | `ccp_alpha` in `scikit-learn` |

---

## Summary

- Unconstrained decision trees memorize noise and achieve 100% training accuracy with poor test performance.
- **Pre-pruning** stops growth early using `max_depth` and `min_samples_leaf`.
- **Post-pruning** trims branches after growing by balancing classification error against tree size ($R_\alpha(T) = R(T) + \alpha |T|$).

**Previous:** [Hyperparameters](../hyperparameters/README.md) · **Next:** [Decision Tree From Scratch](../from-scratch/README.md)

