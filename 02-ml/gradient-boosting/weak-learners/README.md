# Weak Learners: Decision Stumps and Shallow Trees

In boosting, **weak learners** are intentionally simple models — specifically chosen for their **high bias** (they underfit) rather than their power. The most common choice is a **Decision Stump** (a single-split decision tree with depth = 1) or a **Shallow Decision Tree** (depth 2–5).

---

## 🎯 Why Intentional Weakness is a Strength

This seems counterintuitive: why deliberately use bad models? The answer lies in the **Bias-Variance Decomposition** of prediction error:

$$\text{MSE}(x) = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

| Model Property | Bias | Variance | Use in Boosting |
|---|---|---|---|
| **Deep tree (overfit)** | Very Low | **Very High** | ❌ Dangerous: individual tree overfits; boosting can compound variance errors |
| **Shallow tree / stump** | **High** | Low | ✅ Ideal: sequential stages systematically reduce the persistent bias |
| **Linear model** | High | Low | ✅ Works, but limited: cannot capture non-linear interactions |

### The key insight:
- **Bagging** (Random Forest) reduces **variance** by averaging many high-variance trees.
- **Boosting** reduces **bias** by sequentially correcting the systematic errors of low-variance base learners.

---

## 🌳 Decision Stump: The Canonical Weak Learner

A decision stump partitions the input space with a **single axis-aligned hyperplane** at threshold $\theta$ along feature $j$:

$$h(x) = \begin{cases} +1 & \text{if } x_j \ge \theta \\ -1 & \text{if } x_j < \theta \end{cases}$$

The optimal threshold $\theta^*$ is found by exhaustive search over all midpoints between adjacent unique values of feature $j$ across all features, minimizing weighted classification error:

$$\epsilon = \sum_{i=1}^{N} w_i \cdot \mathbf{1}[h(x_i) \ne y_i]$$

**Stump characteristics:**
- Cannot model interactions between features.
- Cannot fit any function requiring more than a single comparison.
- High bias: a stump on a Gaussian dataset achieves only $\approx 60\%$ accuracy.
- Zero variance: the stump structure is fully deterministic given data and weights.

---

## 🔄 How Diversity Emerges: Boosting vs Bagging

Both Random Forest and Gradient Boosting build an ensemble of trees, but diversity arises from completely different mechanisms:

| Property | Random Forest (Bagging) | Gradient Boosting |
|---|---|---|
| **Training parallelism** | Parallel (all trees independent) | Sequential (tree $m$ depends on tree $m-1$) |
| **Source of diversity** | Random bootstrap samples + feature subsets | Different **target signals**: tree $m$ fits the residuals left by trees $1 \dots m-1$ |
| **Individual tree accuracy** | High (deep trees, low bias) | Low (shallow trees / stumps, high bias by design) |
| **Error reduction mechanism** | Variance reduction through averaging | Bias reduction through sequential correction |
| **Data weighting** | Equal weights (uniform bootstrap) | Adaptive: misclassified / high-residual points get implicitly higher attention |

---

## 📏 Optimal Weak Learner Depth in Practice

Gradient Boosting performance is highly sensitive to the depth of base trees:

| `max_depth` | Model Type | Bias | Variance | Typical Performance |
|---|---|---|---|---|
| **1** | Decision Stump | Very High | Very Low | Good for well-separated data; slow convergence |
| **2–4** | Shallow Tree | Moderate | Low | **Optimal sweet spot** for most real-world GBM |
| **5–8** | Medium Tree | Low | Moderate | Faster convergence but requires lower learning rate |
| **> 10** | Deep Tree | Very Low | High | Overfits aggressively; defeats the purpose of boosting |

> [!TIP]
> The scikit-learn default of `max_depth=3` and XGBoost's default of `max_depth=6` are both empirically validated sweet spots across thousands of real-world datasets.

