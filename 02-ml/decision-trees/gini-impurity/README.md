# Gini Impurity

To decide which feature and threshold to split on, a Decision Tree algorithm needs a mathematical metric to evaluate how "pure" or "mixed" a subset of labels is.

The default metric used by CART (Classification and Regression Trees, the algorithm powering `scikit-learn`) is **Gini Impurity**.

---

## 1. Mathematical Definition

**Gini Impurity** measures the probability that a randomly chosen sample from a set would be incorrectly labeled if it were randomly labeled according to the distribution of labels in the set.

For a dataset with $C$ classes:

$$G = 1 - \sum_{i=1}^C p_i^2$$

where $p_i = \frac{N_i}{N}$ is the relative frequency (proportion) of class $i$ in the node.

---

## 2. Interpreting the Extremes

### 1. Perfectly Pure Node ($G = 0.0$)
Every sample in the node belongs to the exact same class:
- $p_1 = 1.0$, all other $p_i = 0$
- $G = 1 - (1.0^2 + 0^2) = 0.0$
- **Ideal outcome:** No further splits are needed; this becomes a leaf node.

### 2. Maximum Impurity in Binary Classification ($G = 0.5$)
An even 50/50 mixture of two classes:
- $p_1 = 0.5, p_2 = 0.5$
- $G = 1 - (0.5^2 + 0.5^2) = 1 - (0.25 + 0.25) = 0.50$
- **Worst outcome:** Complete uncertainty; coin-flip distribution.

### 3. General Multi-Class Maximum Impurity
For $C$ equally distributed classes ($p_i = \frac{1}{C}$):
$$G_{\max} = 1 - \sum_{i=1}^C \left(\frac{1}{C}\right)^2 = 1 - \frac{1}{C}$$
- For 2 classes: $1 - 1/2 = 0.50$
- For 3 classes: $1 - 1/3 \approx 0.67$
- For 10 classes: $1 - 1/10 = 0.90$

---

## 3. Evaluating a Split: Weighted Gini Impurity

When a node with $N$ samples is partitioned into a **Left child** ($N_L$ samples) and a **Right child** ($N_R$ samples), the quality of the split is the weighted average of the child impurities:

$$G_{\text{split}} = \frac{N_L}{N} G_L + \frac{N_R}{N} G_R$$

### Gini Gain (Impurity Reduction):
The improvement achieved by the split is:

$$\Delta G = G_{\text{parent}} - G_{\text{split}}$$

The algorithm evaluates candidate splits and greedily selects the one that **maximizes Gini Gain** (minimizes $G_{\text{split}}$).

---

## 4. Worked Numerical Example

Suppose a parent node has 10 samples: 6 Positive (+), 4 Negative (-).

### Parent Gini:
$$p_+ = \frac{6}{10} = 0.6, \quad p_- = \frac{4}{10} = 0.4$$
$$G_{\text{parent}} = 1 - (0.6^2 + 0.4^2) = 1 - (0.36 + 0.16) = 0.48$$

### Candidate Split A:
- **Left (5 samples):** 5 (+), 0 (-) $\implies G_L = 1 - (1.0^2 + 0^2) = 0.0$ (Pure!)
- **Right (5 samples):** 1 (+), 4 (-) $\implies G_R = 1 - (0.2^2 + 0.8^2) = 1 - (0.04 + 0.64) = 0.32$

$$G_{\text{split A}} = \frac{5}{10}(0.0) + \frac{5}{10}(0.32) = 0.16$$
$$\Delta G_A = 0.48 - 0.16 = \mathbf{0.32}$$

Split A creates a completely pure left child and reduces impurity from $0.48$ down to $0.16$, yielding a substantial Gini Gain of $0.32$.

---

## Summary

- Gini Impurity ranges from $0.0$ (completely pure) to $0.5$ (50/50 binary mix).
- A split is evaluated by the sample-weighted average of child Gini values.
- The tree greedily maximizes Gini Gain $\Delta G = G_{\text{parent}} - G_{\text{split}}$.

**Previous:** [Tree Structure](../tree-structure/README.md) · **Next:** [Entropy & Information Gain](../entropy-and-information-gain/README.md)

