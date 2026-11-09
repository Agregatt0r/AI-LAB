# Prediction Aggregation in Random Forests

Once an ensemble of $B$ diverse decision trees has been trained, how are their individual predictions combined for a new query sample $\mathbf{x}$?

Random Forests use different mathematical aggregation strategies depending on whether the task is **Classification** or **Regression**.

---

## 1. Classification Aggregation

### A. Hard Voting (Majority Voting)
Each individual tree outputs a discrete class label $f_b(\mathbf{x}) \in \{0, 1, \dots, C-1\}$. The forest takes the **statistical mode** (the class with the most votes):

$$\hat{y}_{\text{hard}} = \arg\max_{c} \sum_{b=1}^B \mathbb{1}[f_b(\mathbf{x}) = c]$$

### B. Soft Voting (Probability Averaging — Scikit-Learn Default)
Every individual tree $b$ outputs a class probability vector based on the sample distribution at its terminal leaf node:

$$\mathbf{p}_b(\mathbf{x}) = [P_b(y=0 \mid \mathbf{x}), \; P_b(y=1 \mid \mathbf{x}), \; \dots, \; P_b(y=C-1 \mid \mathbf{x})]$$

The forest computes the **arithmetic average of the probability vectors** across all $B$ trees:

$$P(y = c \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^B P_b(y = c \mid \mathbf{x})$$

The final predicted class is the one with the highest average probability:
$$\hat{y}_{\text{soft}} = \arg\max_c P(y = c \mid \mathbf{x})$$

---

## 2. Why Soft Voting is Superior to Hard Voting

Soft voting allows **highly confident trees to outweigh uncertain trees**:

| Tree | Predicted Class | Probabilities $[P(y=0), P(y=1)]$ | Confidence Level |
|---|:---:|:---:|---|
| Tree 1 | Class 0 | $[0.51, 0.49]$ | Weak / Barely Class 0 |
| Tree 2 | Class 0 | $[0.52, 0.48]$ | Weak / Barely Class 0 |
| Tree 3 | Class 1 | $[0.05, 0.95]$ | **Extremely Confident Class 1** |

- **Hard Voting Outcome:** Class 0 has 2 votes; Class 1 has 1 vote $\implies$ **Predicts Class 0**.
- **Soft Voting Average:**
  - $P(y=0) = \frac{0.51 + 0.52 + 0.05}{3} = \frac{1.08}{3} = \mathbf{0.36}$
  - $P(y=1) = \frac{0.49 + 0.48 + 0.95}{3} = \frac{1.92}{3} = \mathbf{0.64}$
  - $\implies$ **Predicts Class 1!**

Soft voting prevents two marginally uncertain votes from overriding a decisively confident prediction.

---

## 3. Regression Aggregation

In Random Forest Regression, each tree outputs a continuous numerical prediction $f_b(\mathbf{x}) \in \mathbb{R}$.

### 1. Point Prediction (Mean)
The forest output is the arithmetic average across all $B$ trees:

$$\hat{y} = \frac{1}{B} \sum_{b=1}^B f_b(\mathbf{x})$$

### 2. Uncertainty Quantification (Standard Deviation)
Because we have $B$ separate predictions, we can compute the sample standard deviation across trees to gauge how uncertain the forest is about query $\mathbf{x}$:

$$\sigma_{\text{pred}}(\mathbf{x}) = \sqrt{\frac{1}{B} \sum_{b=1}^B (f_b(\mathbf{x}) - \hat{y})^2}$$

- If $\sigma_{\text{pred}}$ is small: All trees agree strongly $\implies$ **High Confidence**.
- If $\sigma_{\text{pred}}$ is large: Trees disagree widely (often due to sparse training data in that region) $\implies$ **High Uncertainty**.

---

## Summary

- **Classification:** Soft voting averages class probabilities and outperforms simple majority vote by factoring in tree confidence.
- **Regression:** Averages continuous outputs and uses inter-tree variance to estimate prediction uncertainty.

**Previous:** [Random Feature Selection](../random-features/README.md) · **Next:** [Feature Importance](../feature-importance/README.md)

