# Binary Cross-Entropy (Log Loss)

Linear regression uses MSE as its loss. For classification, MSE combined with a sigmoid produces a non-convex landscape with weak gradients for confident mistakes. The standard loss for logistic regression is **Binary Cross-Entropy (BCE)**, also called **log loss**.

---

## 1. Loss for One Sample

For a label $y \in \{0, 1\}$ and predicted probability $p = P(y=1 \mid \mathbf{x})$:

$$\ell(y, p) = -\big[\,y \log p + (1 - y)\log(1 - p)\,\big]$$

Only one term is active at a time:

- If $y = 1$: $\ell = -\log p$. Loss is small when $p$ is near 1.
- If $y = 0$: $\ell = -\log(1 - p)$. Loss is small when $p$ is near 0.

---

## 2. Loss over the Dataset

$$J(\mathbf{w}, b) = -\frac{1}{n}\sum_{i=1}^{n}\Big[\,y_i \log p_i + (1 - y_i)\log(1 - p_i)\,\Big], \qquad p_i = \sigma(\mathbf{w}^\top \mathbf{x}_i + b)$$

---

## 3. Why the Log Penalizes Confident Errors

| True $y$ | Predicted $p$ | Loss |
|---|---|---|
| 1 | 0.9 | $0.105$ (good) |
| 1 | 0.5 | $0.693$ (unsure) |
| 1 | 0.1 | $2.303$ (confidently wrong) |
| 1 | $10^{-15}$ | $\approx 34.5$ (catastrophic) |

As $p \to 0$ for a positive sample, $-\log p \to \infty$. The model is punished heavily for being confidently wrong.

---

## 4. Connection to Maximum Likelihood

Under the assumption that each $y_i \sim \text{Bernoulli}(p_i)$, the likelihood of the dataset is:

$$\mathcal{L} = \prod_{i=1}^n p_i^{y_i}(1 - p_i)^{1 - y_i}$$

Taking the negative log and dividing by $n$ gives exactly $J(\mathbf{w}, b)$. **Minimizing BCE is maximum likelihood estimation** for logistic regression. This links back to the probability module in `01-math`.

---

## 5. Baseline Check

A model that always predicts $p = 0.5$ has loss:

$$J = -\log 0.5 = \ln 2 \approx 0.693$$

Any useful classifier should achieve a lower BCE than this.

---

## 6. Numerical Stability

$\log 0$ is undefined, so probabilities are clipped to $[\varepsilon, 1 - \varepsilon]$ with a small $\varepsilon$ (for example $10^{-15}$) before taking logs.

---

## 7. Convexity

For logistic regression, BCE is a convex function of $(\mathbf{w}, b)$. Gradient descent therefore cannot get stuck in a poor local minimum, which is a major advantage over the same loss used in deep networks.

---

## Summary

- $\ell(y, p) = -[y\log p + (1-y)\log(1-p)]$ per sample; $J$ is the mean over the dataset.
- It penalizes confident wrong predictions heavily.
- Minimizing BCE equals maximum likelihood under a Bernoulli model.
- Guessing $p = 0.5$ gives $\ln 2$, a useful baseline.

**Previous:** [Decision Boundary](../decision-boundary/README.md) · **Next:** [Gradient Descent for Classification](../gradient-descent/README.md)
