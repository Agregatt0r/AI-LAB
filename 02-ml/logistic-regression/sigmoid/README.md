# Sigmoid (Logistic) Function

The **sigmoid** function, also called the **logistic function**, maps any real number into the open interval $(0, 1)$. It is the bridge between a linear score and a probability.

---

## 1. Definition

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

where $z = \mathbf{w}^\top \mathbf{x} + b$ is the linear score (also called the **logit** or log-odds).

---

## 2. Key Values

| $z$ | $\sigma(z)$ | Interpretation |
|---|---|---|
| $-\infty$ | $0$ | Certainly class 0 |
| $-2$ | $\approx 0.119$ | Leaning class 0 |
| $0$ | $0.5$ | Maximum uncertainty |
| $2$ | $\approx 0.881$ | Leaning class 1 |
| $+\infty$ | $1$ | Certainly class 1 |

The midpoint $\sigma(0) = 0.5$ is the boundary between the two classes when the default threshold is used.

---

## 3. Properties

1. **Bounded:** $0 < \sigma(z) < 1$ for all real $z$.
2. **Monotonic:** larger $z$ always gives larger $\sigma(z)$.
3. **Symmetric:** $\sigma(-z) = 1 - \sigma(z)$.
4. **Smooth and differentiable everywhere**, which gradient descent requires.

---

## 4. The Derivative

The sigmoid has a particularly convenient derivative:

$$\sigma'(z) = \sigma(z)\,\big(1 - \sigma(z)\big)$$

Derivation with the quotient rule, starting from $\sigma(z) = (1 + e^{-z})^{-1}$:

$$\sigma'(z) = \frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \sigma(z)\,(1 - \sigma(z))$$

This identity is what makes the gradient of the loss simple (see `../gradient-descent`).

---

## 5. Numerical Stability

For very negative $z$, $e^{-z}$ overflows in floating point. A common fix is to clip $z$ to a safe range (for example $[-500, 500]$), since $\sigma$ is already numerically $0$ or $1$ there. The practice script does this.

---

## 6. Geometric Picture

```
 σ(z)
  1.0 |                   ________
      |               ___/
  0.5 |- - - - - - - -/- - - - - - -
      |            /
  0.0 |___________/
      +--------------------------------> z
           -4    0     4
```

---

## Summary

- $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ maps $\mathbb{R} \to (0, 1)$.
- $\sigma(0) = 0.5$, and $\sigma(-z) = 1 - \sigma(z)$.
- $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.

**Previous:** [Classification](../classification/README.md) · **Next:** [Probability & Decision Threshold](../probability-and-threshold/README.md)
