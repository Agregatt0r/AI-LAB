# Entropy & Information Gain

While CART uses **Gini Impurity** by default, other foundational decision tree algorithms (such as **ID3** and **C4.5**) use **Entropy** and **Information Gain** from Information Theory.

---

## 1. Shannon Entropy

Introduced by Claude Shannon in 1948, **Entropy** quantifies the average amount of **uncertainty** or "surprise" in a random variable.

For a set $S$ with $C$ classes:

$$H(S) = -\sum_{i=1}^C p_i \log_2(p_i)$$

where $p_i$ is the proportion of class $i$ in set $S$.

### Mathematical Convention:
When $p_i = 0$, $\lim_{p \to 0^+} p \log_2(p) = 0$. By convention:
$$0 \log_2(0) = 0$$

---

## 2. Key Values in Binary Classification

```
       Entropy H(p) ^
              1.0   |             .-''''-.
                    |           .'        '.
                    |          /            \
                    |         /              \
              0.0   +--------+----------------+--------> p
                   0.0      0.5              1.0
```

1. **Pure Node ($p = 1.0$ or $p = 0.0$):**
   $$H(S) = -(1.0 \log_2 1.0) = -(1.0 \times 0) = \mathbf{0.0}\text{ bits}$$
   Zero uncertainty. The outcome is completely known.

2. **Maximum Uncertainty ($p_1 = 0.5, p_2 = 0.5$):**
   $$H(S) = -\left(0.5 \log_2 0.5 + 0.5 \log_2 0.5\right) = -(0.5(-1) + 0.5(-1)) = \mathbf{1.0}\text{ bit}$$
   Maximum disorder (equivalent to a fair coin toss).

---

## 3. Information Gain ($IG$)

**Information Gain** measures the reduction in entropy (uncertainty) achieved by splitting a parent dataset $S$ into child subsets $S_{\text{left}}$ and $S_{\text{right}}$:

$$IG(S, A) = H(S) - \left[ \frac{|S_L|}{|S|} H(S_L) + \frac{|S_R|}{|S|} H(S_R) \right]$$

### The Splitting Decision:
The algorithm evaluates all candidate splits and picks the one with the **highest Information Gain** (greatest drop in uncertainty).

---

## 4. Gini Impurity vs Entropy Comparison

| Dimension | Gini Impurity | Entropy |
|---|---|---|
| **Formula** | $1 - \sum p_i^2$ | $-\sum p_i \log_2 p_i$ |
| **Origin** | Economics (wealth inequality) | Information Theory (signal uncertainty) |
| **Max Binary Value** | $0.50$ (at $p=0.5$) | $1.00$ (at $p=0.5$) |
| **Computation Speed** | Faster (pure arithmetic) | Slower (requires $\log_2$ evaluations) |
| **Sensitivity** | Balanced across classes | Penalizes impure nodes slightly heavier |
| **Resulting Trees** | Virtually identical in $\approx 98\%$ of datasets |

---

## Summary

- **Entropy** measures uncertainty in bits: $0.0$ (pure) to $1.0$ (equal 50/50 mix).
- **Information Gain** is the parent entropy minus the weighted child entropy.
- Both Gini and Entropy lead to very similar trees; Gini is computationally preferred because it avoids calculating logarithms.

**Previous:** [Gini Impurity](../gini-impurity/README.md) · **Next:** [Split Finding](../split-finding/README.md)

