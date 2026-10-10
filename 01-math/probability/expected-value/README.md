# Expected Value

The **Expected Value** (or mathematical expectation, denoted $E[X]$ or $\mu$) is the long-term average outcome of a random variable weighted by its probabilities.

## Formula

For a **discrete random variable** $X$ with possible outcomes $x_1, x_2, \dots, x_n$ and corresponding probabilities $P(X = x_i)$:

$$E[X] = \sum_{i=1}^{n} x_i \cdot P(X = x_i)$$

Where:
$$\sum_{i=1}^n P(X = x_i) = 1$$

## Properties of Expected Value

1. **Linearity:**
   $$E[aX + b] = aE[X] + b$$
2. **Sum of random variables:**
   $$E[X + Y] = E[X] + E[Y]$$

## Example: Fair 6-Sided Die

Outcomes: $X \in \{1, 2, 3, 4, 5, 6\}$, each with probability $P(X = x) = \frac{1}{6}$.

$$E[X] = 1\left(\frac{1}{6}\right) + 2\left(\frac{1}{6}\right) + 3\left(\frac{1}{6}\right) + 4\left(\frac{1}{6}\right) + 5\left(\frac{1}{6}\right) + 6\left(\frac{1}{6}\right)$$
$$E[X] = \frac{1 + 2 + 3 + 4 + 5 + 6}{6} = \frac{21}{6} = 3.5$$

## Expected Value in ML

- **Loss Minimization:** In supervised learning, we seek to minimize the expected loss / risk: $\min_\theta E_{(x, y)}[L(f_\theta(x), y)]$.
- **Reinforcement Learning:** The state-value function $V(s) = E[G_t \mid S_t = s]$ represents the expected cumulative reward.
