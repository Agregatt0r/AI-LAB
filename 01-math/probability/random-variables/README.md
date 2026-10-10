# Random Variables

A **Random Variable** (usually denoted $X, Y, Z$) is a mathematical rule that assigns a numerical value to each outcome in a sample space $\Omega$:
$$X: \Omega \to \mathbb{R}$$

Instead of describing outcomes as words (e.g. "heads", "tails"), random variables allow us to treat outcomes mathematically.

## 1. Discrete Random Variables

A random variable is **discrete** if it takes on a countable number of distinct values (e.g. $0, 1, 2, \dots$).

- **Probability Mass Function (PMF):**
  $$P(X = x) = p(x)$$
  Properties:
  1. $0 \le P(X = x) \le 1$ for all $x$
  2. $\sum_{x} P(X = x) = 1$

## 2. Continuous Random Variables

A random variable is **continuous** if it can take any value within a continuous interval (e.g. height, temperature, time).

- **Probability Density Function (PDF):**
  A continuous curve $f(x)$ where probability corresponds to the **area under the curve**:
  $$P(a \le X \le b) = \int_a^b f(x) \, dx$$
  Properties:
  1. $f(x) \ge 0$ for all $x$
  2. $\int_{-\infty}^\infty f(x) \, dx = 1$
  3. Probability at an exact single point is zero: $P(X = c) = 0$.

## 3. Cumulative Distribution Function (CDF)

The **CDF** $F(x)$ represents the probability that the random variable $X$ takes a value less than or equal to $x$:
$$F(x) = P(X \le x)$$

- For discrete $X$: $F(x) = \sum_{k \le x} P(X = k)$
- For continuous $X$: $F(x) = \int_{-\infty}^x f(t) \, dt$
- Properties:
  - $0 \le F(x) \le 1$
  - Non-decreasing function
  - $\lim_{x \to -\infty} F(x) = 0$ and $\lim_{x \to \infty} F(x) = 1$
  - $P(a < X \le b) = F(b) - F(a)$

## Random Variables in Machine Learning

- **Classification targets:** Discrete random variables ($y \in \{0, 1\}$ for binary classification, $y \in \{0, 1, \dots, C-1\}$ for multi-class).
- **Regression targets:** Continuous random variables ($y \in \mathbb{R}$).
- **Feature vectors:** Each input feature $x_i$ is treated as a random variable with an underlying distribution.
