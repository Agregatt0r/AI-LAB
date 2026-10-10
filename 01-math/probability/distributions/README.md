# Probability Distributions

A **probability distribution** describes the mathematical probabilities of occurrence of the possible outcomes for a random variable.

---

## 1. Bernoulli Distribution
Models a single trial with binary outcomes: Success ($X = 1$) with probability $p$, or Failure ($X = 0$) with probability $1 - p$.

- **PMF:**
  $$P(X = k) = p^k (1 - p)^{1 - k}, \quad k \in \{0, 1\}$$
- **Mean:** $E[X] = p$
- **Variance:** $\text{Var}(X) = p(1 - p)$
- **ML context:** Logistic Regression output, binary cross-entropy loss.

---

## 2. Binomial Distribution
Models the number of successes $k$ across $n$ independent Bernoulli trials with probability $p$.

- **PMF:**
  $$P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k} = \frac{n!}{k!(n - k)!} p^k (1 - p)^{n - k}$$
- **Mean:** $E[X] = n p$
- **Variance:** $\text{Var}(X) = n p (1 - p)$
- **ML context:** Click-through rate modeling, A/B test conversion counts.

---

## 3. Uniform Distribution
All outcomes in a given interval $[a, b]$ have equal probability.

- **Continuous PDF:**
  $$f(x) = \frac{1}{b - a}, \quad a \le x \le b$$
- **Mean:** $E[X] = \frac{a + b}{2}$
- **Variance:** $\text{Var}(X) = \frac{(b - a)^2}{12}$
- **ML context:** Weight initialization (e.g. Xavier/Glorot Uniform).

---

## 4. Normal (Gaussian) Distribution
The symmetric, bell-shaped distribution that arises naturally due to the **Central Limit Theorem**.

- **PDF:**
  $$f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)$$
- **Parameters:** Mean $\mu$ (center) and Variance $\sigma^2$ (spread).
- **Standard Normal:** When $\mu = 0$ and $\sigma = 1$, denoted $\mathcal{N}(0, 1)$ or $Z$.
- **68-95-99.7 Empirical Rule:**
  - $\approx 68.27\%$ of data lies within $[\mu - \sigma, \mu + \sigma]$
  - $\approx 95.45\%$ of data lies within $[\mu - 2\sigma, \mu + 2\sigma]$
  - $\approx 99.73\%$ of data lies within $[\mu - 3\sigma, \mu + 3\sigma]$
- **ML context:** Gaussian Naive Bayes, Linear Regression error assumptions, VAE latent space, He/Normal weight initialization.
