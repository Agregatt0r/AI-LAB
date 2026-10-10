# Practice: Key Theoretical Probability Distributions

import math


# 1. Bernoulli PMF: P(X = k) for k in {0, 1}
def bernoulli_pmf(k, p):
    if k not in (0, 1):
        return 0
    return (p ** k) * ((1 - p) ** (1 - k))


# 2. Binomial PMF: P(X = k) = (n choose k) * p^k * (1 - p)^(n - k)
def binomial_pmf(k, n, p):
    comb = math.comb(n, k)
    return comb * (p ** k) * ((1 - p) ** (n - k))


# 3. Gaussian (Normal) PDF: f(x) = (1 / (sigma * sqrt(2*pi))) * exp(-0.5 * ((x - mu)/sigma)^2)
def normal_pdf(x, mu=0.0, sigma=1.0):
    coefficient = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * (((x - mu) / sigma) ** 2)
    return coefficient * math.exp(exponent)


if __name__ == "__main__":
    # Test Bernoulli with p = 0.7
    print("--- 1. Bernoulli Distribution (p = 0.7) ---")
    print(f"P(Success / 1): {bernoulli_pmf(1, 0.7):.2f}")
    print(f"P(Failure / 0): {bernoulli_pmf(0, 0.7):.2f}")

    # Test Binomial: Flipping a coin 10 times (p = 0.5), probability of exactly 5 heads
    print("\n--- 2. Binomial Distribution (n = 10, p = 0.5) ---")
    k = 5
    p_5_heads = binomial_pmf(k, 10, 0.5)
    print(f"Probability of exactly {k} heads in 10 flips: {p_5_heads:.4f} ({p_5_heads * 100:.2f}%)")

    # Test Normal Distribution (mu = 0, sigma = 1) at center x = 0
    print("\n--- 3. Standard Normal PDF (mu = 0, sigma = 1) ---")
    print(f"Density at peak x = 0: {normal_pdf(0, 0, 1):.4f}")
    print(f"Density at 1 std x = 1: {normal_pdf(1, 0, 1):.4f}")
    print(f"Density at 2 std x = 2: {normal_pdf(2, 0, 1):.4f}")
