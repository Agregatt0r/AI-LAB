# Practice: Discrete Random Variable, PMF, and CDF

# Consider rolling a biased 4-sided die:
# X = {1, 2, 3, 4}
pmf = {
    1: 0.1,
    2: 0.2,
    3: 0.4,
    4: 0.3,
}

# 1. Verify PMF property: sum(P(X = x)) == 1
total_prob = sum(pmf.values())
print(f"Sum of probabilities: {total_prob:.4f}")  # 1.0000


# 2. Compute Cumulative Distribution Function (CDF): F(x) = P(X <= x)
def cdf(x_val, distribution):
    return sum(prob for outcome, prob in distribution.items() if outcome <= x_val)


print("\nCumulative Distribution Function (CDF):")
for x in sorted(pmf.keys()):
    print(f"F({x}) = P(X <= {x}) = {cdf(x, pmf):.2f}")

# 3. Calculate interval probability P(a < X <= b) = F(b) - F(a)
# Let's find P(1 < X <= 3):
p_1_to_3 = cdf(3, pmf) - cdf(1, pmf)
print(f"\nP(1 < X <= 3) = F(3) - F(1) = {cdf(3, pmf):.2f} - {cdf(1, pmf):.2f} = {p_1_to_3:.2f}")
# Directly: P(X=2) + P(X=3) = 0.2 + 0.4 = 0.6
