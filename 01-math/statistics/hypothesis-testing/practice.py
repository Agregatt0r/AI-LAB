# Practice: One-Sample Hypothesis Testing (Z-Test) from scratch

import math


def one_sample_z_test(sample_data, null_mean, alpha=0.05):
    n = len(sample_data)
    x_bar = sum(sample_data) / n
    s = math.sqrt(sum((x - x_bar) ** 2 for x in sample_data) / (n - 1))
    se = s / math.sqrt(n)

    # 1. Compute Z-statistic
    z_stat = (x_bar - null_mean) / se

    # 2. Approximate two-tailed p-value using standard normal CDF approximation
    # Error function approximation: math.erf(x)
    p_one_tail = 0.5 * (1.0 - math.erf(abs(z_stat) / math.sqrt(2)))
    p_value = 2.0 * p_one_tail

    # 3. Decision
    reject_null = p_value <= alpha

    return {
        "sample_mean": x_bar,
        "null_mean": null_mean,
        "z_statistic": z_stat,
        "p_value": p_value,
        "reject_null": reject_null,
    }


# Scenario: A new ML optimization claims to reduce API latency below historical baseline of 100ms
# H0: mu = 100 ms (no improvement)
# Ha: mu != 100 ms (latency is significantly different)
test_latencies = [92, 88, 95, 90, 85, 89, 93, 87, 91, 84, 88, 90, 86, 92, 89]

results = one_sample_z_test(test_latencies, null_mean=100, alpha=0.05)

print(f"Sample Mean (x̄):     {results['sample_mean']:.2f} ms")
print(f"Null Hypothesis (μ0): {results['null_mean']:.2f} ms")
print(f"Z-Statistic:          {results['z_statistic']:.4f}")
print(f"P-Value:              {results['p_value']:.6f}")
print(f"Significance Level α: 0.05")

if results["reject_null"]:
    print("Decision: REJECT H0 -> The improvement is statistically significant!")
else:
    print("Decision: FAIL TO REJECT H0 -> Insufficient evidence of improvement.")
