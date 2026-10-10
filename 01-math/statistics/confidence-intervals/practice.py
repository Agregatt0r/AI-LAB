# Practice: Computing Confidence Intervals for the Mean

import math


def confidence_interval(data, confidence=0.95):
    n = len(data)
    if n < 2:
        raise ValueError("Need at least 2 data points.")

    x_bar = sum(data) / n
    s = math.sqrt(sum((x - x_bar) ** 2 for x in data) / (n - 1))
    se = s / math.sqrt(n)

    # Standard normal critical values (z*)
    z_critical = {
        0.90: 1.645,
        0.95: 1.960,
        0.99: 2.576,
    }

    z_star = z_critical.get(confidence, 1.960)
    margin_of_error = z_star * se

    lower_bound = x_bar - margin_of_error
    upper_bound = x_bar + margin_of_error

    return x_bar, margin_of_error, (lower_bound, upper_bound)


# Sample model inference latencies (in milliseconds)
latencies_ms = [45, 52, 48, 50, 54, 49, 47, 51, 53, 46, 50, 52]

mean_val, me, (ci_low, ci_high) = confidence_interval(latencies_ms, confidence=0.95)

print(f"Sample Size (n): {len(latencies_ms)}")
print(f"Sample Mean Latency (x̄): {mean_val:.2f} ms")
print(f"Margin of Error (95%): ±{me:.2f} ms")
print(f"95% Confidence Interval for True Mean: [{ci_low:.2f} ms, {ci_high:.2f} ms]")
