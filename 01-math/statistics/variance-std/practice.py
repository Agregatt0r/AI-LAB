# Practice: Sample Variance (with Bessel's correction), Sample Std, and Standard Error

import math


def sample_mean(data):
    return sum(data) / len(data)


def sample_variance(data):
    n = len(data)
    if n < 2:
        raise ValueError("Sample variance requires at least 2 data points.")
    x_bar = sample_mean(data)
    return sum((x - x_bar) ** 2 for x in data) / (n - 1)


def sample_standard_deviation(data):
    return math.sqrt(sample_variance(data))


def standard_error(data):
    s = sample_standard_deviation(data)
    return s / math.sqrt(len(data))


sample_data = [4, 8, 6, 5, 3, 7]
n = len(sample_data)

x_bar = sample_mean(sample_data)
s2 = sample_variance(sample_data)
s = sample_standard_deviation(sample_data)
se = standard_error(sample_data)

print(f"Sample data: {sample_data} (n = {n})")
print(f"Sample Mean (x̄): {x_bar:.4f}")
print(f"Sample Variance (s²): {s2:.4f} (divided by n - 1 = {n - 1})")
print(f"Sample Std Dev (s): {s:.4f}")
print(f"Standard Error (SE = s / √n): {se:.4f}")
