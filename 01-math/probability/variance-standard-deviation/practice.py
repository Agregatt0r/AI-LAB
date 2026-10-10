# Practice: Mean, Population Variance, and Standard Deviation

import math


def mean(data):
    return sum(data) / len(data)


def variance(data):
    mu = mean(data)
    return sum((x - mu) ** 2 for x in data) / len(data)


def standard_deviation(data):
    return math.sqrt(variance(data))


data = [2, 4, 6, 8, 10]

mu = mean(data)
var = variance(data)
std = standard_deviation(data)

print(f"Data: {data}")
print(f"Mean (μ): {mu}")
print(f"Population Variance (σ²): {var}")
print(f"Standard Deviation (σ): {std:.4f}")
