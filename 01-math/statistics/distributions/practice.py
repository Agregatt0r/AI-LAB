# Practice: Z-Score Standardization (StandardScaler from scratch)

import math


def standardize(data):
    n = len(data)
    mean_val = sum(data) / n
    std_val = math.sqrt(sum((x - mean_val) ** 2 for x in data) / (n - 1))

    z_scores = [(x - mean_val) / std_val for x in data]
    return z_scores, mean_val, std_val


raw_features = [170, 180, 165, 175, 190, 160]  # Heights in cm

z_scores, mu, sigma = standardize(raw_features)

print("Original Data:", raw_features)
print(f"Sample Mean: {mu:.2f} cm, Sample Std: {sigma:.2f} cm")
print("\nStandardized Z-Scores (mean=0, std=1):")
for orig, z in zip(raw_features, z_scores):
    print(f"Original: {orig:3d} cm -> Z-Score: {z:+.4f}")
