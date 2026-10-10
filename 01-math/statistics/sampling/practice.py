# Practice: Sampling, Law of Large Numbers, and Central Limit Theorem

import random


# 1. Simulate rolling a fair die multiple times (True population mean = 3.5)
def simulate_dice_rolls(n_rolls):
    rolls = [random.randint(1, 6) for _ in range(n_rolls)]
    sample_mean = sum(rolls) / len(rolls)
    return sample_mean


# Demonstrate Law of Large Numbers (LLN)
random.seed(42)
print("--- Law of Large Numbers (True mean = 3.5) ---")
for n in [5, 50, 500, 5000, 50000]:
    avg = simulate_dice_rolls(n)
    print(f"Sample size n={n:5d} -> Sample mean = {avg:.4f}")


# 2. Demonstrate Central Limit Theorem (CLT)
# Take 1000 samples of size n=30 from a non-normal (uniform die roll) distribution
def simulate_clt(num_samples=1000, sample_size=30):
    sample_means = [simulate_dice_rolls(sample_size) for _ in range(num_samples)]
    overall_mean = sum(sample_means) / len(sample_means)
    variance_means = sum((m - overall_mean) ** 2 for m in sample_means) / len(sample_means)
    std_error = variance_means ** 0.5
    return overall_mean, std_error


mean_of_means, se = simulate_clt()
print("\n--- Central Limit Theorem (1000 samples of size n=30) ---")
print(f"Mean of sample means: {mean_of_means:.4f} (Expected: 3.5000)")
print(f"Standard Error of means: {se:.4f}")
