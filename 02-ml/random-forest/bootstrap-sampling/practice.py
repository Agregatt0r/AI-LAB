# Practice: Bootstrap Sampling & Out-Of-Bag (OOB) Fraction
# Demonstrates:
#   1. Generating bootstrap samples with replacement
#   2. Mathematical convergence of OOB ratio to 1/e ≈ 0.3679
#   3. OOB evaluation logic across multiple trees

import numpy as np


def create_bootstrap_sample(X, y, random_state=None):
    """Draw a bootstrap sample of size N with replacement from (X, y).

    Returns:
        X_boot, y_boot, oob_indices
    """
    rng = np.random.default_rng(random_state)
    n = len(X)

    # Sample indices with replacement
    boot_indices = rng.choice(n, size=n, replace=True)

    # Out-of-Bag indices: indices NOT present in boot_indices
    unique_boot_indices = set(boot_indices)
    oob_indices = np.array([i for i in range(n) if i not in unique_boot_indices])

    return X[boot_indices], y[boot_indices], oob_indices


def verify_oob_convergence(n_values, n_trials=500):
    """Compute empirical out-of-bag fraction across various dataset sizes."""
    results = {}
    for n in n_values:
        oob_fractions = []
        for trial in range(n_trials):
            boot_idx = np.random.choice(n, size=n, replace=True)
            n_unique = len(np.unique(boot_idx))
            oob_fraction = (n - n_unique) / n
            oob_fractions.append(oob_fraction)
        results[n] = np.mean(oob_fractions)
    return results


if __name__ == "__main__":
    # 1. Inspect a Tiny Toy Example (N = 10)
    print("--- 1. Single Bootstrap Sample on 10 Samples ---")
    X_toy = np.arange(10)[:, np.newaxis]
    y_toy = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    X_boot, y_boot, oob_idx = create_bootstrap_sample(X_toy, y_toy, random_state=42)

    print(f"Original Indices: {list(range(10))}")
    print(f"Bootstrap Sample: {X_boot.ravel().tolist()}")
    print(f"OOB Indices:      {oob_idx.tolist()}")
    print(f"Included Count:   {len(np.unique(X_boot.ravel()))} / 10 ({len(np.unique(X_boot.ravel()))*10}%)")
    print(f"OOB Count:        {len(oob_idx)} / 10 ({len(oob_idx)*10}%)\n")

    # 2. Empirical Verification of 1/e Convergence
    print("--- 2. Empirical OOB Fraction vs Mathematical Limit 1/e ≈ 36.79% ---")
    print(f"{'Dataset Size (N)':<18} | {'Empirical OOB %':<16} | {'(1 - 1/N)^N':<14} | {'Limit (1/e)'}")
    print("-" * 65)

    n_sizes = [5, 10, 50, 100, 500, 1000, 10000]
    emp_results = verify_oob_convergence(n_sizes, n_trials=300)

    for n in n_sizes:
        theoretical_finite = (1.0 - 1.0 / n) ** n
        emp = emp_results[n]
        print(f"N = {n:<14} | {emp * 100:>6.2f}%          | {theoretical_finite * 100:>6.2f}%       | 36.79%")

    limit_1_over_e = 1.0 / np.e
    assert np.isclose(emp_results[10000], limit_1_over_e, atol=0.01), "Empirical OOB must match 1/e!"

    # 3. Simulate OOB Predictions Tracker
    print("\n--- 3. Tracking OOB Availability for Ensemble of 15 Trees ---")
    n_samples = 20
    n_trees = 15
    sample_oob_counts = np.zeros(n_samples, dtype=int)

    for t in range(n_trees):
        _, _, oob_idx = create_bootstrap_sample(np.zeros(n_samples), np.zeros(n_samples))
        sample_oob_counts[oob_idx] += 1

    print(f"Average number of trees evaluating each sample OOB: {np.mean(sample_oob_counts):.1f} / {n_trees} trees")
    print(f"Fraction of trees each sample is OOB for: {np.mean(sample_oob_counts)/n_trees * 100:.1f}% (Expected ~36.8%)")

    assert np.all(sample_oob_counts > 0), "Every sample should be OOB for at least one tree in the forest"

    print("\nBootstrap sampling and Out-of-Bag properties successfully verified!")

