# Practice: Entropy, Information Gain & Comparison with Gini
# Formulas:
#   Entropy: H(S) = - Σ p_i * log2(p_i)
#   Information Gain: IG = H_parent - [(N_L/N)*H_L + (N_R/N)*H_R]
#   Gini: G = 1 - Σ (p_i)^2

import numpy as np


def entropy(y):
    """Compute Shannon Entropy in bits for a 1D array of class labels."""
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)

    # Filter out 0 probabilities to avoid log2(0) error
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))


def information_gain(y_parent, y_left, y_right):
    """Compute Information Gain achieved by a binary split."""
    n_total = len(y_parent)
    n_l, n_r = len(y_left), len(y_right)

    if n_l == 0 or n_r == 0:
        return 0.0

    h_parent = entropy(y_parent)
    h_left = entropy(y_left)
    h_right = entropy(y_right)

    h_split = (n_l / n_total) * h_left + (n_r / n_total) * h_right
    return h_parent - h_split


def gini(y):
    """Compute Gini Impurity."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


if __name__ == "__main__":
    # 1. Test Entropy on Standard Distributions
    pure = np.array([1, 1, 1, 1])
    balanced = np.array([0, 0, 1, 1])
    four_classes = np.array([0, 1, 2, 3])

    h_pure = entropy(pure)
    h_bal = entropy(balanced)
    h_four = entropy(four_classes)

    print("--- 1. Shannon Entropy (Bits) ---")
    print(f"Pure Node (4 ones):           H = {h_pure:.4f} bits (Expected: 0.0000)")
    print(f"Balanced (2 zeros, 2 ones):   H = {h_bal:.4f} bits (Expected: 1.0000)")
    print(f"4 Equal Classes (1 of each):  H = {h_four:.4f} bits (Expected: log2(4) = 2.0000)")

    assert np.isclose(h_pure, 0.0)
    assert np.isclose(h_bal, 1.0)
    assert np.isclose(h_four, 2.0)

    # 2. Information Gain on Candidate Splits
    # Parent: 8 samples [5 positive, 3 negative]
    y_parent = np.array([1, 1, 1, 1, 1, 0, 0, 0])
    h_parent = entropy(y_parent)

    # Split 1: Left=[1, 1, 1, 1], Right=[1, 0, 0, 0]
    y_left_1 = np.array([1, 1, 1, 1])
    y_right_1 = np.array([1, 0, 0, 0])
    ig_1 = information_gain(y_parent, y_left_1, y_right_1)

    # Split 2: Left=[1, 1, 0, 0], Right=[1, 1, 1, 0]
    y_left_2 = np.array([1, 1, 0, 0])
    y_right_2 = np.array([1, 1, 1, 0])
    ig_2 = information_gain(y_parent, y_left_2, y_right_2)

    print(f"\n--- 2. Information Gain Comparison ---")
    print(f"Parent Entropy: {h_parent:.4f} bits")
    print(f"Split 1 Information Gain: {ig_1:.4f} bits (Isolates pure subset)")
    print(f"Split 2 Information Gain: {ig_2:.4f} bits")

    assert ig_1 > ig_2, "Split 1 should produce higher information gain!"

    # 3. Direct Comparison: Gini vs Entropy Across Probabilities p in [0, 1]
    print("\n--- 3. Numerical Comparison: Gini vs Entropy Across Binary Probability p ---")
    print(f"{'p':<6} | {'Gini':<8} | {'Entropy (bits)':<14} | {'Gini * 2':<8}")
    print("-" * 45)

    probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for p in probs:
        # Simulate binary distribution with probability p
        g_val = 2.0 * p * (1.0 - p)
        if p == 0.0 or p == 1.0:
            h_val = 0.0
        else:
            h_val = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
        print(f"{p:<6.1f} | {g_val:<8.4f} | {h_val:<14.4f} | {g_val * 2:<8.4f}")

    print("\nEntropy and Information Gain successfully verified!")

