# Practice: Gini Impurity & Split Evaluation from Scratch
# Formula:
#   Gini = 1 - Σ (p_i)^2
#   Gini_split = (N_L / N) * Gini_L + (N_R / N) * Gini_R
#   Gini_gain = Gini_parent - Gini_split

import numpy as np


def gini_impurity(y):
    """Compute Gini Impurity for a 1D array of class labels."""
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1.0 - np.sum(probabilities ** 2)


def evaluate_split_gini(y_parent, y_left, y_right):
    """Compute weighted Gini impurity and Gini Gain for a candidate binary split."""
    n_total = len(y_parent)
    n_l, n_r = len(y_left), len(y_right)

    if n_l == 0 or n_r == 0:
        return gini_impurity(y_parent), 0.0

    g_parent = gini_impurity(y_parent)
    g_left = gini_impurity(y_left)
    g_right = gini_impurity(y_right)

    # Weighted split Gini
    g_split = (n_l / n_total) * g_left + (n_r / n_total) * g_right
    g_gain = g_parent - g_split

    return g_split, g_gain


if __name__ == "__main__":
    # 1. Test Gini Impurity on Canonical Distributions
    pure_set = np.array([1, 1, 1, 1, 1])
    fifty_fifty = np.array([0, 0, 0, 1, 1, 1])
    unbalanced = np.array([0, 0, 0, 0, 1])
    three_classes = np.array([0, 0, 1, 1, 2, 2])

    g_pure = gini_impurity(pure_set)
    g_fifty = gini_impurity(fifty_fifty)
    g_unbal = gini_impurity(unbalanced)
    g_three = gini_impurity(three_classes)

    print("--- 1. Gini Impurity Values for Various Node Distributions ---")
    print(f"Pure node (5 ones):           G = {g_pure:.4f} (Expected: 0.0000)")
    print(f"50/50 binary mix (3 vs 3):     G = {g_fifty:.4f} (Expected: 0.5000)")
    print(f"Unbalanced (4 zeros, 1 one):   G = {g_unbal:.4f} (Expected: 1 - (0.8^2 + 0.2^2) = 0.3200)")
    print(f"3 Equal Classes (2 of each):   G = {g_three:.4f} (Expected: 1 - 3*(1/3)^2 ≈ 0.6667)")

    assert np.isclose(g_pure, 0.0)
    assert np.isclose(g_fifty, 0.5)
    assert np.isclose(g_unbal, 0.32)
    assert np.isclose(g_three, 2.0 / 3.0)

    # 2. Compare Candidate Splits on a Parent Node
    # Parent node: 10 samples (6 Positive, 4 Negative)
    y_parent = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    g_parent = gini_impurity(y_parent)
    print(f"\nParent Node Gini (6 pos, 4 neg): {g_parent:.4f}")

    # Split A: Isolates 5 pure positives in Left
    # Left: [1, 1, 1, 1, 1], Right: [1, 0, 0, 0, 0]
    y_left_A = np.array([1, 1, 1, 1, 1])
    y_right_A = np.array([1, 0, 0, 0, 0])

    # Split B: Suboptimal split that barely separates classes
    # Left: [1, 1, 1, 0, 0], Right: [1, 1, 1, 0, 0]
    y_left_B = np.array([1, 1, 1, 0, 0])
    y_right_B = np.array([1, 1, 1, 0, 0])

    g_split_A, gain_A = evaluate_split_gini(y_parent, y_left_A, y_right_A)
    g_split_B, gain_B = evaluate_split_gini(y_parent, y_left_B, y_right_B)

    print("\n--- 2. Evaluating Candidate Splits ---")
    print(f"Split A: Weighted Gini = {g_split_A:.4f} | Gini Gain = {gain_A:.4f}")
    print(f"Split B: Weighted Gini = {g_split_B:.4f} | Gini Gain = {gain_B:.4f}")

    # Split A should be selected because it maximizes Gini Gain
    assert gain_A > gain_B, "Split A should have higher gain than Split B!"
    assert np.isclose(g_split_A, 0.16), "Split A weighted Gini should be 0.16"
    assert np.isclose(gain_A, 0.32), "Split A gain should be 0.32"

    print("\nGini impurity and split evaluation successfully verified!")

