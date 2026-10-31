# Practice: Finding the Best Split from Scratch
# Searches all features and all adjacent midpoints to maximize Gini Gain.
# Also verifies the Scale Invariance property of decision trees.

import numpy as np


def gini_impurity(y):
    """Compute Gini Impurity for a label vector."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


def find_best_split(X, y):
    """Exhaustively find the feature and threshold that maximize Gini Gain.

    Returns:
        best_feature (int)
        best_threshold (float)
        best_gain (float)
    """
    n_samples, n_features = X.shape
    parent_gini = gini_impurity(y)

    best_gain = -1.0
    best_feature = None
    best_threshold = None

    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        unique_values = np.unique(feature_values)

        # If all values are identical, cannot split this feature
        if len(unique_values) <= 1:
            continue

        # Sort values and compute midpoints as candidate thresholds
        sorted_vals = np.sort(unique_values)
        thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0

        for thresh in thresholds:
            left_mask = feature_values <= thresh
            right_mask = ~left_mask

            y_left = y[left_mask]
            y_right = y[right_mask]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            # Weighted Gini of the split
            n_l, n_r = len(y_left), len(y_right)
            g_split = (n_l / n_samples) * gini_impurity(y_left) + (n_r / n_samples) * gini_impurity(y_right)
            gain = parent_gini - g_split

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = thresh

    return best_feature, best_threshold, best_gain


if __name__ == "__main__":
    # 2D Toy Dataset:
    # Feature 0: Age (18 to 60)
    # Feature 1: Income ($k) (20 to 120)
    # Pattern: People with Income > 50 are Class 1, others are Class 0
    X = np.array([
        [22, 25.0],   # Class 0
        [28, 35.0],   # Class 0
        [45, 40.0],   # Class 0
        [50, 48.0],   # Class 0
        [24, 60.0],   # Class 1
        [32, 75.0],   # Class 1
        [38, 90.0],   # Class 1
        [55, 110.0],  # Class 1
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    feature_names = ["Age", "Income ($k)"]

    print("--- 1. Exhaustive Split Search ---")
    feat_idx, thresh, gain = find_best_split(X, y)

    print(f"Parent Gini Impurity: {gini_impurity(y):.4f}")
    print(f"Best Feature:         {feature_names[feat_idx]} (Feature #{feat_idx})")
    print(f"Best Threshold:       {thresh:.2f}")
    print(f"Maximum Gini Gain:    {gain:.4f}")

    # Verifications:
    # Best split should be Income between 48.0 and 60.0 (threshold = 54.0)
    assert feat_idx == 1, "Best feature should be Income (feature 1)"
    assert np.isclose(thresh, 54.0), f"Threshold should be 54.0, got {thresh}"
    assert np.isclose(gain, 0.5), "Perfect separation should achieve maximum Gini gain of 0.5"

    # 2. Scale Invariance Test
    print("\n--- 2. Scale Invariance Test ---")
    # Multiply Income by 10,000 (convert $k to raw dollars) and add an arbitrary offset
    X_scaled = X.copy()
    X_scaled[:, 1] = X[:, 1] * 10000.0 + 500.0

    feat_idx_s, thresh_s, gain_s = find_best_split(X_scaled, y)
    print(f"Scaled Best Feature:   {feature_names[feat_idx_s]}")
    print(f"Scaled Best Threshold: {thresh_s:.2f}")
    print(f"Scaled Gini Gain:      {gain_s:.4f}")

    # Feature selected must be identical, and threshold should be mapped proportionally
    assert feat_idx_s == 1, "Feature selection must remain invariant to scaling!"
    assert np.isclose(gain_s, 0.5), "Gini gain must remain identical under monotonic scaling!"
    assert np.isclose(thresh_s, 54.0 * 10000.0 + 500.0)

    print("\nDecision tree split finding and scale invariance verified successfully!")

