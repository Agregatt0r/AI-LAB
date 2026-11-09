# Practice: Computing Feature Importance (Mean Decrease in Impurity) from Scratch
# Formula for node t split on feature j:
#   ΔI(t) = N_t * Gini(t) - [N_L * Gini(t_L) + N_R * Gini(t_R)]
#   Importance(j) = Σ ΔI(t) for all nodes split on j across all trees
#   Normalized Importance(j) = Importance(j) / Σ Importance(k)

import numpy as np


class TreeWithImportance:
    def __init__(self, max_depth=4, max_features="sqrt", random_state=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)
        self.feature_importances_ = None
        self.tree_ = None

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        return 1.0 - np.sum((counts / len(y)) ** 2)

    def _build(self, X, y, importances, depth=0):
        n_samples, n_features = X.shape
        parent_gini = self._gini(y)

        if len(np.unique(y)) <= 1 or (self.max_depth and depth >= self.max_depth):
            classes, counts = np.unique(y, return_counts=True)
            return {"leaf": True, "value": classes[np.argmax(counts)]}

        # Sample m features
        m = max(1, int(np.sqrt(n_features)))
        candidates = self.rng.choice(n_features, size=m, replace=False)

        best_gain = -1.0
        best_feat, best_thresh = None, None
        best_left_mask = None

        for j in candidates:
            vals = np.sort(np.unique(X[:, j]))
            if len(vals) <= 1:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2.0

            for t in thresholds:
                left_mask = X[:, j] <= t
                n_l = np.sum(left_mask)
                n_r = n_samples - n_l
                if n_l == 0 or n_r == 0:
                    continue

                g_split = (n_l / n_samples) * self._gini(y[left_mask]) + (n_r / n_samples) * self._gini(y[~left_mask])
                gain = parent_gini - g_split

                if gain > best_gain:
                    best_gain = gain
                    best_feat = j
                    best_thresh = t
                    best_left_mask = left_mask

        if best_feat is None or best_gain <= 0.0:
            classes, counts = np.unique(y, return_counts=True)
            return {"leaf": True, "value": classes[np.argmax(counts)]}

        # Track weighted impurity decrease for best_feat:
        # ΔI(t) = N_t * Gini_parent - (N_L * Gini_L + N_R * Gini_R) = N_t * gain
        impurity_decrease = n_samples * best_gain
        importances[best_feat] += impurity_decrease

        return {
            "leaf": False,
            "feat": best_feat,
            "thresh": best_thresh,
            "left": self._build(X[best_left_mask], y[best_left_mask], importances, depth + 1),
            "right": self._build(X[~best_left_mask], y[~best_left_mask], importances, depth + 1),
        }

    def fit(self, X, y):
        n_features = X.shape[1]
        raw_importances = np.zeros(n_features)
        self.tree_ = self._build(X, y, raw_importances, depth=0)
        self.feature_importances_ = raw_importances
        return self


def compute_forest_feature_importances(X, y, n_trees=30, random_state=42):
    """Train Random Forest and aggregate normalized feature importances."""
    rng = np.random.default_rng(random_state)
    n_samples, n_features = X.shape

    total_importances = np.zeros(n_features)

    for t in range(n_trees):
        # Bootstrap sample
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        tree = TreeWithImportance(max_depth=5, random_state=t)
        tree.fit(X[boot_idx], y[boot_idx])
        total_importances += tree.feature_importances_

    # Normalize across features so they sum to 1.0 (100%)
    if np.sum(total_importances) > 0:
        normalized_importances = total_importances / np.sum(total_importances)
    else:
        normalized_importances = np.ones(n_features) / n_features

    return normalized_importances


if __name__ == "__main__":
    # Generate Synthetic Dataset with 4 Features:
    # Feature 0: Strong dominant predictor (weight = 3.0)
    # Feature 1: Moderate predictor (weight = 1.5)
    # Feature 2: Weak predictor (weight = 0.5)
    # Feature 3: Pure random noise (weight = 0.0)
    rng = np.random.default_rng(42)
    n = 300

    f0 = rng.standard_normal(n)
    f1 = rng.standard_normal(n)
    f2 = rng.standard_normal(n)
    f3_noise = rng.standard_normal(n)  # Irrelevant noise

    X = np.column_stack([f0, f1, f2, f3_noise])
    latent = 3.0 * f0 + 1.5 * f1 + 0.5 * f2
    y = (latent > 0.0).astype(int)

    feature_names = ["Feature 0 (Strong)", "Feature 1 (Moderate)", "Feature 2 (Weak)", "Feature 3 (Noise)"]

    # Compute Feature Importances across 30 Trees
    importances = compute_forest_feature_importances(X, y, n_trees=30, random_state=42)

    print("--- 1. Random Forest Mean Decrease in Impurity (MDI) ---")
    for name, imp in zip(feature_names, importances):
        bar = "█" * int(imp * 40)
        print(f"{name:<24} | {imp * 100:>5.1f}% | {bar}")

    print(f"\nSum of normalized importances: {np.sum(importances):.4f} (Must equal 1.0)")

    # Assertions:
    # Feature 0 > Feature 1 > Feature 2 > Feature 3
    assert np.isclose(np.sum(importances), 1.0), "Normalized importances must sum to 1.0!"
    assert importances[0] > importances[1], "Feature 0 should be more important than Feature 1"
    assert importances[1] > importances[2], "Feature 1 should be more important than Feature 2"
    assert importances[2] > importances[3], "Feature 2 should be more important than pure noise Feature 3"
    assert importances[3] < 0.10, "Noise feature importance should be very low (<10%)"

    print("\nFeature importance calculations successfully verified!")

