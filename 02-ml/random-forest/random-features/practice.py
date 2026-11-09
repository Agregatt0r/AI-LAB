# Practice: Random Feature Selection (Feature Bagging)
# Demonstrates:
#   1. Split finding with random feature subset selection (m = sqrt(d))
#   2. Comparison of root feature diversity: Pure Bagging vs Random Forest
#   3. Accuracy comparison on a multi-feature dataset with a dominant feature

import numpy as np


class RandomForestTree:
    """Decision tree that selects a random subset of m features at each node split."""
    def __init__(self, max_depth=5, max_features="sqrt", random_state=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)
        self.tree_ = None

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        return 1.0 - np.sum((counts / len(y)) ** 2)

    def _get_m_features(self, total_features):
        """Determine number of candidate features to sample."""
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(total_features)))
        elif self.max_features is None or self.max_features == "all":
            return total_features
        elif isinstance(self.max_features, int):
            return min(total_features, self.max_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * total_features))
        return total_features

    def _find_split(self, X, y):
        n_samples, n_features = X.shape
        parent_gini = self._gini(y)

        # 1. Random Feature Selection: Pick m random features WITHOUT replacement
        m = self._get_m_features(n_features)
        feature_candidates = self.rng.choice(n_features, size=m, replace=False)

        best_gain = -1.0
        best_feat, best_thresh = None, None

        for j in feature_candidates:
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

        return best_feat, best_thresh

    def _build(self, X, y, depth=0):
        if len(np.unique(y)) <= 1 or (self.max_depth and depth >= self.max_depth):
            classes, counts = np.unique(y, return_counts=True)
            return {"leaf": True, "value": classes[np.argmax(counts)], "feat": None}

        feat, thresh = self._find_split(X, y)
        if feat is None:
            classes, counts = np.unique(y, return_counts=True)
            return {"leaf": True, "value": classes[np.argmax(counts)], "feat": None}

        mask = X[:, feat] <= thresh
        return {
            "leaf": False,
            "feat": feat,
            "thresh": thresh,
            "left": self._build(X[mask], y[mask], depth + 1),
            "right": self._build(X[~mask], y[~mask], depth + 1),
        }

    def fit(self, X, y):
        self.tree_ = self._build(X, y)
        return self

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["value"]
        if x[node["feat"]] <= node["thresh"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree_) for x in X])


if __name__ == "__main__":
    # Generate Synthetic Dataset with 9 features:
    # Feature 0: Dominant strong signal (correlates with y with 75% accuracy alone)
    # Features 1..8: Moderately informative secondary signals
    rng = np.random.default_rng(42)
    n = 250
    d = 9

    X = rng.standard_normal((n, d))
    # Target depends on linear combination of features 0, 1, 2, 3
    latent = 2.5 * X[:, 0] + 1.2 * X[:, 1] + 1.0 * X[:, 2] + 0.8 * X[:, 3]
    y = (latent > 0.0).astype(int)

    # 1. Compare Root Node Splits: Pure Bagging vs Random Forest
    n_trees = 30
    bagging_root_features = []
    rf_root_features = []

    for t in range(n_trees):
        # Bootstrap sample
        boot_idx = rng.choice(n, size=n, replace=True)
        X_b, y_b = X[boot_idx], y[boot_idx]

        # Pure Bagging: evaluates ALL 9 features at every split
        tree_bag = RandomForestTree(max_depth=4, max_features="all", random_state=t)
        tree_bag.fit(X_b, y_b)
        bagging_root_features.append(tree_bag.tree_["feat"])

        # Random Forest: evaluates m = sqrt(9) = 3 features at every split
        tree_rf = RandomForestTree(max_depth=4, max_features="sqrt", random_state=t)
        tree_rf.fit(X_b, y_b)
        rf_root_features.append(tree_rf.tree_["feat"])

    print("--- 1. Root Split Feature Diversity Across 30 Trees ---")
    bag_unique, bag_counts = np.unique(bagging_root_features, return_counts=True)
    rf_unique, rf_counts = np.unique(rf_root_features, return_counts=True)

    print("Pure Bagging (max_features = 'all'):")
    for feat, cnt in zip(bag_unique, bag_counts):
        print(f"  Feature #{feat}: chosen as root {cnt:>2d} / 30 times ({cnt/30*100:.0f}%)")

    print("\nRandom Forest (max_features = 'sqrt' = 3):")
    for feat, cnt in zip(rf_unique, rf_counts):
        print(f"  Feature #{feat}: chosen as root {cnt:>2d} / 30 times ({cnt/30*100:.0f}%)")

    # Pure bagging should overwhelmingly pick Feature 0 at root (>85% of trees)
    assert bag_counts[np.where(bag_unique == 0)[0][0]] >= 25, "Pure Bagging should pick dominant Feature 0 almost exclusively"
    # Random Forest should have more diverse root features (at least 3 distinct root features)
    assert len(rf_unique) >= 3, "Random Forest must produce more diverse root features"

    print("\nRandom feature selection successfully decorrelated decision trees!")

