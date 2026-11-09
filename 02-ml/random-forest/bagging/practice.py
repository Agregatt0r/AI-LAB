# Practice: Bagging Classifier from Scratch & Variance Reduction
# Demonstrates:
#   1. Training an ensemble of decision trees on bootstrap samples
#   2. Variance reduction: Bagging stabilizes test accuracy across different training subsamples
#   3. Test accuracy improvement over a single decision tree

import numpy as np


class SimpleDecisionTree:
    """Minimal decision tree classifier for ensemble experiments."""
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        return 1.0 - np.sum((counts / len(y)) ** 2)

    def _find_split(self, X, y):
        best_gain = -1.0
        best_feat, best_thresh = None, None
        parent_gini = self._gini(y)
        n = len(y)

        for j in range(X.shape[1]):
            vals = np.sort(np.unique(X[:, j]))
            if len(vals) <= 1:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2.0

            for t in thresholds:
                left_mask = X[:, j] <= t
                n_l = np.sum(left_mask)
                n_r = n - n_l
                if n_l == 0 or n_r == 0:
                    continue

                g_split = (n_l / n) * self._gini(y[left_mask]) + (n_r / n) * self._gini(y[~left_mask])
                gain = parent_gini - g_split
                if gain > best_gain:
                    best_gain = gain
                    best_feat = j
                    best_thresh = t

        return best_feat, best_thresh

    def _build(self, X, y, depth=0):
        if len(np.unique(y)) <= 1 or (self.max_depth and depth >= self.max_depth) or len(y) < self.min_samples_split:
            classes, counts = np.unique(y, return_counts=True)
            return {"leaf": True, "value": classes[np.argmax(counts)]}

        feat, thresh = self._find_split(X, y)
        if feat is None:
            classes, counts = np.unique(y, return_counts=True)
            return {"leaf": True, "value": classes[np.argmax(counts)]}

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


class BaggingClassifier:
    """Bagging ensemble of Decision Trees."""
    def __init__(self, n_estimators=20, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators_ = []

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        self.estimators_ = []

        for _ in range(self.n_estimators):
            # Bootstrap sample (draw N with replacement)
            boot_idx = rng.choice(n, size=n, replace=True)
            tree = SimpleDecisionTree(max_depth=self.max_depth)
            tree.fit(X[boot_idx], y[boot_idx])
            self.estimators_.append(tree)
        return self

    def predict(self, X):
        # Gather predictions from all trees: shape (n_estimators, n_samples)
        all_preds = np.array([tree.predict(X) for tree in self.estimators_])
        # Majority vote across estimators (axis 0)
        final_preds = []
        for j in range(len(X)):
            classes, counts = np.unique(all_preds[:, j], return_counts=True)
            final_preds.append(classes[np.argmax(counts)])
        return np.array(final_preds)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


if __name__ == "__main__":
    # Generate Synthetic Dataset with label noise
    rng = np.random.default_rng(42)
    n = 300
    X = rng.uniform(-2.0, 2.0, size=(n, 2))
    # True non-linear boundary: circle
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2) < 1.0).astype(int)
    # Inject 15% noise
    noise = rng.choice(n, size=int(0.15 * n), replace=False)
    y[noise] = 1 - y[noise]

    # Split Train / Test
    X_train, X_test = X[:200], X[200:]
    y_train, y_test = y[:200], y[200:]

    # 1. Single Decision Tree vs Bagging on Fixed Train Set
    single_tree = SimpleDecisionTree(max_depth=None)
    single_tree.fit(X_train, y_train)
    single_train_acc = np.mean(single_tree.predict(X_train) == y_train)
    single_test_acc = np.mean(single_tree.predict(X_test) == y_test)

    bagging_model = BaggingClassifier(n_estimators=25, max_depth=None, random_state=42)
    bagging_model.fit(X_train, y_train)
    bag_train_acc = bagging_model.score(X_train, y_train)
    bag_test_acc = bagging_model.score(X_test, y_test)

    print("--- 1. Single Decision Tree vs Bagging Ensemble ---")
    print(f"Single Tree -> Train Acc: {single_train_acc * 100:.2f}% | Test Acc: {single_test_acc * 100:.2f}%")
    print(f"Bagging (25) -> Train Acc: {bag_train_acc * 100:.2f}% | Test Acc: {bag_test_acc * 100:.2f}%")
    print(f"Test Accuracy Improvement: +{(bag_test_acc - single_test_acc) * 100:.2f}%")

    assert bag_test_acc > single_test_acc, "Bagging should improve generalization test accuracy!"

    # 2. Stability Analysis Across Random Data Subsamples (Variance Reduction)
    print("\n--- 2. Performance Stability (Variance Reduction Across 10 Trials) ---")
    single_accs = []
    bagging_accs = []

    for trial in range(10):
        sub_idx = rng.choice(len(X_train), size=int(0.8 * len(X_train)), replace=True)
        st = SimpleDecisionTree(max_depth=None).fit(X_train[sub_idx], y_train[sub_idx])
        bg = BaggingClassifier(n_estimators=15, max_depth=None, random_state=trial).fit(X_train[sub_idx], y_train[sub_idx])

        single_accs.append(np.mean(st.predict(X_test) == y_test))
        bagging_accs.append(bg.score(X_test, y_test))

    std_single = np.std(single_accs)
    std_bagging = np.std(bagging_accs)
    mean_single = np.mean(single_accs)
    mean_bagging = np.mean(bagging_accs)

    print(f"Single Tree: Mean Acc = {mean_single * 100:.2f}% | Std = {std_single * 100:.2f}%")
    print(f"Bagging:     Mean Acc = {mean_bagging * 100:.2f}% | Std = {std_bagging * 100:.2f}%")
    print(f"Variance / Std Reduction: {std_single / std_bagging:.1f}x more stable predictions!")

    assert mean_bagging > mean_single, "Bagging average accuracy must exceed single tree"
    assert std_bagging < std_single, "Bagging standard deviation must be lower than single tree"

    print("\nBagging mechanism and variance reduction verified successfully!")
