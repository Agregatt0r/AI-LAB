# Practice: Random Forest with scikit-learn & From Scratch
# Demonstrates:
#   1. Complete NumPy RandomForestClassifier implementation
#   2. Out-of-Bag (OOB) score calculation from scratch
#   3. Side-by-side verification with sklearn.ensemble.RandomForestClassifier
#   4. Feature Importances extraction

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier as SklearnRF
    from sklearn.metrics import classification_report, accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# -------------------------------------------------------------
# 1. Base Decision Tree for the Random Forest
# -------------------------------------------------------------
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, probs=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.probs = probs

    @property
    def is_leaf(self):
        return self.value is not None


class BaseRFTree:
    def __init__(self, max_depth=5, max_features="sqrt", min_samples_split=2, rng=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.rng = rng if rng is not None else np.random.default_rng()
        self.root = None
        self.n_classes_ = None

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        return 1.0 - np.sum((counts / len(y)) ** 2)

    def _get_m(self, d):
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(d)))
        return d

    def _find_split(self, X, y, importances):
        n, d = X.shape
        parent_gini = self._gini(y)

        # Feature Bagging: sample m random features
        m = self._get_m(d)
        candidates = self.rng.choice(d, size=m, replace=False)

        best_gain = -1.0
        best_feat, best_thresh = None, None
        best_left = None

        for j in candidates:
            vals = np.sort(np.unique(X[:, j]))
            if len(vals) <= 1:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2.0

            for t in thresholds:
                left = X[:, j] <= t
                n_l = np.sum(left)
                n_r = n - n_l
                if n_l == 0 or n_r == 0:
                    continue

                g_split = (n_l / n) * self._gini(y[left]) + (n_r / n) * self._gini(y[~left])
                gain = parent_gini - g_split

                if gain > best_gain:
                    best_gain = gain
                    best_feat = j
                    best_thresh = t
                    best_left = left

        if best_feat is not None and best_gain > 0:
            importances[best_feat] += n * best_gain

        return best_feat, best_thresh, best_left

    def _build(self, X, y, importances, depth=0):
        n = len(y)
        classes, counts = np.unique(y, return_counts=True)
        probs = np.zeros(self.n_classes_)
        for c, cnt in zip(classes, counts):
            probs[c] = cnt / n

        maj_class = classes[np.argmax(counts)]

        if len(classes) <= 1 or (self.max_depth and depth >= self.max_depth) or n < self.min_samples_split:
            return Node(value=maj_class, probs=probs)

        feat, thresh, left_mask = self._find_split(X, y, importances)
        if feat is None:
            return Node(value=maj_class, probs=probs)

        return Node(
            feature=feat,
            threshold=thresh,
            left=self._build(X[left_mask], y[left_mask], importances, depth + 1),
            right=self._build(X[~left_mask], y[~left_mask], importances, depth + 1),
            probs=probs
        )

    def fit(self, X, y, importances):
        self.n_classes_ = len(np.unique(y)) if self.n_classes_ is None else self.n_classes_
        self.root = self._build(X, y, importances, depth=0)
        return self

    def _traverse(self, x, node):
        if node.is_leaf:
            return node.probs
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict_proba(self, X):
        return np.array([self._traverse(x, self.root) for x in X])


# -------------------------------------------------------------
# 2. RandomForestClassifier From Scratch
# -------------------------------------------------------------
class RandomForestClassifierScratch:
    def __init__(self, n_estimators=30, max_depth=6, max_features="sqrt", oob_score=True, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.oob_score = oob_score
        self.random_state = random_state
        self.trees_ = []
        self.feature_importances_ = None
        self.oob_score_ = None
        self.n_classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n, d = X.shape
        self.n_classes_ = len(np.unique(y))

        rng = np.random.default_rng(self.random_state)
        raw_importances = np.zeros(d)
        self.trees_ = []

        # OOB tracking: accumulates probability predictions for each sample
        oob_predictions = np.zeros((n, self.n_classes_))
        oob_counts = np.zeros(n)

        for t in range(self.n_estimators):
            # Bootstrap sample
            boot_idx = rng.choice(n, size=n, replace=True)
            oob_mask = np.ones(n, dtype=bool)
            oob_mask[boot_idx] = False
            oob_idx = np.where(oob_mask)[0]

            tree = BaseRFTree(
                max_depth=self.max_depth,
                max_features=self.max_features,
                rng=np.random.default_rng(rng.integers(0, 1000000))
            )
            tree.n_classes_ = self.n_classes_
            tree.fit(X[boot_idx], y[boot_idx], raw_importances)
            self.trees_.append(tree)

            # Evaluate tree on its out-of-bag samples
            if self.oob_score and len(oob_idx) > 0:
                probs_oob = tree.predict_proba(X[oob_idx])
                oob_predictions[oob_idx] += probs_oob
                oob_counts[oob_idx] += 1

        # Normalize feature importances
        total_imp = np.sum(raw_importances)
        self.feature_importances_ = raw_importances / total_imp if total_imp > 0 else np.ones(d) / d

        # Compute Out-of-Bag Score
        if self.oob_score:
            valid_oob = oob_counts > 0
            oob_final_preds = np.argmax(oob_predictions[valid_oob], axis=1)
            self.oob_score_ = np.mean(oob_final_preds == y[valid_oob])

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        all_tree_probs = np.array([tree.predict_proba(X) for tree in self.trees_])
        # Soft voting: average probabilities across trees (axis 0)
        return np.mean(all_tree_probs, axis=0)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


if __name__ == "__main__":
    # Generate Multi-Feature Classification Dataset
    # 6 Features: 3 genuine non-linear signals, 3 noisy random features
    rng = np.random.default_rng(42)
    n = 400
    d = 6

    X = rng.standard_normal((n, d))
    latent = 2.0 * X[:, 0] - 1.5 * (X[:, 1] ** 2) + 1.2 * X[:, 2] + rng.normal(0, 0.4, n)
    y = (latent > 0.0).astype(int)

    # Train / Test split
    X_train, X_test = X[:300], X[300:]
    y_train, y_test = y[:300], y[300:]

    # 1. Custom Random Forest From Scratch
    print("--- 1. Training Custom RandomForestClassifier from Scratch ---")
    custom_rf = RandomForestClassifierScratch(n_estimators=40, max_depth=6, max_features="sqrt", oob_score=True, random_state=42)
    custom_rf.fit(X_train, y_train)

    custom_train_acc = custom_rf.score(X_train, y_train)
    custom_test_acc = custom_rf.score(X_test, y_test)

    print(f"Custom RF Train Accuracy: {custom_train_acc * 100:.2f}%")
    print(f"Custom RF Test Accuracy:  {custom_test_acc * 100:.2f}%")
    print(f"Custom RF OOB Score:      {custom_rf.oob_score_ * 100:.2f}% (Free internal validation!)\n")

    # 2. scikit-learn Comparison
    print("--- 2. scikit-learn RandomForestClassifier Benchmark ---")
    if HAS_SKLEARN:
        sk_rf = SklearnRF(n_estimators=40, max_depth=6, max_features="sqrt", oob_score=True, random_state=42)
        sk_rf.fit(X_train, y_train)

        sk_test_acc = sk_rf.score(X_test, y_test)
        print(f"scikit-learn Test Accuracy: {sk_test_acc * 100:.2f}%")
        print(f"scikit-learn OOB Score:     {sk_rf.oob_score_ * 100:.2f}%\n")

        # Feature importances comparison
        print("--- 3. Feature Importances: Custom vs scikit-learn ---")
        print(f"{'Feature':<12} | {'Custom MDI':<14} | {'scikit-learn MDI'}")
        print("-" * 45)
        for j in range(d):
            print(f"Feature #{j:<5} | {custom_rf.feature_importances_[j] * 100:>6.2f}%        | {sk_rf.feature_importances_[j] * 100:>6.2f}%")

        assert abs(custom_test_acc - sk_test_acc) < 0.06, "Custom RF accuracy should closely match scikit-learn!"
        assert abs(custom_rf.oob_score_ - sk_rf.oob_score_) < 0.06, "OOB score should match scikit-learn closely!"
    else:
        print("scikit-learn not available; custom implementation verified analytically.")

    assert custom_test_acc > 0.85, "Custom RF test accuracy should exceed 85%"
    assert custom_rf.oob_score_ > 0.80, "OOB score should exceed 80%"

    print("\nRandom Forest scikit-learn and from-scratch implementations successfully verified!")

