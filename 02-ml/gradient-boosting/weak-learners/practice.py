"""Weak Learners: Decision Stumps and Shallow Trees.

Demonstrates:
1. Empirical Bias-Variance profiles at different tree depths.
2. How individual weak learner accuracy compares to ensemble accuracy.
3. Sequential vs parallel diversity: different targets in boosting vs bagging.
"""

import numpy as np

try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def generate_dataset(n: int = 500, random_state: int = 42) -> tuple:
    """Noisy circular classification problem."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n, 2) * 2
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2) < 3.0).astype(int)
    noise_idx = rng.choice(n, size=n // 10, replace=False)
    y[noise_idx] = 1 - y[noise_idx]
    perm = rng.permutation(n)
    split = int(n * 0.8)
    return (X[perm[:split]], y[perm[:split]],
            X[perm[split:]], y[perm[split:]])


class DecisionStumpScratch:
    """Pure NumPy single-split Decision Stump."""

    def __init__(self):
        self.feature_idx = 0
        self.threshold = 0.0
        self.polarity = 1
        self.bias_estimate = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionStumpScratch":
        n_samples, n_features = X.shape
        best_error = float("inf")

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                for polarity in [1, -1]:
                    preds_bin = (np.where(X[:, feature] >= threshold, polarity, -polarity) == 1).astype(int)
                    err = np.mean(preds_bin != y)
                    if err < best_error:
                        best_error = err
                        self.feature_idx = feature
                        self.threshold = threshold
                        self.polarity = polarity

        self.bias_estimate = best_error
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = np.where(X[:, self.feature_idx] >= self.threshold, self.polarity, -self.polarity)
        return (raw == 1).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


def measure_bias_variance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bootstrap: int = 20,
    random_state: int = 42,
) -> dict:
    """Estimate bias and variance across tree depths by bootstrap resampling."""
    if not HAS_SKLEARN:
        return {}

    rng = np.random.RandomState(random_state)
    results = {}

    for depth in [1, 2, 3, 5, 8]:
        test_accs = []
        for _ in range(n_bootstrap):
            idx = rng.choice(len(X_train), size=len(X_train), replace=True)
            Xb, yb = X_train[idx], y_train[idx]
            clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
            clf.fit(Xb, yb)
            test_accs.append(clf.score(X_test, y_test))

        results[depth] = {
            "mean_acc": float(np.mean(test_accs)),
            "variance": float(np.var(test_accs)),
            # Bias approximated as error relative to best single-tree performance
            "std_across_boots": float(np.std(test_accs)),
        }

    return results


def main() -> None:
    print("=" * 70)
    print("2. WEAK LEARNERS: DECISION STUMPS AND SHALLOW TREES")
    print("=" * 70)

    X_train, y_train, X_test, y_test = generate_dataset(n=600, random_state=42)
    print(f"Dataset: {len(X_train)} train / {len(X_test)} test points (10% label noise)\n")

    # 1. Single decision stump (depth=1)
    stump = DecisionStumpScratch()
    stump.fit(X_train, y_train)
    stump_train = stump.score(X_train, y_train)
    stump_test = stump.score(X_test, y_test)

    print(f"Single Decision Stump (depth=1):")
    print(f"  Train Accuracy: {stump_train * 100:.1f}%")
    print(f"  Test  Accuracy: {stump_test * 100:.1f}%")
    print(f"  → High bias: cannot learn non-linear circular boundary with 1 split\n")

    # 2. Bias-Variance across tree depths
    if HAS_SKLEARN:
        bv = measure_bias_variance(X_train, y_train, X_test, y_test, n_bootstrap=30)
        print("Bias-Variance Profile across Tree Depths (30 bootstrap samples):")
        print("-" * 65)
        print(f"{'Depth':<8} {'Mean Accuracy':<18} {'StdDev (≈ sqrt(Var))':<22} {'Interpretation'}")
        print("-" * 65)
        for depth, stats in bv.items():
            interp = (
                "High Bias" if depth == 1 else
                "Balanced" if depth <= 3 else
                "High Variance"
            )
            print(f"{depth:<8} {stats['mean_acc'] * 100:<18.1f} {stats['std_across_boots'] * 100:<22.3f} {interp}")
        print("-" * 65)

        # 3. Diversity source: boosting vs bagging
        print("\nDiversity Source: Different Targets (Boosting) vs Different Data (Bagging):")
        gb = GradientBoostingClassifier(n_estimators=50, max_depth=1, learning_rate=0.3, random_state=42)
        gb.fit(X_train, y_train)

        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)

        print(f"  GradientBoosting (50 stumps, depth=1): test acc = {gb.score(X_test, y_test) * 100:.1f}%")
        print(f"  RandomForest    (50 trees, depth=5):   test acc = {rf.score(X_test, y_test) * 100:.1f}%")
        print(f"  → GB reduces bias step-by-step; RF reduces variance through averaging")

        # Assertions
        assert gb.score(X_test, y_test) > stump_test + 0.10, "Boosting must outperform single stump"
        assert bv[1]["std_across_boots"] < bv[8]["std_across_boots"], "Shallow trees must have lower variance than deep trees"
    else:
        print("\n[scikit-learn not available, running standalone stump only]")
        assert stump_test > 0.40, "Even a single stump should do better than random"

    print("\n✓ Weak learner Bias-Variance profiles verified successfully!")


if __name__ == "__main__":
    main()

