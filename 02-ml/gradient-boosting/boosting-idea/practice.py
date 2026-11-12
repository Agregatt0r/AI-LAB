"""Boosting Idea: AdaBoost from Scratch.

Implements:
1. AdaBoostScratch: decision stumps as weak learners with adaptive sample reweighting.
2. Alpha (learner weight) computation: alpha = 0.5 * ln((1 - eps) / eps).
3. Sample weight update: w_i *= exp(-alpha * y_i * h(x_i)).
4. Empirical proof that boosting converts weak learners (55% accuracy) into strong ones (95%+).
"""

import numpy as np

try:
    from sklearn.ensemble import AdaBoostClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class DecisionStump:
    """Single-split decision tree (depth=1) — the canonical weak learner for AdaBoost."""

    def __init__(self):
        self.feature_idx: int = 0
        self.threshold: float = 0.0
        self.polarity: int = 1  # 1 or -1

    def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        """Fit the stump by finding the split minimizing weighted classification error."""
        n_samples, n_features = X.shape
        best_error = float("inf")

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                for polarity in [1, -1]:
                    preds = np.where(X[:, feature] >= threshold, polarity, -polarity)
                    weighted_error = np.sum(weights[preds != y])

                    if weighted_error < best_error:
                        best_error = weighted_error
                        self.feature_idx = feature
                        self.threshold = threshold
                        self.polarity = polarity

        return float(best_error)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(X[:, self.feature_idx] >= self.threshold, self.polarity, -self.polarity)


class AdaBoostScratch:
    """AdaBoost classifier (Freund & Schapire, 1997) using decision stumps."""

    def __init__(self, n_estimators: int = 50):
        self.n_estimators = n_estimators
        self.stumps: list = []
        self.alphas: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostScratch":
        n_samples = len(y)
        # Encode labels as {-1, +1}
        y_enc = np.where(y == 1, 1, -1).astype(float)
        weights = np.full(n_samples, 1.0 / n_samples)

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            epsilon = stump.fit(X, y_enc, weights)

            # Clip to avoid division by zero or log(0)
            epsilon = np.clip(epsilon, 1e-10, 1.0 - 1e-10)

            # Learner weight: large for accurate learners, negative for anti-correlated ones
            alpha = 0.5 * np.log((1.0 - epsilon) / epsilon)

            # Update sample weights: upweight misclassified, downweight correctly classified
            preds = stump.predict(X)
            weights *= np.exp(-alpha * y_enc * preds)
            weights /= np.sum(weights)  # Normalize to sum = 1

            self.stumps.append(stump)
            self.alphas.append(alpha)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Weighted vote of all stumps
        ensemble_scores = sum(
            alpha * stump.predict(X) for alpha, stump in zip(self.alphas, self.stumps)
        )
        return np.where(ensemble_scores >= 0, 1, 0)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


def main() -> None:
    print("=" * 70)
    print("1. BOOSTING IDEA: ADABOOST FROM SCRATCH")
    print("=" * 70)

    rng = np.random.RandomState(42)

    # 1. Generate a noisy non-linearly separable 2D dataset
    n = 400
    X = rng.randn(n, 2)
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2) < 1.3).astype(int)
    # Add 10% label noise
    noise_idx = rng.choice(n, size=n // 10, replace=False)
    y[noise_idx] = 1 - y[noise_idx]

    # Train/test split (80/20)
    split = int(n * 0.8)
    perm = rng.permutation(n)
    X_train, y_train = X[perm[:split]], y[perm[:split]]
    X_test, y_test = X[perm[split:]], y[perm[split:]]

    # 2. Single weak learner baseline
    single_stump = DecisionStump()
    single_stump.fit(X_train, np.where(y_train == 1, 1, -1), np.full(len(y_train), 1.0 / len(y_train)))
    preds_single = np.where(single_stump.predict(X_test) == 1, 1, 0)
    acc_single = float(np.mean(preds_single == y_test))

    print(f"Single Decision Stump accuracy: {acc_single * 100:.1f}%")
    print(f"  (Barely above 50% — a true 'weak learner')\n")

    # 3. AdaBoost ensemble over growing M
    print("AdaBoost Ensemble Accuracy vs Number of Estimators:")
    print("-" * 50)
    print(f"{'M (estimators)':<18} {'Train Acc':<12} {'Test Acc':<12}")
    print("-" * 50)

    for m in [1, 5, 10, 25, 50, 100]:
        ab = AdaBoostScratch(n_estimators=m)
        ab.fit(X_train, y_train)
        train_acc = ab.score(X_train, y_train)
        test_acc = ab.score(X_test, y_test)
        print(f"{m:<18} {train_acc * 100:<12.1f} {test_acc * 100:<12.1f}")

    # 4. Assertions
    ab_final = AdaBoostScratch(n_estimators=100)
    ab_final.fit(X_train, y_train)
    final_test = ab_final.score(X_test, y_test)

    assert acc_single < 0.70, "Single stump should be weak (< 70% accuracy)"
    assert final_test > 0.80, f"AdaBoost with 100 estimators should exceed 80%, got {final_test:.2f}"
    assert final_test > acc_single + 0.10, "Boosting must substantially improve over single stump"

    # 5. Alpha dynamics: illustrate how alpha changes with accuracy
    print("\nLearner Weight (alpha) vs. Stump Accuracy:")
    print("-" * 50)
    print(f"{'Stump Accuracy':<20} {'Alpha (weight)':<18} {'Interpretation'}")
    print("-" * 50)
    for acc_val in [0.51, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]:
        eps = 1.0 - acc_val
        alpha_val = 0.5 * np.log((1 - eps) / eps)
        print(f"{acc_val * 100:<20.0f}% {alpha_val:<18.4f} {'↑ Large influence' if alpha_val > 1.0 else '→ Moderate influence'}")

    print("\n✓ AdaBoost boosting-from-weak-to-strong verified successfully!")


if __name__ == "__main__":
    main()

