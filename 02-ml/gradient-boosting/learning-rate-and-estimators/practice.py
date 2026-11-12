"""
Learning Rate and Number of Estimators in Gradient Boosting
============================================================
Demonstrates the ν–M trade-off using a pure NumPy GBM implementation.
Also validates against sklearn.GradientBoostingRegressor when available.

Run with system python3 (NumPy only) or .venv/bin/python3 (NumPy + sklearn).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Optional sklearn imports
# ---------------------------------------------------------------------------
try:
    from sklearn.ensemble import GradientBoostingRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Pure NumPy GBM implementation
# ---------------------------------------------------------------------------

class RegressionStump:
    """Depth-1 regression tree (decision stump)."""

    def fit(self, X, y):
        best_loss = np.inf
        self.feature_ = 0
        self.threshold_ = 0.0
        self.left_val_ = 0.0
        self.right_val_ = 0.0

        for f in range(X.shape[1]):
            thresholds = np.unique(X[:, f])
            for t in thresholds:
                left = y[X[:, f] <= t]
                right = y[X[:, f] > t]
                if len(left) == 0 or len(right) == 0:
                    continue
                loss = len(left) * left.var() + len(right) * right.var()
                if loss < best_loss:
                    best_loss = loss
                    self.feature_ = f
                    self.threshold_ = t
                    self.left_val_ = left.mean()
                    self.right_val_ = right.mean()
        return self

    def predict(self, X):
        return np.where(
            X[:, self.feature_] <= self.threshold_,
            self.left_val_,
            self.right_val_,
        )


class GBMRegressorScratch:
    """Gradient Boosting Regressor built from scratch using NumPy."""

    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def fit(self, X, y):
        self.F0_ = y.mean()
        F = np.full(len(y), self.F0_)
        self.trees_ = []
        self.train_mse_ = []
        for _ in range(self.n_estimators):
            r = y - F
            t = RegressionStump().fit(X, r)
            F += self.learning_rate * t.predict(X)
            self.trees_.append(t)
            self.train_mse_.append(np.mean((y - F) ** 2))
        return self

    def predict(self, X):
        F = np.full(len(X), self.F0_)
        for t in self.trees_:
            F += self.learning_rate * t.predict(X)
        return F


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("4. LEARNING RATE AND NUMBER OF ESTIMATORS")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    rng = np.random.RandomState(42)
    X = rng.randn(300, 2)
    y = np.sin(X[:, 0]) + X[:, 1] ** 2 + 0.3 * rng.randn(300)

    X_train, X_test = X[:200], X[200:]
    y_train, y_test = y[:200], y[200:]

    # -----------------------------------------------------------------------
    # Sweep 1: Learning Rate (n_estimators fixed = 200)
    # -----------------------------------------------------------------------
    print()
    print("--- Learning Rate Sweep (n_estimators=200, fixed) ---")

    learning_rates = [0.01, 0.05, 0.1, 0.3, 1.0]
    results_lr = {}

    for lr in learning_rates:
        gbm = GBMRegressorScratch(n_estimators=200, learning_rate=lr)
        gbm.fit(X_train, y_train)
        train_mse = np.mean((y_train - gbm.predict(X_train)) ** 2)
        test_mse = np.mean((y_test - gbm.predict(X_test)) ** 2)
        results_lr[lr] = {"train_mse": train_mse, "test_mse": test_mse}
        print(f"lr={lr:<5}  train_mse={train_mse:.4f}  test_mse={test_mse:.4f}")

    # -----------------------------------------------------------------------
    # Sweep 2: N Estimators (learning_rate fixed = 0.1)
    # -----------------------------------------------------------------------
    print()
    print("--- N Estimators Sweep (learning_rate=0.10, fixed) ---")

    n_estimators_list = [5, 10, 20, 50, 100, 200, 500]
    results_n = {}

    for n in n_estimators_list:
        gbm = GBMRegressorScratch(n_estimators=n, learning_rate=0.1)
        gbm.fit(X_train, y_train)
        test_mse = np.mean((y_test - gbm.predict(X_test)) ** 2)
        results_n[n] = {"test_mse": test_mse}
        print(f"n={n:<4}   test_mse={test_mse:.4f}")

    # -----------------------------------------------------------------------
    # Optional: sklearn validation
    # -----------------------------------------------------------------------
    if SKLEARN_AVAILABLE:
        print()
        print("--- sklearn GradientBoostingRegressor Validation ---")
        sk_model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=1, random_state=42
        )
        sk_model.fit(X_train, y_train)
        sk_test_mse = np.mean((y_test - sk_model.predict(X_test)) ** 2)
        scratch_test_mse = results_lr[0.1]["test_mse"]
        print(f"scratch GBM test_mse : {scratch_test_mse:.4f}")
        print(f"sklearn  GBM test_mse: {sk_test_mse:.4f}")
        print(
            f"Difference           : {abs(scratch_test_mse - sk_test_mse):.4f}"
            f"  (small difference expected due to split criteria)"
        )
    else:
        print()
        print("(sklearn not available — skipping validation)")

    # -----------------------------------------------------------------------
    # Assertions
    # -----------------------------------------------------------------------
    # Very high lr (1.0) should overfit → higher test MSE than lr=0.1
    assert results_lr[1.0]["test_mse"] > results_lr[0.1]["test_mse"], (
        f"Expected lr=1.0 test_mse ({results_lr[1.0]['test_mse']:.4f}) > "
        f"lr=0.1 test_mse ({results_lr[0.1]['test_mse']:.4f})"
    )

    # Too few estimators (n=5) should underfit → higher test MSE than n=200
    assert results_n[5]["test_mse"] > results_n[200]["test_mse"], (
        f"Expected n=5 test_mse ({results_n[5]['test_mse']:.4f}) > "
        f"n=200 test_mse ({results_n[200]['test_mse']:.4f})"
    )

    print()
    print("✓ Learning rate and estimator sweep verified!")


if __name__ == "__main__":
    main()
