"""Residuals and Gradients: Gradient Boosting for Regression from Scratch.

Implements the full Friedman GBM algorithm using MSE loss and decision tree stumps:
1. Initialization: F_0 = mean(y).
2. Pseudo-residuals: r_i = y_i - F_{m-1}(x_i).
3. Tree fitting to residuals.
4. Sequential ensemble update: F_m = F_{m-1} + nu * h_m.
5. Comparison of pseudo-residuals for MSE vs Log-Loss vs MAE.
"""

import numpy as np

try:
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def pseudo_residuals_mse(y: np.ndarray, F: np.ndarray) -> np.ndarray:
    """MSE: r_i = y_i - F(x_i)  [ordinary residuals]"""
    return y - F


def pseudo_residuals_logloss(y: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Log-loss: r_i = y_i - sigmoid(F(x_i))  [probability residuals]"""
    return y - sigmoid(F)


def pseudo_residuals_mae(y: np.ndarray, F: np.ndarray) -> np.ndarray:
    """MAE: r_i = sign(y_i - F(x_i))  [robust, outlier-insensitive]"""
    return np.sign(y - F)


class RegressionStump:
    """Depth-1 regression tree in pure NumPy."""

    def __init__(self):
        self.feature_idx = 0
        self.threshold = 0.0
        self.left_val = 0.0
        self.right_val = 0.0

    def fit(self, X: np.ndarray, r: np.ndarray) -> "RegressionStump":
        n_samples, n_features = X.shape
        best_mse = float("inf")

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] < threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                left_mean = r[left_mask].mean()
                right_mean = r[right_mask].mean()
                mse = (np.sum((r[left_mask] - left_mean) ** 2) +
                       np.sum((r[right_mask] - right_mean) ** 2))

                if mse < best_mse:
                    best_mse = mse
                    self.feature_idx = feature
                    self.threshold = threshold
                    self.left_val = left_mean
                    self.right_val = right_mean

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.where(X[:, self.feature_idx] < self.threshold, self.left_val, self.right_val)
        return preds


class GBMRegressorScratch:
    """Gradient Boosting Regressor (MSE loss) from first principles."""

    def __init__(self, n_estimators: int = 50, learning_rate: float = 0.1, max_depth: int = 1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.F0_: float = 0.0
        self.trees_: list = []
        self.train_losses_: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBMRegressorScratch":
        # Step 1: Initialize with mean (optimal constant for MSE)
        self.F0_ = float(np.mean(y))
        F = np.full(len(y), self.F0_)

        for m in range(self.n_estimators):
            # Step 2: Pseudo-residuals = negative gradient of MSE
            r = pseudo_residuals_mse(y, F)

            # Step 3: Fit regression stump to residuals
            stump = RegressionStump()
            stump.fit(X, r)
            h_m = stump.predict(X)

            # Step 4: Update ensemble (learning rate = shrinkage)
            F += self.learning_rate * h_m
            self.trees_.append(stump)

            mse = float(np.mean((y - F) ** 2))
            self.train_losses_.append(mse)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        F = np.full(len(X), self.F0_)
        for stump in self.trees_:
            F += self.learning_rate * stump.predict(X)
        return F

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1.0 - ss_res / ss_tot)


def main() -> None:
    print("=" * 70)
    print("3. RESIDUALS AND GRADIENTS: GBM FROM SCRATCH (MSE LOSS)")
    print("=" * 70)

    rng = np.random.RandomState(42)
    n = 300
    X = rng.uniform(-3.0, 3.0, size=(n, 1))
    # Non-linear target: sin wave + quadratic + noise
    y = np.sin(X[:, 0]) + 0.5 * X[:, 0] ** 2 + rng.normal(0, 0.3, n)

    split = int(n * 0.8)
    perm = rng.permutation(n)
    X_train, y_train = X[perm[:split]], y[perm[:split]]
    X_test, y_test = X[perm[split:]], y[perm[split:]]

    # 1. Baseline: constant prediction F_0 = mean(y)
    F0 = np.mean(y_train)
    baseline_mse_train = float(np.mean((y_train - F0) ** 2))
    baseline_mse_test = float(np.mean((y_test - F0) ** 2))
    print(f"Baseline (F_0 = mean(y) = {F0:.3f}):")
    print(f"  Train MSE: {baseline_mse_train:.4f}")
    print(f"  Test  MSE: {baseline_mse_test:.4f}\n")

    # 2. GBM from scratch
    gbm = GBMRegressorScratch(n_estimators=100, learning_rate=0.1, max_depth=1)
    gbm.fit(X_train, y_train)

    train_r2 = gbm.score(X_train, y_train)
    test_r2 = gbm.score(X_test, y_test)
    final_train_mse = gbm.train_losses_[-1]
    final_test_mse = float(np.mean((y_test - gbm.predict(X_test)) ** 2))

    print(f"GBM Scratch (100 stumps, lr=0.1):")
    print(f"  Train MSE: {final_train_mse:.4f}  (↓ from {baseline_mse_train:.4f})")
    print(f"  Test  MSE: {final_test_mse:.4f}  (↓ from {baseline_mse_test:.4f})")
    print(f"  Train R²:  {train_r2:.4f}")
    print(f"  Test  R²:  {test_r2:.4f}\n")

    # 3. Stage-by-stage residuals demonstration
    print("Stage-by-Stage Pseudo-Residuals (first 5 training points):")
    print("-" * 65)
    print(f"{'Stage m':<10} {'y[0]':<8} {'F(x[0])':<12} {'residual r[0]':<16}")
    print("-" * 65)

    F_track = np.full(len(X_train), np.mean(y_train))
    pt = 0  # Track point index 0
    print(f"{'0 (init)':<10} {y_train[pt]:<8.4f} {F_track[pt]:<12.4f} {y_train[pt] - F_track[pt]:<16.4f}")
    for m_show in [1, 5, 10, 25, 50, 100]:
        stump_m = gbm.trees_[m_show - 1]
        F_track_snapshot = np.full(len(X_train), gbm.F0_)
        for t in range(m_show):
            F_track_snapshot += gbm.learning_rate * gbm.trees_[t].predict(X_train)
        r = y_train[pt] - F_track_snapshot[pt]
        print(f"{m_show:<10} {y_train[pt]:<8.4f} {F_track_snapshot[pt]:<12.4f} {r:<16.4f}")
    print("-" * 65)
    print("→ Residuals shrink to near-zero as the ensemble converges!\n")

    # 4. Compare pseudo-residuals for different loss functions
    F_demo = np.array([0.5, 1.0, 1.5])
    y_demo_reg = np.array([2.0, 2.0, 2.0])
    y_demo_cls = np.array([1.0, 1.0, 1.0])

    print("Pseudo-Residuals for Different Loss Functions (y=2.0 or 1.0, F=[0.5, 1.0, 1.5]):")
    print(f"  MSE   (y=2, F={F_demo}): r = {pseudo_residuals_mse(y_demo_reg, F_demo)}")
    print(f"  Log   (y=1, F={F_demo}): r = {np.round(pseudo_residuals_logloss(y_demo_cls, F_demo), 4)}")
    print(f"  MAE   (y=2, F={F_demo}): r = {pseudo_residuals_mae(y_demo_reg, F_demo)}")

    # 5. Optional sklearn validation
    if HAS_SKLEARN:
        gbm_sk = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=1, random_state=42
        )
        gbm_sk.fit(X_train, y_train)
        sk_test_r2 = gbm_sk.score(X_test, y_test)
        print(f"\nScikit-learn GBM R²: {sk_test_r2:.4f}  (Scratch GBM R²: {test_r2:.4f})")
        assert abs(test_r2 - sk_test_r2) < 0.15, "Scratch and sklearn should have similar R²"

    # Assertions
    assert final_train_mse < baseline_mse_train * 0.5, "GBM must reduce training MSE by at least 50%"
    assert test_r2 > 0.7, f"Test R² should exceed 0.7, got {test_r2:.4f}"

    print("\n✓ GBM residuals and gradient descent in function space verified!")


if __name__ == "__main__":
    main()

