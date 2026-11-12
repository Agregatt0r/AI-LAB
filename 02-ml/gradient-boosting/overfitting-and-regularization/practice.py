"""
Overfitting and Regularization in Gradient Boosting
=====================================================
Demonstrates:
  1. Overfitting curve (train vs val loss over boosting rounds)
  2. Stochastic GBM via subsample parameter
  3. Learning rate regularization
  4. (Optional) sklearn GradientBoostingRegressor comparison
"""

import numpy as np

# ---------------------------------------------------------------------------
# Core classes
# ---------------------------------------------------------------------------

class RegressionStump:
    def fit(self, X, y):
        best_loss = np.inf
        self.feature_, self.threshold_, self.left_val_, self.right_val_ = 0, 0.0, 0.0, 0.0
        for f in range(X.shape[1]):
            for t in np.unique(X[:, f]):
                left = y[X[:, f] <= t]; right = y[X[:, f] > t]
                if len(left) == 0 or len(right) == 0: continue
                loss = len(left)*left.var() + len(right)*right.var()
                if loss < best_loss:
                    best_loss = loss; self.feature_ = f; self.threshold_ = t
                    self.left_val_ = left.mean(); self.right_val_ = right.mean()
        return self
    def predict(self, X):
        return np.where(X[:, self.feature_] <= self.threshold_, self.left_val_, self.right_val_)


class GBMRegressorScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, subsample=1.0, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y, X_val=None, y_val=None):
        rng = np.random.RandomState(self.random_state)
        self.F0_ = y.mean()
        F = np.full(len(y), self.F0_)
        self.trees_ = []
        self.train_losses_ = []
        self.val_losses_ = []
        for _ in range(self.n_estimators):
            r = y - F
            # stochastic: subsample rows
            n = len(X)
            idx = rng.choice(n, size=max(1, int(n * self.subsample)), replace=False)
            t = RegressionStump().fit(X[idx], r[idx])
            F += self.learning_rate * t.predict(X)
            self.trees_.append(t)
            self.train_losses_.append(np.mean((y - F)**2))
            if X_val is not None:
                F_val = self.predict_staged(X_val, len(self.trees_))
                self.val_losses_.append(np.mean((y_val - F_val)**2))
        return self

    def predict_staged(self, X, n_trees=None):
        trees = self.trees_[:n_trees]
        F = np.full(len(X), self.F0_)
        for t in trees:
            F += self.learning_rate * t.predict(X)
        return F

    def predict(self, X):
        return self.predict_staged(X)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

rng0 = np.random.RandomState(0)
X = rng0.randn(400, 3)
y = np.sin(2 * X[:, 0]) + X[:, 1]**2 - X[:, 2] + 0.5 * rng0.randn(400)

X_train, y_train = X[:300], y[:300]
X_test,  y_test  = X[300:], y[300:]

print("=" * 60)
print("Dataset: 400 samples, 3 features")
print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
print("=" * 60)


# ---------------------------------------------------------------------------
# Demo 1 — Overfitting demonstration
# ---------------------------------------------------------------------------

print("\n--- Demo 1: Overfitting Demonstration ---")

gbm_overfit = GBMRegressorScratch(
    n_estimators=500,
    learning_rate=0.5,
    subsample=1.0,
    random_state=42,
)
gbm_overfit.fit(X_train, y_train, X_val=X_test, y_val=y_test)

val_losses = np.array(gbm_overfit.val_losses_)
train_losses = np.array(gbm_overfit.train_losses_)

best_stage = int(np.argmin(val_losses))          # 0-indexed
best_val_loss = val_losses[best_stage]
train_at_best = train_losses[best_stage]
final_val_loss = val_losses[-1]
final_train_loss = train_losses[-1]

print(f"  Best stage (early-stop point): round {best_stage + 1}")
print(f"  Train loss at best stage : {train_at_best:.4f}")
print(f"  Val   loss at best stage : {best_val_loss:.4f}")
print(f"  Train loss at round 500  : {final_train_loss:.4f}")
print(f"  Val   loss at round 500  : {final_val_loss:.4f}")
print(f"  → Val loss rose by {final_val_loss - best_val_loss:.4f} after optimal stage (overfitting)")

assert best_val_loss < final_val_loss, (
    f"Expected val loss at optimal stage ({best_val_loss:.4f}) < "
    f"val loss at round 500 ({final_val_loss:.4f})"
)
print("  ✓ Overfitting confirmed: val loss rises after optimal stage")

# ---------------------------------------------------------------------------
# Demo 2 — Subsample (Stochastic GBM) comparison
# ---------------------------------------------------------------------------

print("\n--- Demo 2: Subsample Comparison (Stochastic GBM) ---")

gbm_full = GBMRegressorScratch(
    n_estimators=200, learning_rate=0.1, subsample=1.0, random_state=42
)
gbm_full.fit(X_train, y_train)
mse_full = float(np.mean((y_test - gbm_full.predict(X_test))**2))

gbm_stoch = GBMRegressorScratch(
    n_estimators=200, learning_rate=0.1, subsample=0.5, random_state=42
)
gbm_stoch.fit(X_train, y_train)
mse_stoch = float(np.mean((y_test - gbm_stoch.predict(X_test))**2))

print(f"  subsample=1.0 (full batch)  → test MSE: {mse_full:.4f}")
print(f"  subsample=0.5 (stochastic)  → test MSE: {mse_stoch:.4f}")

assert mse_full < 1.5,  f"Full-batch GBM test MSE too high: {mse_full:.4f}"
assert mse_stoch < 1.5, f"Stochastic GBM test MSE too high: {mse_stoch:.4f}"
print("  ✓ Both subsample variants achieve test MSE < 1.5")

# ---------------------------------------------------------------------------
# Demo 3 — Learning rate regularization
# ---------------------------------------------------------------------------

print("\n--- Demo 3: Learning Rate Regularization ---")

gbm_lr_high = GBMRegressorScratch(
    n_estimators=200, learning_rate=0.5, subsample=1.0, random_state=42
)
gbm_lr_high.fit(X_train, y_train)
mse_lr_high = float(np.mean((y_test - gbm_lr_high.predict(X_test))**2))

gbm_lr_low = GBMRegressorScratch(
    n_estimators=200, learning_rate=0.05, subsample=1.0, random_state=42
)
gbm_lr_low.fit(X_train, y_train)
mse_lr_low = float(np.mean((y_test - gbm_lr_low.predict(X_test))**2))

print(f"  lr=0.50 (overfit risk)  → test MSE: {mse_lr_high:.4f}")
print(f"  lr=0.05 (regularized)   → test MSE: {mse_lr_low:.4f}")
print("  Note: with only 200 trees, low lr may underfit; more trees + low lr = best combo")

# ---------------------------------------------------------------------------
# Demo 4 — sklearn comparison (optional)
# ---------------------------------------------------------------------------

print("\n--- Demo 4: sklearn GradientBoostingRegressor (optional) ---")

try:
    from sklearn.ensemble import GradientBoostingRegressor

    sklearn_gbm = GradientBoostingRegressor(
        n_estimators=200,
        subsample=0.5,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    sklearn_gbm.fit(X_train, y_train)
    mse_sklearn = float(np.mean((y_test - sklearn_gbm.predict(X_test))**2))

    print(f"  sklearn  subsample=0.5  → test MSE: {mse_sklearn:.4f}")
    print(f"  scratch  subsample=0.5  → test MSE: {mse_stoch:.4f}")
    print(f"  Difference: {abs(mse_sklearn - mse_stoch):.4f}")
    print("  ✓ sklearn available and cross-check complete")

except ImportError:
    print("  sklearn not available — skipping cross-check")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Summary of Regularization Techniques:")
print(f"  Optimal early-stop stage : round {best_stage + 1} / 500")
print(f"  Val loss improvement     : {final_val_loss - best_val_loss:+.4f} (overfit saved)")
print(f"  Stochastic GBM MSE       : {mse_stoch:.4f}  vs full {mse_full:.4f}")
print(f"  Low lr (0.05) MSE        : {mse_lr_low:.4f}  vs high lr {mse_lr_high:.4f}")
print("=" * 60)

print("\n✓ Overfitting and regularization verified!")
