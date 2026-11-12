"""
RF vs Gradient Boosting — scratch implementations + sklearn comparison.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Scratch implementations
# ---------------------------------------------------------------------------

class RegressionStump:
    def fit(self, X, y):
        best_loss = np.inf
        self.feature_, self.threshold_, self.left_val_, self.right_val_ = 0, 0.0, 0.0, 0.0
        for f in range(X.shape[1]):
            for t in np.unique(X[:, f]):
                left = y[X[:, f] <= t]; right = y[X[:, f] > t]
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


class BaggingRegressorScratch:
    """Simple bagging regressor (Random Forest analog) using stumps."""

    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.trees_ = []
        n = len(X)
        for _ in range(self.n_estimators):
            idx = rng.randint(0, n, size=n)  # bootstrap sample
            t = RegressionStump().fit(X[idx], y[idx])
            self.trees_.append(t)
        return self

    def predict(self, X):
        return np.mean([t.predict(X) for t in self.trees_], axis=0)


class GBMRegressorScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        self.F0_ = y.mean()
        F = np.full(len(y), self.F0_)
        self.trees_ = []
        for _ in range(self.n_estimators):
            r = y - F
            t = RegressionStump().fit(X, r)
            F += self.learning_rate * t.predict(X)
            self.trees_.append(t)
        return self

    def predict(self, X):
        F = np.full(len(X), self.F0_)
        for t in self.trees_:
            F += self.learning_rate * t.predict(X)
        return F


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

rng = np.random.RandomState(42)

# Dataset 1 — Low Noise (clean)
X1 = rng.randn(300, 3)
y1 = np.sin(X1[:, 0]) + X1[:, 1] ** 2 + 0.1 * rng.randn(300)

# Dataset 2 — High Noise
X2 = rng.randn(300, 3)
y2 = np.sin(X2[:, 0]) + X2[:, 1] ** 2 + 2.0 * rng.randn(300)

# Dataset 3 — Small
X3 = rng.randn(60, 3)
y3 = np.sin(X3[:, 0]) + X3[:, 1] ** 2 + 0.3 * rng.randn(60)

datasets = [
    ("Low Noise   ", X1, y1),
    ("High Noise  ", X2, y2),
    ("Small Dataset", X3, y3),
]


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def split(X, y, test_ratio=0.3, seed=0):
    rng_s = np.random.RandomState(seed)
    n = len(X)
    idx = rng_s.permutation(n)
    split_at = int(n * (1 - test_ratio))
    train_idx, test_idx = idx[:split_at], idx[split_at:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ---------------------------------------------------------------------------
# Scratch results
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  Scratch Implementations (Bagging Stumps vs GBM Stumps)")
print("=" * 60)
header = f"{'Dataset':<17} | {'Bagging R²':>10} | {'GBM R²':>8} | {'Winner':<10}"
print(header)
print("-" * len(header))

r2_low_noise_bagging = None
r2_low_noise_gbm = None

for name, X, y in datasets:
    X_tr, X_te, y_tr, y_te = split(X, y)

    bag = BaggingRegressorScratch(n_estimators=100).fit(X_tr, y_tr)
    gbm = GBMRegressorScratch(n_estimators=100, learning_rate=0.1).fit(X_tr, y_tr)

    r2_bag = r2_score(y_te, bag.predict(X_te))
    r2_gbm = r2_score(y_te, gbm.predict(X_te))

    winner = "GBM" if r2_gbm > r2_bag else "Bagging"
    print(f"{name:<17} | {r2_bag:>10.3f} | {r2_gbm:>8.3f} | {winner:<10}")

    if name.strip() == "Low Noise":
        r2_low_noise_bagging = r2_bag
        r2_low_noise_gbm = r2_gbm

# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

assert r2_low_noise_bagging is not None and r2_low_noise_gbm is not None, (
    "Low Noise dataset results not captured."
)
assert r2_low_noise_bagging > 0, (
    f"Bagging R² on Low Noise dataset should be > 0, got {r2_low_noise_bagging:.3f}"
)
assert r2_low_noise_gbm > 0, (
    f"GBM R² on Low Noise dataset should be > 0, got {r2_low_noise_gbm:.3f}"
)

# ---------------------------------------------------------------------------
# sklearn comparison (optional)
# ---------------------------------------------------------------------------

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    print("\n" + "=" * 60)
    print("  sklearn: RandomForestRegressor vs GradientBoostingRegressor")
    print("=" * 60)
    header_sk = f"{'Dataset':<17} | {'RF R²':>8} | {'GBM R²':>8} | {'Winner':<10}"
    print(header_sk)
    print("-" * len(header_sk))

    for name, X, y in datasets:
        X_tr, X_te, y_tr, y_te = split(X, y)

        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
        gbm_sk = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42
        ).fit(X_tr, y_tr)

        r2_rf = r2_score(y_te, rf.predict(X_te))
        r2_gbm_sk = r2_score(y_te, gbm_sk.predict(X_te))

        winner = "GBM" if r2_gbm_sk > r2_rf else "RF"
        print(f"{name:<17} | {r2_rf:>8.3f} | {r2_gbm_sk:>8.3f} | {winner:<10}")

    print("\n  [sklearn available — extended comparison shown above]")

except ImportError:
    print("\n  [sklearn not available — skipping extended comparison]")

print("\n✓ RF vs Gradient Boosting comparison verified!")
