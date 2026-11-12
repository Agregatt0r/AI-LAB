"""
practice.py — Gradient Boosting: scratch implementation + scikit-learn API demo
================================================================================
Demonstrates:
  1. Pure-NumPy GBM classifier (log-loss + decision stumps)
  2. Hyperparameter sweep over learning rate
  3. scikit-learn GradientBoostingClassifier comparison (if available)
  4. staged_predict learning curve (if sklearn available)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ---------------------------------------------------------------------------
# Depth-1 regression tree (stump)
# ---------------------------------------------------------------------------

class RegressionStump:
    """Depth-1 regression tree."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegressionStump":
        best_loss = np.inf
        self.feature_, self.threshold_, self.left_val_, self.right_val_ = 0, 0.0, 0.0, 0.0
        for f in range(X.shape[1]):
            for t in np.unique(X[:, f]):
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(
            X[:, self.feature_] <= self.threshold_,
            self.left_val_,
            self.right_val_,
        )


# ---------------------------------------------------------------------------
# GBM classifier — pure NumPy
# ---------------------------------------------------------------------------

class GBMClassifierScratch:
    """Binary GBM classifier using log-loss and decision stumps."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBMClassifierScratch":
        # Initialize: log-odds of mean probability
        p0 = np.clip(y.mean(), 1e-7, 1 - 1e-7)
        self.F0_ = np.log(p0 / (1 - p0))
        F = np.full(len(y), self.F0_)
        self.trees_ = []
        for _ in range(self.n_estimators):
            # Pseudo-residuals for log-loss: y - sigmoid(F)
            r = y - sigmoid(F)
            t = RegressionStump().fit(X, r)
            F += self.learning_rate * t.predict(X)
            self.trees_.append(t)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        F = np.full(len(X), self.F0_)
        for t in self.trees_:
            F += self.learning_rate * t.predict(X)
        prob = sigmoid(F)
        return np.column_stack([1 - prob, prob])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def staged_predict(self, X: np.ndarray):
        """Yield predictions after each boosting stage (like sklearn's API)."""
        F = np.full(len(X), self.F0_)
        for t in self.trees_:
            F += self.learning_rate * t.predict(X)
            prob = sigmoid(F)
            yield (prob >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

rng = np.random.RandomState(42)
n = 400
X = rng.randn(n, 2)
y = ((X[:, 0] ** 2 + X[:, 1] ** 2) < 1.5).astype(float)  # inside circle = class 1

split = int(0.7 * n)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("=" * 60)
print("Dataset: circular boundary (n=400, 70/30 split)")
print(f"  Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")
print(f"  Class balance (train): {y_train.mean():.2%} positive")
print("=" * 60)


# ---------------------------------------------------------------------------
# Demo 1: GBMClassifierScratch
# ---------------------------------------------------------------------------

print("\n── Demo 1: GBMClassifierScratch (n_estimators=200, lr=0.1) ──")
scratch_clf = GBMClassifierScratch(n_estimators=200, learning_rate=0.1)
scratch_clf.fit(X_train, y_train)

train_acc = (scratch_clf.predict(X_train) == y_train).mean()
test_acc  = (scratch_clf.predict(X_test)  == y_test ).mean()

print(f"  Train accuracy : {train_acc:.4f}")
print(f"  Test  accuracy : {test_acc:.4f}")

assert test_acc >= 0.70, f"Expected test_acc >= 0.70, got {test_acc:.4f}"
print("  ✓ test_acc >= 0.70")


# ---------------------------------------------------------------------------
# Demo 2: Hyperparameter sweep — learning rate
# ---------------------------------------------------------------------------

print("\n── Demo 2: Learning-rate sweep (n_estimators=200) ──")
print(f"  {'learning_rate':>15} | {'Test Accuracy':>14}")
print("  " + "-" * 32)

lr_values = [0.01, 0.05, 0.1, 0.3]
lr_results = {}
for lr in lr_values:
    clf_lr = GBMClassifierScratch(n_estimators=200, learning_rate=lr)
    clf_lr.fit(X_train, y_train)
    acc = (clf_lr.predict(X_test) == y_test).mean()
    lr_results[lr] = acc
    print(f"  {lr:>15.2f} | {acc:>14.4f}")


# ---------------------------------------------------------------------------
# Assertion: more estimators improve learning (using staged_predict on scratch)
# ---------------------------------------------------------------------------

print("\n── Assertion: accuracy improves with more estimators ──")
scratch_clf_10 = GBMClassifierScratch(n_estimators=10, learning_rate=0.1)
scratch_clf_10.fit(X_train, y_train)
acc_10  = (scratch_clf_10.predict(X_test) == y_test).mean()
acc_200 = test_acc  # already computed above

print(f"  Accuracy @ n=10  : {acc_10:.4f}")
print(f"  Accuracy @ n=200 : {acc_200:.4f}")
assert acc_200 >= acc_10, (
    f"Expected acc@200 ({acc_200:.4f}) >= acc@10 ({acc_10:.4f})"
)
print("  ✓ acc@200 >= acc@10")


# ---------------------------------------------------------------------------
# Demo 3: scikit-learn GradientBoostingClassifier (optional)
# ---------------------------------------------------------------------------

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score

    print("\n── Demo 3: sklearn GradientBoostingClassifier ──")

    sk_clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=1,
        random_state=42,
    )
    sk_clf.fit(X_train, y_train)

    sk_train_acc = accuracy_score(y_train, sk_clf.predict(X_train))
    sk_test_acc  = accuracy_score(y_test,  sk_clf.predict(X_test))

    print("\n  Comparison table (n_estimators=200, lr=0.1, max_depth=1):")
    print(f"  {'Model':>22} | {'Train Acc':>10} | {'Test Acc':>10}")
    print("  " + "-" * 48)
    print(f"  {'GBMClassifierScratch':>22} | {train_acc:>10.4f} | {test_acc:>10.4f}")
    print(f"  {'sklearn GBM':>22} | {sk_train_acc:>10.4f} | {sk_test_acc:>10.4f}")

    # staged_predict learning curve
    print("\n── staged_predict: test accuracy per stage ──")
    print(f"  {'Stage':>8} | {'Test Accuracy':>14}")
    print("  " + "-" * 26)
    target_stages = {10, 50, 100, 200}
    for stage_idx, y_stage in enumerate(sk_clf.staged_predict(X_test), start=1):
        if stage_idx in target_stages:
            stage_acc = accuracy_score(y_test, y_stage)
            print(f"  {stage_idx:>8} | {stage_acc:>14.4f}")

except ImportError:
    print("\n  [sklearn not installed — skipping Demo 3]")


# ---------------------------------------------------------------------------
# Final
# ---------------------------------------------------------------------------

print("\n✓ scikit-learn GBM API and scratch classifier verified!")
