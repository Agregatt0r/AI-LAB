# Practice: Binary Classification & Why Linear Regression Is Not Enough
# Target: y ∈ {0, 1}  (0 = fail, 1 = pass)
# Linear model (regression on labels): ŷ = w * x + b
# Problem: ŷ is unbounded (can be < 0 or > 1), so it cannot be a probability.

import numpy as np


def make_binary_dataset(n=100, seed=42):
    """Hours studied (x) -> pass (1) if more than 5 hours, else fail (0)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    y = (x > 5).astype(float)
    return x, y


def fit_linear_regression(x, y):
    """Ordinary least squares on the 0/1 labels (treating them as numbers)."""
    X = np.column_stack([np.ones(len(x)), x])
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    b, w = theta[0], theta[1]
    return b, w


def linear_score(x, b, w):
    return w * x + b


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


if __name__ == "__main__":
    x, y = make_binary_dataset()

    print("--- 1. Binary Labels ---")
    classes, counts = np.unique(y, return_counts=True)
    print(f"Classes: {classes.tolist()} | Counts: {counts.tolist()}")

    b, w = fit_linear_regression(x, y)
    scores = linear_score(x, b, w)

    print("\n--- 2. Linear Regression on Labels ---")
    print(f"Learned w: {w:.4f}, b: {b:.4f}")
    print(f"Output range: [{scores.min():.4f}, {scores.max():.4f}]  (probabilities must be in [0, 1])")

    # Outputs fall outside [0, 1], so they are not valid probabilities
    assert scores.min() < 0 and scores.max() > 1, "Expected raw linear outputs outside [0, 1]"

    print("\n--- 3. Thresholding the Linear Output at 0.5 ---")
    y_pred = (scores >= 0.5).astype(float)
    acc = accuracy(y, y_pred)
    print(f"Accuracy: {acc:.4f}")

    assert acc > 0.95, "Thresholded linear model should separate this simple dataset"
    print("\nClassification is solved by a bounded, non-linear output (sigmoid), see the next module.")
