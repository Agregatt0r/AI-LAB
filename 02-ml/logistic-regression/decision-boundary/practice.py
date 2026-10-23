# Practice: Decision Boundary of Logistic Regression from Scratch
# Model:            z = w1*x1 + w2*x2 + b
# Decision boundary: z = 0  (equivalently σ(z) = 0.5)
# Solving for x2:   x2 = -(w1*x1 + b) / w2
# The boundary is a straight line (a hyperplane in higher dimensions).

import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def decision_boundary_x2(x1, w, b):
    """Return x2 values lying exactly on the boundary for the given x1 values."""
    return -(w[0] * x1 + b) / w[1]


def predict_by_boundary(X, w, b):
    """Class 1 when the point lies on the positive side of the line: w2*(x2 - boundary) >= 0."""
    boundary = decision_boundary_x2(X[:, 0], w, b)
    return (w[1] * (X[:, 1] - boundary) >= 0).astype(int)


def predict_by_threshold(X, w, b, threshold=0.5):
    """Same rule expressed through probabilities: σ(z) >= threshold."""
    return (sigmoid(X @ w + b) >= threshold).astype(int)


if __name__ == "__main__":
    w = np.array([2.0, -1.5])
    b = 0.5

    print("--- 1. Boundary Points Have Probability 0.5 ---")
    x1 = np.linspace(-3, 3, 7)
    x2 = decision_boundary_x2(x1, w, b)
    boundary_points = np.column_stack([x1, x2])
    proba_on_boundary = sigmoid(boundary_points @ w + b)
    for p, pr in zip(boundary_points, proba_on_boundary):
        print(f"x1={p[0]:+.2f}, x2={p[1]:+.4f} -> P(y=1|x) = {pr:.6f}")
    assert np.allclose(proba_on_boundary, 0.5), "Points on the boundary must have probability 0.5"

    print("\n--- 2. Points on Either Side of the Boundary ---")
    below = np.array([[0.0, decision_boundary_x2(0.0, w, b) - 0.5]])
    above = np.array([[0.0, decision_boundary_x2(0.0, w, b) + 0.5]])
    print(f"Below the line -> class {predict_by_boundary(below, w, b)[0]} (P = {sigmoid(below @ w + b)[0]:.4f})")
    print(f"Above the line -> class {predict_by_boundary(above, w, b)[0]} (P = {sigmoid(above @ w + b)[0]:.4f})")

    print("\n--- 3. Geometric Rule Matches the Threshold Rule ---")
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1.5, (500, 2))
    agree = np.mean(predict_by_boundary(X, w, b) == predict_by_threshold(X, w, b))
    print(f"Agreement between boundary rule and σ(z) >= 0.5 rule: {agree * 100:.2f}%")
    assert agree == 1.0, "Boundary rule and threshold rule must give identical labels"

    print("\n--- 4. Boundary Moves When Parameters Change ---")
    for bias in [-1.0, 0.0, 1.0]:
        y_line = decision_boundary_x2(0.0, w, bias)
        print(f"b={bias:+.1f} -> boundary crosses x2 axis at {y_line:+.4f}")

    print("\nThe weight vector w is perpendicular to the boundary and points toward class 1.")
