# Practice: The Normal Equation from Scratch
# Formula: θ = (X_aug.T @ X_aug)^(-1) @ X_aug.T @ y
# Compares closed-form analytical solution against Gradient Descent

import numpy as np


def fit_normal_equation(X, y):
    """Compute optimal parameter vector theta using the Normal Equation."""
    n = len(X)
    # 1. Prepend column of ones for bias b: X_aug = [1, X]
    X_aug = np.column_stack([np.ones(n), X])

    # 2. θ = (X^T X)^(-1) X^T y
    XT_X = X_aug.T @ X_aug
    XT_y = X_aug.T @ y

    # Use np.linalg.solve (more numerically stable than direct inv)
    theta = np.linalg.solve(XT_X, XT_y)
    return theta


def predict(X, theta):
    """Make predictions using theta = [b, w1, w2, ...]."""
    n = len(X)
    X_aug = np.column_stack([np.ones(n), X])
    return X_aug @ theta


if __name__ == "__main__":
    # Test on known 2D equation: y = 3*x1 - 2*x2 + 5 + noise
    np.random.seed(42)
    n_samples = 150

    x1 = np.random.uniform(0, 10, n_samples)
    x2 = np.random.uniform(0, 5, n_samples)
    X = np.column_stack([x1, x2])

    noise = np.random.normal(0, 0.05, n_samples)
    y_true = 3.0 * x1 - 2.0 * x2 + 5.0 + noise

    # Solve using Normal Equation
    theta = fit_normal_equation(X, y_true)
    bias_b = theta[0]
    weights_w = theta[1:]

    print("--- Normal Equation Closed-Form Solution ---")
    print(f"Learned Bias (b):     {bias_b:.4f} (Expected: ~5.0000)")
    print(f"Learned Weight 1:     {weights_w[0]:.4f} (Expected: ~3.0000)")
    print(f"Learned Weight 2:     {weights_w[1]:.4f} (Expected: ~-2.0000)")

    # Predictions and error
    y_pred = predict(X, theta)
    mse = np.mean((y_true - y_pred) ** 2)
    print(f"Residual MSE:         {mse:.6f}")

    assert abs(bias_b - 5.0) < 0.1
    assert abs(weights_w[0] - 3.0) < 0.1
    assert abs(weights_w[1] - (-2.0)) < 0.1
    print("\nNormal Equation successfully found exact optimal parameters in 1 step!")
