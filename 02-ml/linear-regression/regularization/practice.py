# Practice: Ridge Regression (L2 Regularization) from Scratch
# Loss: J(w, b) = MSE + λ * ||w||^2
# Gradients:
#   dw = (2 / n) * X.T @ (ŷ - y) + 2 * λ * w
#   db = (2 / n) * np.sum(ŷ - y)

import numpy as np


class RegularizedLinearRegression:
    def __init__(self, l2_lambda=0.0, learning_rate=0.01, epochs=1500):
        self.l2_lambda = l2_lambda
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        for _ in range(self.epochs):
            y_pred = X @ self.w + self.b
            errors = y_pred - y

            # L2 regularization adds 2 * lambda * w to weights gradient (bias is not regularized)
            dw = (2.0 / n) * (X.T @ errors) + 2.0 * self.l2_lambda * self.w
            db = (2.0 / n) * np.sum(errors)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        return X @ self.w + self.b


if __name__ == "__main__":
    # Simulate multicollinear data with redundant correlated features
    # True relationship: y = 3*x1 + 0*x2 + noise (x2 is a noisy clone of x1)
    np.random.seed(42)
    n = 80

    x1 = np.random.uniform(1, 10, n)
    x2 = x1 + np.random.normal(0, 0.05, n)  # Highly correlated with x1
    X = np.column_stack([x1, x2])

    noise = np.random.normal(0, 1.0, n)
    y = 3.0 * x1 + 5.0 + noise

    # Standardize X for fair weight comparison
    X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # 1. Unregularized OLS (λ = 0)
    ols = RegularizedLinearRegression(l2_lambda=0.0, learning_rate=0.05, epochs=2000)
    ols.fit(X_scaled, y)

    # 2. Ridge Regression (λ = 0.5)
    ridge = RegularizedLinearRegression(l2_lambda=0.5, learning_rate=0.05, epochs=2000)
    ridge.fit(X_scaled, y)

    # 3. Heavy Regularization (λ = 5.0)
    heavy_ridge = RegularizedLinearRegression(l2_lambda=5.0, learning_rate=0.05, epochs=2000)
    heavy_ridge.fit(X_scaled, y)

    print("--- Effect of L2 Regularization on Weight Magnitudes ---")
    print(f"OLS   (λ = 0.0): Weights = {np.round(ols.w, 4)}, ||w||_2 = {np.linalg.norm(ols.w):.4f}")
    print(f"Ridge (λ = 0.5): Weights = {np.round(ridge.w, 4)}, ||w||_2 = {np.linalg.norm(ridge.w):.4f}")
    print(f"Heavy (λ = 5.0): Weights = {np.round(heavy_ridge.w, 4)}, ||w||_2 = {np.linalg.norm(heavy_ridge.w):.4f}")

    # Verify that L2 penalty reduces the norm of weights
    norm_ols = np.linalg.norm(ols.w)
    norm_ridge = np.linalg.norm(ridge.w)
    norm_heavy = np.linalg.norm(heavy_ridge.w)

    assert norm_heavy < norm_ridge < norm_ols, "L2 regularization failed to shrink weight norms!"
    print("\nVerified: Increasing λ successfully shrinks weights and stabilizes model!")
