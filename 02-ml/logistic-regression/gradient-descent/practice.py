# Practice: Gradient Descent for Logistic Regression from Scratch
# Model:     p_i = σ(z_i),  z_i = w·x_i + b
# Loss:      J(w, b) = -(1/n) Σ [ y_i*log(p_i) + (1 - y_i)*log(1 - p_i) ]
# Gradients (the sigmoid and log cancel neatly):
#   ∂J/∂z_i = p_i - y_i
#   ∂J/∂w   = (1/n) X^T (p - y)
#   ∂J/∂b   = (1/n) Σ (p_i - y_i)
# Update:    w ← w - α * ∂J/∂w,   b ← b - α * ∂J/∂b

import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true, y_prob, eps=1e-15):
    p = np.clip(y_prob, eps, 1.0 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def make_dataset(n=200, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 2))
    p = sigmoid(X @ np.array([2.0, -1.5]) + 0.5)
    y = (rng.uniform(size=n) < p).astype(float)
    return X, y


def compute_gradients(X, y, w, b):
    """Analytical gradients of the BCE loss with respect to w and b."""
    n = len(y)
    p = sigmoid(X @ w + b)
    error = p - y  # ∂J/∂z
    dw = (X.T @ error) / n
    db = np.mean(error)
    return dw, db


def loss_at(X, y, w, b):
    return binary_cross_entropy(y, sigmoid(X @ w + b))


def train_gradient_descent(X, y, learning_rate=0.1, epochs=1000):
    """Batch gradient descent for logistic regression."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        dw, db = compute_gradients(X, y, w, b)
        w -= learning_rate * dw
        b -= learning_rate * db

        if epoch % 200 == 0 or epoch == 1:
            current = loss_at(X, y, w, b)
            history.append((epoch, current))
            print(f"Epoch {epoch:4d} | w={np.round(w, 4)}, b={b:.4f} | BCE={current:.6f}")

    return w, b, history


if __name__ == "__main__":
    X, y = make_dataset()

    print("--- 1. Gradient Check (Analytical vs Finite Difference) ---")
    rng = np.random.default_rng(1)
    w_test = rng.normal(0, 0.5, X.shape[1])
    b_test = 0.2
    dw, db = compute_gradients(X, y, w_test, b_test)

    h = 1e-6
    num_dw = np.zeros_like(w_test)
    for j in range(len(w_test)):
        e = np.zeros_like(w_test)
        e[j] = h
        num_dw[j] = (loss_at(X, y, w_test + e, b_test) - loss_at(X, y, w_test - e, b_test)) / (2 * h)
    num_db = (loss_at(X, y, w_test, b_test + h) - loss_at(X, y, w_test, b_test - h)) / (2 * h)

    print(f"dw analytical: {np.round(dw, 6)} | numerical: {np.round(num_dw, 6)}")
    print(f"db analytical: {db:.6f} | numerical: {num_db:.6f}")
    assert np.allclose(dw, num_dw, atol=1e-6) and np.isclose(db, num_db, atol=1e-6), \
        "Analytical gradients do not match finite differences"

    print("\n--- 2. Training ---")
    w, b, history = train_gradient_descent(X, y, learning_rate=0.1, epochs=1000)

    print("\n--- 3. Results ---")
    print(f"Learned weights: {np.round(w, 4)} (true generating weights: [2.0, -1.5])")
    print(f"Learned bias:    {b:.4f} (true generating bias: 0.5)")

    initial_loss = loss_at(X, y, np.zeros(X.shape[1]), 0.0)
    final_loss = loss_at(X, y, w, b)
    accuracy = np.mean((sigmoid(X @ w + b) >= 0.5) == y)
    print(f"Initial BCE: {initial_loss:.6f} (ln 2 = {np.log(2):.6f})")
    print(f"Final BCE:   {final_loss:.6f}")
    print(f"Training accuracy: {accuracy:.4f}")

    assert final_loss < initial_loss, "Gradient descent should reduce the loss"
    assert accuracy > 0.8, "Model should classify the data well"
    print("\nGradient descent converged: the loss decreased and the decision boundary was learned.")
