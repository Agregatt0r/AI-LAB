# Practice: Gradient Descent Optimization from Scratch
# Model: ŷ = w * x + b
# Cost: MSE = (1/n) * Σ(ŷ - y)^2
# Gradients:
#   dw = (2/n) * Σ(ŷ - y) * x
#   db = (2/n) * Σ(ŷ - y)

def predict(x, w, b):
    return w * x + b


def compute_mse(y_true, y_pred):
    n = len(y_true)
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n


def train_gradient_descent(x, y, w_init=0.0, b_init=0.0, learning_rate=0.05, epochs=2000):
    """Train linear regression using batch gradient descent."""
    w = w_init
    b = b_init
    n = len(x)
    history = []

    for epoch in range(1, epochs + 1):
        dw = 0.0
        db = 0.0

        for i in range(n):
            y_pred = predict(x[i], w, b)
            error = y_pred - y[i]  # (ŷ - y)

            dw += error * x[i]
            db += error

        dw = (2.0 / n) * dw
        db = (2.0 / n) * db

        # Parameter update
        w -= learning_rate * dw
        b -= learning_rate * db

        if epoch % 400 == 0 or epoch == 1:
            preds = [predict(val, w, b) for val in x]
            current_loss = compute_mse(y, preds)
            history.append((epoch, w, b, current_loss))
            print(f"Epoch {epoch:4d} | w={w:.4f}, b={b:.4f} | MSE={current_loss:.6f}")

    return w, b, history


if __name__ == "__main__":
    # Relationship: y = 2*x + 1
    x = [1, 2, 3]
    y = [3, 5, 7]

    print("--- Training Linear Regression with Gradient Descent ---")
    w, b, history = train_gradient_descent(x, y, w_init=0.0, b_init=0.0, learning_rate=0.05, epochs=2000)

    final_preds = [predict(val, w, b) for val in x]
    final_mse = compute_mse(y, final_preds)

    print("\n--- Training Results ---")
    print(f"Learned Weight (w): {w:.4f} (Expected: ~2.0000)")
    print(f"Learned Bias   (b): {b:.4f} (Expected: ~1.0000)")
    print(f"Final Predictions:  {[round(p, 4) for p in final_preds]}")
    print(f"Final MSE:          {final_mse:.8f}")

    assert abs(w - 2.0) < 1e-2, "Weight did not converge to 2"
    assert abs(b - 1.0) < 1e-2, "Bias did not converge to 1"
    print("\nModel successfully converged to optimal line parameters!")
