# Practice: Vectorized Linear Regression with NumPy
# Fast vectorized operations instead of slow Python loops:
#   y_pred = w * x + b
#   mse = np.mean((y_true - y_pred) ** 2)
#   dw = (2 / n) * np.dot(errors, x)
#   db = (2 / n) * np.sum(errors)

import time
import numpy as np


def train_vectorized(x, y, learning_rate=0.05, epochs=1000):
    """Train linear regression using vectorized NumPy operations."""
    w = 0.0
    b = 0.0
    n = len(x)

    for _ in range(epochs):
        y_pred = w * x + b
        errors = y_pred - y

        dw = (2.0 / n) * np.dot(errors, x)
        db = (2.0 / n) * np.sum(errors)

        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b


if __name__ == "__main__":
    # 1. Simple Verification Example
    x = np.array([1.0, 2.0, 3.0])
    y_true = np.array([3.0, 5.0, 7.0])

    w = 2.0
    b = 1.0

    # Vectorized forward pass
    y_pred = w * x + b
    mse = np.mean((y_true - y_pred) ** 2)

    print("--- 1. Vectorized Prediction & Loss ---")
    print("x:          ", x)
    print("y_true:     ", y_true)
    print("y_pred:     ", y_pred)
    print(f"MSE:         {mse:.6f}\n")

    # 2. Vectorized Training Loop
    print("--- 2. Vectorized Training ---")
    w_fit, b_fit = train_vectorized(x, y_true, learning_rate=0.05, epochs=1500)
    print(f"Fitted w: {w_fit:.4f} (Expected: ~2.0)")
    print(f"Fitted b: {b_fit:.4f} (Expected: ~1.0)\n")

    assert np.isclose(w_fit, 2.0, atol=1e-2)
    assert np.isclose(b_fit, 1.0, atol=1e-2)

    # 3. Performance Benchmark: Loops vs Vectorization on 100,000 points
    print("--- 3. Performance Benchmark (100,000 samples) ---")
    np.random.seed(42)
    big_x = np.random.rand(100_000)
    big_y = 3.5 * big_x + 2.0 + np.random.normal(0, 0.1, 100_000)

    # Benchmark: Loop
    start_loop = time.perf_counter()
    loop_preds = [3.5 * val + 2.0 for val in big_x]
    loop_mse = sum((yt - yp) ** 2 for yt, yp in zip(big_y, loop_preds)) / len(big_y)
    time_loop = time.perf_counter() - start_loop

    # Benchmark: NumPy Vectorized
    start_vec = time.perf_counter()
    vec_preds = 3.5 * big_x + 2.0
    vec_mse = np.mean((big_y - vec_preds) ** 2)
    time_vec = time.perf_counter() - start_vec

    speedup = time_loop / time_vec
    print(f"Pure Python Loop: {time_loop * 1000:.2f} ms")
    print(f"NumPy Vectorized: {time_vec * 1000:.2f} ms")
    print(f"Vectorization Speedup: {speedup:.1f}x faster!")
