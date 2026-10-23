# Practice: The Sigmoid (Logistic) Function from Scratch
# Sigmoid:       σ(z) = 1 / (1 + e^(-z))
# Derivative:    σ'(z) = σ(z) * (1 - σ(z))
# Symmetry:      σ(-z) = 1 - σ(z)

import numpy as np


def sigmoid(z):
    """Map any real number (or array) into the open interval (0, 1)."""
    # Clip to avoid overflow in exp for very large |z|; σ is already ~0 or ~1 there.
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1.0 - s)


if __name__ == "__main__":
    print("--- 1. Key Values ---")
    print(f"σ(0)   = {sigmoid(0.0):.4f}  (decision midpoint)")
    print(f"σ(2)   = {sigmoid(2.0):.4f}")
    print(f"σ(-2)  = {sigmoid(-2.0):.4f}")
    print(f"σ(10)  = {sigmoid(10.0):.6f}  (saturates near 1)")
    print(f"σ(-10) = {sigmoid(-10.0):.6f} (saturates near 0)")

    assert np.isclose(sigmoid(0.0), 0.5), "σ(0) must be 0.5"

    print("\n--- 2. Output Range ---")
    z = np.linspace(-8, 8, 17)
    s = sigmoid(z)
    print(f"Min: {s.min():.6f}, Max: {s.max():.6f}")
    assert np.all((s > 0) & (s < 1)), "Sigmoid outputs must lie strictly in (0, 1)"

    print("\n--- 3. Symmetry: σ(-z) = 1 - σ(z) ---")
    assert np.allclose(sigmoid(-z), 1.0 - sigmoid(z)), "Symmetry property failed"
    print("Symmetry holds for all test points.")

    print("\n--- 4. Derivative vs Numerical Finite Difference ---")
    h = 1e-5
    numerical = (sigmoid(z + h) - sigmoid(z - h)) / (2 * h)
    analytical = sigmoid_derivative(z)
    max_err = np.max(np.abs(numerical - analytical))
    print(f"Max |numerical - analytical| = {max_err:.2e}")
    assert max_err < 1e-8, "Analytical derivative does not match finite difference"

    print("\n--- 5. Large Inputs Do Not Overflow ---")
    big = sigmoid(np.array([-1000.0, 1000.0]))
    print(f"σ(-1000) = {big[0]}, σ(1000) = {big[1]}")
    assert np.all(np.isfinite(big)), "Sigmoid produced non-finite values"

    print("\nSigmoid behaves as expected: bounded, smooth, and centered at 0.5.")
