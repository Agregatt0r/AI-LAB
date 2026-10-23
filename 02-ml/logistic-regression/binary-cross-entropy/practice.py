# Practice: Binary Cross-Entropy (Log Loss) from Scratch
# For one sample:  ℓ = -[ y*log(p) + (1 - y)*log(1 - p) ]
# Over n samples:  BCE = -(1/n) * Σ [ y_i*log(p_i) + (1 - y_i)*log(1 - p_i) ]
# Derived from Maximum Likelihood: minimizing BCE = maximizing the Bernoulli log-likelihood.

import numpy as np
from sklearn.metrics import log_loss


def binary_cross_entropy(y_true, y_prob, eps=1e-15):
    """Average negative log-likelihood of Bernoulli labels given predicted probabilities."""
    p = np.clip(y_prob, eps, 1.0 - eps)  # avoid log(0)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def single_sample_loss(y, p):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


if __name__ == "__main__":
    y = np.array([1, 0, 1, 1, 0])

    print("--- 1. Single Sample Behaviour ---")
    for p in [0.9, 0.5, 0.1]:
        print(f"y=1, p={p:.2f} -> loss = {single_sample_loss(1, p):.4f}")
    print("A confident wrong prediction is penalised much more heavily than a mild one.")
    assert single_sample_loss(1, 0.9) < single_sample_loss(1, 0.5) < single_sample_loss(1, 0.1)

    print("\n--- 2. Uninformative Model (p = 0.5 everywhere) ---")
    uniform = np.full(len(y), 0.5)
    bce_uniform = binary_cross_entropy(y, uniform)
    print(f"BCE = {bce_uniform:.6f}  (ln 2 = {np.log(2):.6f})")
    assert np.isclose(bce_uniform, np.log(2)), "Guessing 0.5 should give loss ln(2)"

    print("\n--- 3. Good vs Bad Predictions ---")
    good = np.array([0.9, 0.1, 0.8, 0.95, 0.2])
    bad = np.array([0.2, 0.9, 0.3, 0.1, 0.8])
    bce_good = binary_cross_entropy(y, good)
    bce_bad = binary_cross_entropy(y, bad)
    print(f"Good predictions BCE: {bce_good:.4f}")
    print(f"Bad predictions BCE:  {bce_bad:.4f}")
    assert bce_good < bce_uniform < bce_bad, "Ordering of losses is incorrect"

    print("\n--- 4. Verification Against scikit-learn log_loss ---")
    sk_good = log_loss(y, good)
    print(f"Ours: {bce_good:.10f} | sklearn: {sk_good:.10f}")
    assert np.isclose(bce_good, sk_good), "Our BCE disagrees with sklearn.metrics.log_loss"

    print("\n--- 5. Numerical Stability at Probability 0 or 1 ---")
    extreme = np.array([1.0, 0.0])
    worst = binary_cross_entropy(np.array([1, 1]), extreme)
    print(f"Loss with fully wrong certainty (clipped): {worst:.4f}")
    assert np.isfinite(worst), "Loss must stay finite thanks to clipping"

    print("\nFor logistic regression, BCE is convex in the parameters (w, b), so gradient descent can reach the global minimum.")
