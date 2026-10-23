# Practice: Logistic Regression From Scratch (NumPy only)
# Pipeline:
#   1. Standardize features:      X_s = (X - μ) / σ   (μ, σ learned on training data only)
#   2. Linear score:               z = X_s·w + b
#   3. Probability:                p = σ(z)
#   4. Loss:                       BCE(y, p)
#   5. Gradient descent:           w ← w - α (1/n) X_s^T (p - y),   b ← b - α mean(p - y)
#   6. Predict:                    ŷ = 1 if p >= threshold else 0

import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true, y_prob, eps=1e-15):
    p = np.clip(y_prob, eps, 1.0 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


class LogisticRegressionScratch:
    """Binary logistic regression trained with batch gradient descent."""

    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = 0.0
        self.mean = None
        self.std = None
        self.loss_history = []

    def _standardize(self, X, fit=False):
        if fit:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0)
            self.std[self.std == 0] = 1.0  # avoid division by zero
        return (X - self.mean) / self.std

    def fit(self, X, y):
        Xs = self._standardize(X, fit=True)
        n, d = Xs.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.loss_history = []

        for _ in range(self.epochs):
            p = sigmoid(Xs @ self.w + self.b)
            error = p - y
            dw = (Xs.T @ error) / n
            db = np.mean(error)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            self.loss_history.append(binary_cross_entropy(y, p))

        return self

    def predict_proba(self, X):
        Xs = self._standardize(X)
        return sigmoid(Xs @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


def make_dataset(n=300, seed=42):
    rng = np.random.default_rng(seed)
    # Second feature is on a different scale on purpose, to show why standardization matters
    X = np.column_stack([rng.normal(0, 1, n), rng.normal(50, 10, n)])
    z = 2.0 * X[:, 0] - 0.3 * (X[:, 1] - 50) + 0.5
    y = (rng.uniform(size=n) < sigmoid(z)).astype(float)
    return X, y


if __name__ == "__main__":
    X, y = make_dataset()

    # Train / test split (first 240 for training, last 60 for testing)
    X_train, X_test = X[:240], X[240:]
    y_train, y_test = y[:240], y[240:]

    print("--- 1. Training ---")
    model = LogisticRegressionScratch(learning_rate=0.5, epochs=1000).fit(X_train, y_train)
    print(f"Initial BCE: {model.loss_history[0]:.6f}")
    print(f"Final BCE:   {model.loss_history[-1]:.6f}")

    print("\n--- 2. Learned Parameters (in standardized feature space) ---")
    print(f"w = {np.round(model.w, 4)}, b = {model.b:.4f}")

    print("\n--- 3. Evaluation ---")
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    test_bce = binary_cross_entropy(y_test, model.predict_proba(X_test))
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print(f"Test BCE:       {test_bce:.4f}")

    print("\n--- 4. Sample Predictions ---")
    probs = model.predict_proba(X_test[:5])
    for i, (p, label) in enumerate(zip(probs, y_test[:5])):
        print(f"Sample {i}: P(y=1)={p:.4f} -> predicted {int(p >= 0.5)}, actual {int(label)}")

    assert model.loss_history[-1] < model.loss_history[0], "Loss should decrease during training"
    assert test_acc > 0.8, "Test accuracy should be above 80%"
    print("\nFrom-scratch logistic regression trained successfully!")
