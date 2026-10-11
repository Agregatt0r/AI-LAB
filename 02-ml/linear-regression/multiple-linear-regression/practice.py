# Practice: Multiple Linear Regression with Feature Standardization
# Model: ŷ = X @ w + b
# Gradients:
#   dw = (2 / n) * X.T @ (ŷ - y)
#   db = (2 / n) * np.sum(ŷ - y)

import numpy as np


class MultipleLinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = 0.0
        self.mean = None
        self.std = None

    def fit_standardizer(self, X):
        """Compute mean and std for feature scaling."""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Avoid division by zero if a feature is constant
        self.std[self.std == 0] = 1.0

    def transform(self, X):
        """Apply Z-score standardization (X - mean) / std."""
        return (X - self.mean) / self.std

    def fit(self, X, y):
        """Fit multiple linear regression model using gradient descent."""
        n, d = X.shape
        self.fit_standardizer(X)
        X_scaled = self.transform(X)

        self.w = np.zeros(d)
        self.b = 0.0

        for epoch in range(self.epochs):
            y_pred = X_scaled @ self.w + self.b
            errors = y_pred - y

            dw = (2.0 / n) * (X_scaled.T @ errors)
            db = (2.0 / n) * np.sum(errors)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        """Predict target values for new data."""
        X_scaled = self.transform(X)
        return X_scaled @ self.w + self.b


if __name__ == "__main__":
    # Synthetic Multi-Feature Dataset:
    # Feature 1: House Size (m^2) [40 - 120]
    # Feature 2: Number of Bedrooms [1 - 4]
    # Feature 3: Age of House (years) [1 - 20]
    # Ground Truth: Price = 3000 * Size + 15000 * Beds - 2000 * Age + 50000
    np.random.seed(42)
    n_samples = 200

    size = np.random.uniform(40, 120, n_samples)
    beds = np.random.randint(1, 5, n_samples)
    age = np.random.uniform(1, 20, n_samples)

    X = np.column_stack([size, beds, age])
    noise = np.random.normal(0, 5000, n_samples)
    y = 3000 * size + 15000 * beds - 2000 * age + 50000 + noise

    # Fit Model
    model = MultipleLinearRegression(learning_rate=0.05, epochs=1500)
    model.fit(X, y)

    y_pred = model.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)

    print("--- Multiple Linear Regression Results ---")
    print(f"Dataset Shape:       X={X.shape}, y={y.shape}")
    print(f"Learned Scaled w:    {np.round(model.w, 2)}")
    print(f"Learned Bias b:      {model.b:.2f}")
    print(f"Training RMSE:       ${rmse:.2f} (Noise std was $5000)")

    # Test prediction on a new house: 85 m^2, 3 beds, 5 years old
    test_house = np.array([[85.0, 3.0, 5.0]])
    expected_approx = 3000 * 85 + 15000 * 3 - 2000 * 5 + 50000  # 340,000
    predicted_price = model.predict(test_house)[0]

    print(f"\nNew House Prediction (85m², 3 beds, 5 yrs old):")
    print(f"Expected True Price: ~${expected_approx:,.2f}")
    print(f"Model Prediction:     ${predicted_price:,.2f}")

    assert abs(predicted_price - expected_approx) < 15000
    print("\nMultiple Linear Regression working accurately!")
