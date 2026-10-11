# Practice: Complete End-to-End ML Pipeline from Scratch
# Workflow: Raw Data -> Cleaning -> Scaler (No Leakage) -> Train -> Eval -> Inference

import numpy as np


class StandardScalerScratch:
    """Z-score feature scaler: (X - mean) / std."""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1.0  # Prevent division by zero
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class LinearRegressionScratch:
    """Simple multivariate linear regression using Normal Equation."""
    def __init__(self):
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n = len(X)
        X_aug = np.column_stack([np.ones(n), X])
        theta = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y)
        self.b = theta[0]
        self.w = theta[1:]

    def predict(self, X):
        return X @ self.w + self.b


class EndToEndPipeline:
    """Encapsulates Data Preprocessing, Model Training, and Production Inference."""
    def __init__(self):
        self.scaler = StandardScalerScratch()
        self.model = LinearRegressionScratch()

    def train_test_split(self, X, y, test_size=0.2, random_state=42):
        np.random.seed(random_state)
        n = len(X)
        indices = np.random.permutation(n)
        test_count = int(n * test_size)
        test_idx = indices[:test_count]
        train_idx = indices[test_count:]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    def fit(self, X_train, y_train):
        # 1. Fit scaler ONLY on training data (Prevents Data Leakage)
        X_train_scaled = self.scaler.fit_transform(X_train)
        # 2. Train model
        self.model.fit(X_train_scaled, y_train)

    def evaluate(self, X_test, y_test):
        # Transform test data using TRAINING statistics
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)

        # Compute evaluation metrics
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot)

        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

    def predict_new(self, x_raw):
        """End-to-end inference on raw incoming unscaled sample."""
        x_raw_2d = np.atleast_2d(x_raw)
        x_scaled = self.scaler.transform(x_raw_2d)
        return self.model.predict(x_scaled)[0]


if __name__ == "__main__":
    # 1. Raw Data Generation: Car Price Dataset
    # Features: [Engine Size (L), Car Age (Years), Mileage (thousands km)]
    # Target: Price ($) = 15000 * Engine - 1200 * Age - 80 * Mileage + 10000 + noise
    np.random.seed(42)
    n_samples = 300

    engine = np.random.uniform(1.2, 4.0, n_samples)
    age = np.random.uniform(1.0, 15.0, n_samples)
    mileage = np.random.uniform(10.0, 250.0, n_samples)
    X_raw = np.column_stack([engine, age, mileage])

    noise = np.random.normal(0, 1000, n_samples)
    y_raw = 15000 * engine - 1200 * age - 80 * mileage + 10000 + noise

    # 2. Pipeline Initialization
    pipeline = EndToEndPipeline()

    # 3. Data Splitting
    X_train, X_test, y_train, y_test = pipeline.train_test_split(X_raw, y_raw, test_size=0.2)
    print(f"Dataset split: Train={len(X_train)} samples, Test={len(X_test)} samples")

    # 4. Pipeline Training (Scaling + Fitting)
    pipeline.fit(X_train, y_train)
    print(f"Model successfully trained! Learned weights: {np.round(pipeline.model.w, 2)}")

    # 5. Pipeline Evaluation on Unseen Test Data
    metrics = pipeline.evaluate(X_test, y_test)
    print("\n--- Pipeline Evaluation Metrics on Test Set ---")
    for metric_name, val in metrics.items():
        print(f"{metric_name:<6}: {val:.4f}")

    # 6. End-to-End Production Inference on Raw Customer Input
    # Customer brings car: 2.5L engine, 3 years old, 45,000 km
    raw_customer_car = np.array([2.5, 3.0, 45.0])
    predicted_val = pipeline.predict_new(raw_customer_car)
    expected_val = 15000 * 2.5 - 1200 * 3.0 - 80 * 45.0 + 10000  # $40,300

    print(f"\n--- Production Inference on Raw Input ---")
    print(f"Input: Engine={raw_customer_car[0]}L, Age={raw_customer_car[1]}y, Mileage={raw_customer_car[2]}k km")
    print(f"Predicted Price: ${predicted_val:,.2f} (Expected: ~${expected_val:,.2f})")

    assert metrics["R2"] > 0.95
    assert abs(predicted_val - expected_val) < 2000
    print("\nComplete ML Pipeline successfully executed!")
