# Practice: Training vs Inference Lifecycle from Scratch
# Step 1: Train a model on training data (heavy optimization)
# Step 2: Export / Serialize learned parameters
# Step 3: Run ultra-fast production inference on new stream data

import time
import json
import numpy as np


# -------------------------------------------------------------
# Stage 1: Training System (Offline Training Job)
# -------------------------------------------------------------
class ModelTrainer:
    def __init__(self, learning_rate=0.05, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self, X, y):
        """Train weights w and bias b via Gradient Descent."""
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0

        for _ in range(self.epochs):
            y_pred = X @ w + b
            errors = y_pred - y
            dw = (2.0 / n) * (X.T @ errors)
            db = (2.0 / n) * np.sum(errors)

            w -= self.learning_rate * dw
            b -= self.learning_rate * db

        # Return serializable model artifact (state dict)
        artifact = {
            "model_type": "LinearRegression",
            "weights": w.tolist(),
            "bias": float(b),
            "trained_at_samples": n,
        }
        return artifact


# -------------------------------------------------------------
# Stage 2: Inference Engine (Production Microservice / Real-Time Serving)
# Notice: No training loops, no gradients, no loss functions.
# -------------------------------------------------------------
class ProductionInferenceEngine:
    def __init__(self, model_artifact_json):
        artifact = json.loads(model_artifact_json)
        self.w = np.array(artifact["weights"])
        self.b = artifact["bias"]

    def predict(self, x_new):
        """Ultra-fast forward pass prediction: ŷ = x · w + b."""
        return np.dot(x_new, self.w) + self.b


if __name__ == "__main__":
    # Generate Training Data (e.g. Server response time vs CPU usage and RAM usage)
    # y = 2.0 * cpu + 0.5 * ram + 10.0 + noise
    np.random.seed(42)
    n_train = 5000
    cpu = np.random.uniform(10, 95, n_train)
    ram = np.random.uniform(20, 90, n_train)
    X_train = np.column_stack([cpu, ram])
    y_train = 2.0 * cpu + 0.5 * ram + 10.0 + np.random.normal(0, 1.0, n_train)

    # --- 1. Training Phase ---
    print("--- 1. Training Phase (Offline / Optimization) ---")
    start_train = time.perf_counter()
    trainer = ModelTrainer(learning_rate=0.0001, epochs=1500)
    model_artifact = trainer.train(X_train, y_train)
    train_duration = time.perf_counter() - start_train

    print(f"Training completed in: {train_duration * 1000:.2f} ms")
    print(f"Model Artifact to Export: {model_artifact}\n")

    # Serialize to JSON (simulating saving to disk / cloud model registry)
    serialized_model = json.dumps(model_artifact)

    # --- 2. Production Inference Phase ---
    print("--- 2. Production Inference Phase (Real-Time Serving) ---")
    engine = ProductionInferenceEngine(serialized_model)

    # Simulate real-time streaming production request:
    # Incoming user request: CPU=75%, RAM=60%
    incoming_request = np.array([75.0, 60.0])

    # Measure inference latency
    start_infer = time.perf_counter()
    predicted_response_time = engine.predict(incoming_request)
    infer_latency_us = (time.perf_counter() - start_infer) * 1_000_000  # microseconds

    expected = 2.0 * 75.0 + 0.5 * 60.0 + 10.0  # 190.0 ms

    print(f"Input: CPU={incoming_request[0]}%, RAM={incoming_request[1]}%")
    print(f"Predicted Response Time: {predicted_response_time:.2f} ms (Expected: ~{expected:.1f} ms)")
    print(f"Inference Latency:       {infer_latency_us:.2f} µs ({infer_latency_us / 1000:.4f} ms)")

    assert abs(predicted_response_time - expected) < 2.0
    print("\nTraining and Inference stages successfully separated and benchmarked!")
