# Practice: Rule-Based Logic vs Learning from Data
# Demonstrates how Machine Learning automatically discovers a rule from data
# instead of requiring a programmer to manually hardcode conditions.

import numpy as np


# 1. Classical Programming: Programmer writes hardcoded rules
def rule_based_predict(temperature_celsius):
    """Hardcoded programmer rules based on intuition."""
    if temperature_celsius < 10:
        return "Cold"
    elif temperature_celsius < 22:
        return "Mild"
    else:
        return "Hot"


# 2. Machine Learning: Algorithm discovers boundary thresholds from data
class SimpleThresholdLearner:
    """Learns an optimal decision threshold from (X, y) data pairs."""
    def __init__(self):
        self.threshold = None

    def fit(self, X, y):
        """Find the threshold value that maximizes classification accuracy."""
        best_acc = -1.0
        best_thresh = 0.0

        # Search across candidate thresholds
        candidates = np.linspace(X.min(), X.max(), 50)
        for thresh in candidates:
            # Predict 1 if X >= thresh, else 0
            preds = (X >= thresh).astype(int)
            acc = np.mean(preds == y)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh

        self.threshold = best_thresh
        return best_acc

    def predict(self, X):
        return (X >= self.threshold).astype(int)


if __name__ == "__main__":
    # Task: Learn whether a day is "Comfortable for Running" (1 = Yes, 0 = No)
    # Experience (Data): temperatures and runner satisfaction labels
    np.random.seed(42)
    temperatures = np.array([5.0, 8.0, 12.0, 15.0, 18.0, 20.0, 24.0, 28.0, 32.0, 35.0])
    # Ground truth: runner likes temperatures 15°C and above
    labels = (temperatures >= 15.0).astype(int)

    print("--- 1. Data Provided to ML System ---")
    for t, label in zip(temperatures, labels):
        print(f"Temperature: {t:4.1f}°C -> Comfortable: {'Yes' if label else 'No'}")

    # Machine Learning fit phase
    learner = SimpleThresholdLearner()
    training_accuracy = learner.fit(temperatures, labels)

    print("\n--- 2. Learned Rule from Data ---")
    print(f"Learned Decision Threshold: {learner.threshold:.1f}°C")
    print(f"Training Accuracy:           {training_accuracy * 100:.1f}%")

    # Unseen test data
    unseen_temperatures = np.array([7.0, 16.5, 23.0])
    test_preds = learner.predict(unseen_temperatures)

    print("\n--- 3. Generalization to Unseen Temperatures ---")
    for t, pred in zip(unseen_temperatures, test_preds):
        verdict = "Comfortable" if pred == 1 else "Too Cold"
        print(f"Test Temp: {t:4.1f}°C -> Prediction: {verdict}")

    assert test_preds[0] == 0 and test_preds[1] == 1 and test_preds[2] == 1
    print("\nPattern successfully learned directly from data!")
