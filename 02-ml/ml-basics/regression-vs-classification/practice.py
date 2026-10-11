# Practice: Regression vs Classification Side-by-Side from Scratch
# Compares continuous value prediction vs discrete class decision boundary

import numpy as np


# -------------------------------------------------------------------
# 1. Regression Task: Predict Exam Score (0 to 100) based on Study Hours
# -------------------------------------------------------------------
class SimpleRegressor:
    """Fits line ŷ = w * x + b using OLS Normal Equation."""
    def fit(self, x, y):
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        self.w = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        self.b = y_mean - self.w * x_mean

    def predict(self, x):
        return self.w * x + self.b


# -------------------------------------------------------------------
# 2. Classification Task: Predict Pass (1) vs Fail (0) based on Study Hours
# -------------------------------------------------------------------
class SimpleClassifier:
    """Fits a threshold boundary to classify Pass (1) or Fail (0)."""
    def fit(self, x, y):
        # Find threshold that maximizes classification accuracy
        thresholds = np.linspace(x.min(), x.max(), 100)
        best_acc, best_t = -1, 0
        for t in thresholds:
            preds = (x >= t).astype(int)
            acc = np.mean(preds == y)
            if acc > best_acc:
                best_acc, best_t = acc, t
        self.threshold = best_t

    def predict(self, x):
        return (x >= self.threshold).astype(int)


# Metric Functions
def compute_regression_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    return mse, mae


def compute_classification_metrics(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return accuracy, precision, recall, {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


if __name__ == "__main__":
    # Same input feature: Hours spent studying (x)
    study_hours = np.array([1.0, 2.0, 2.5, 3.5, 4.0, 5.0, 6.0, 7.5, 8.0, 9.5])

    # Target 1 (Continuous): Exam Score (0 - 100)
    scores = np.array([35.0, 42.0, 48.0, 60.0, 65.0, 74.0, 80.0, 88.0, 91.0, 98.0])

    # Target 2 (Discrete Class): Passed Exam (1 if score >= 60, else 0)
    passed = (scores >= 60.0).astype(int)

    # --- 1. Run Regression ---
    reg = SimpleRegressor()
    reg.fit(study_hours, scores)
    score_preds = reg.predict(study_hours)
    mse, mae = compute_regression_metrics(scores, score_preds)

    print("--- 1. Regression (Continuous Target: Exam Score) ---")
    print(f"Fitted Equation: Score = {reg.w:.2f} * Hours + {reg.b:.2f}")
    print(f"MSE: {mse:.2f} | MAE: {mae:.2f} points")
    print(f"Prediction for 5 hours of study: {reg.predict(5.0):.1f} points\n")

    # --- 2. Run Classification ---
    clf = SimpleClassifier()
    clf.fit(study_hours, passed)
    pass_preds = clf.predict(study_hours)
    acc, prec, rec, conf = compute_classification_metrics(passed, pass_preds)

    print("--- 2. Classification (Discrete Target: Pass / Fail) ---")
    print(f"Decision Boundary: Study Hours >= {clf.threshold:.2f} hours -> PASS")
    print(f"Accuracy: {acc * 100:.1f}% | Precision: {prec:.2f} | Recall: {rec:.2f}")
    print(f"Confusion Matrix: {conf}")
    print(f"Prediction for 5 hours of study: {'PASS' if clf.predict(5.0) == 1 else 'FAIL'}")

    assert acc == 1.0, "Classification threshold should achieve 100% on linearly separable data"
    print("\nRegression and Classification successfully compared!")
