# Practice: Predicted Probabilities & Decision Threshold from Scratch
# Model:         z = X·w + b,   p = σ(z) = P(y = 1 | x)
# Decision rule: ŷ = 1 if p >= t else 0     (t = decision threshold, default 0.5)
# Metrics:       Precision = TP / (TP + FP),  Recall = TP / (TP + FN)

import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def make_dataset(n=200, seed=42):
    """Two features; the true class is drawn from a logistic model (labels are noisy)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 2))
    true_w = np.array([2.0, -1.5])
    true_b = 0.5
    p = sigmoid(X @ true_w + true_b)
    y = (rng.uniform(size=n) < p).astype(int)
    return X, y


def predict_proba(X, w, b):
    """Probability that each sample belongs to class 1."""
    return sigmoid(X @ w + b)


def predict_label(proba, threshold=0.5):
    """Turn probabilities into hard class labels using a decision threshold."""
    return (proba >= threshold).astype(int)


def confusion_counts(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp, tn, fp, fn


def precision_recall(y_true, y_pred):
    tp, _, fp, fn = confusion_counts(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


if __name__ == "__main__":
    X, y = make_dataset()
    # Parameters of an already-trained model (true generating weights, for illustration)
    w = np.array([2.0, -1.5])
    b = 0.5

    proba = predict_proba(X, w, b)

    print("--- 1. Predicted Probabilities ---")
    print(f"First 5 probabilities: {np.round(proba[:5], 4)}")
    print(f"Probability range: [{proba.min():.4f}, {proba.max():.4f}]")
    assert np.all((proba > 0) & (proba < 1)), "Probabilities must lie in (0, 1)"

    print("\n--- 2. Default Threshold t = 0.5 ---")
    y_pred = predict_label(proba, threshold=0.5)
    tp, tn, fp, fn = confusion_counts(y, y_pred)
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Accuracy: {np.mean(y_pred == y):.4f}")

    print("\n--- 3. Effect of Changing the Threshold ---")
    print(f"{'threshold':>10} | {'precision':>9} | {'recall':>7}")
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    recalls = []
    for t in thresholds:
        pred_t = predict_label(proba, threshold=t)
        prec, rec = precision_recall(y, pred_t)
        recalls.append(rec)
        print(f"{t:>10.2f} | {prec:>9.4f} | {rec:>7.4f}")

    # Lower threshold -> more positives predicted -> recall can only stay the same or go down as t rises
    assert all(recalls[i] >= recalls[i + 1] for i in range(len(recalls) - 1)), \
        "Recall should not increase when the threshold increases"

    print("\nLowering the threshold favours recall; raising it favours precision.")
