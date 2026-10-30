# Practice: KNN for Classification and Regression Side-by-Side
# Compares:
#   1. KNN Classifier: majority voting, class probabilities, accuracy
#   2. KNN Regressor: uniform vs distance-weighted averaging, MSE, R^2

import numpy as np


def pairwise_distances(X_query, X_train):
    """Vectorized Euclidean distance matrix."""
    dists_sq = (
        np.sum(X_query ** 2, axis=1, keepdims=True)
        - 2.0 * np.dot(X_query, X_train.T)
        + np.sum(X_train ** 2, axis=1, keepdims=True).T
    )
    return np.sqrt(np.maximum(dists_sq, 0.0))


# -------------------------------------------------------------
# 1. KNN Classification Functions
# -------------------------------------------------------------
def knn_classify(X_train, y_train, X_query, k=5):
    """Predict class labels and class probabilities using majority vote."""
    dists = pairwise_distances(X_query, X_train)
    k_indices = np.argsort(dists, axis=1)[:, :k]
    neighbor_labels = y_train[k_indices]

    classes = np.unique(y_train)
    predictions = []
    probabilities = []

    for labels in neighbor_labels:
        # Class probabilities = fraction of k neighbors in each class
        counts = {c: np.sum(labels == c) for c in classes}
        probs = [counts[c] / k for c in classes]
        pred_class = classes[np.argmax(probs)]

        predictions.append(pred_class)
        probabilities.append(probs)

    return np.array(predictions), np.array(probabilities)


# -------------------------------------------------------------
# 2. KNN Regression Functions
# -------------------------------------------------------------
def knn_regress(X_train, y_train, X_query, k=5, weights="uniform"):
    """Predict continuous targets via uniform or distance-weighted averaging."""
    dists = pairwise_distances(X_query, X_train)
    k_indices = np.argsort(dists, axis=1)[:, :k]

    predictions = []
    for i, indices in enumerate(k_indices):
        neighbor_y = y_train[indices]
        if weights == "uniform":
            pred = np.mean(neighbor_y)
        elif weights == "distance":
            neighbor_dists = dists[i, indices]
            # Inverse distance weights with epsilon to avoid division by zero
            w = 1.0 / (neighbor_dists + 1e-5)
            pred = np.sum(w * neighbor_y) / np.sum(w)
        else:
            raise ValueError(f"Unknown weights: {weights}")
        predictions.append(pred)

    return np.array(predictions)


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Task 1: Classification (Predict Pass/Fail on Exam)
    # ---------------------------------------------------------
    print("--- 1. KNN Classification (Exam Pass / Fail) ---")
    rng = np.random.default_rng(42)
    hours = rng.uniform(1.0, 10.0, 100)
    # Target: 1 if hours > 5.0, with minor noise
    y_clf = (hours > 5.0).astype(int)

    X_clf_train, X_clf_test = hours[:70, np.newaxis], hours[70:, np.newaxis]
    y_clf_train, y_clf_test = y_clf[:70], y_clf[70:]

    clf_preds, clf_probs = knn_classify(X_clf_train, y_clf_train, X_clf_test, k=5)
    acc = np.mean(y_clf_test == clf_preds)

    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("Sample Probabilities [P(Fail), P(Pass)] for first 3 test students:")
    for h, p, label in zip(X_clf_test[:3, 0], clf_probs[:3], clf_preds[:3]):
        print(f"  Hours: {h:4.1f} -> Probabilities: {np.round(p, 2).tolist()} -> Predicted: {'PASS' if label==1 else 'FAIL'}")

    assert acc > 0.90, "Classifier accuracy should exceed 90%"

    # ---------------------------------------------------------
    # Task 2: Regression (Predict Actual Score: 0 - 100)
    # ---------------------------------------------------------
    print("\n--- 2. KNN Regression (Predicting Continuous Exam Score) ---")
    # True relationship: score = 8.0 * hours + 20.0 + noise
    scores = 8.0 * hours + 20.0 + rng.normal(0, 3.0, 100)
    X_reg_train, X_reg_test = hours[:70, np.newaxis], hours[70:, np.newaxis]
    y_reg_train, y_reg_test = scores[:70], scores[70:]

    # Compare Uniform vs Distance Weighting
    reg_preds_uniform = knn_regress(X_reg_train, y_reg_train, X_reg_test, k=5, weights="uniform")
    reg_preds_dist = knn_regress(X_reg_train, y_reg_train, X_reg_test, k=5, weights="distance")

    def evaluate_reg(y_true, y_pred, name):
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot)
        print(f"[{name}] RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")
        return r2

    r2_uni = evaluate_reg(y_reg_test, reg_preds_uniform, "Uniform Weights")
    r2_dst = evaluate_reg(y_reg_test, reg_preds_dist, "Distance Weights")

    # Sample predictions
    print("\nSample Score Predictions (Distance Weighted):")
    for h, true_s, pred_s in zip(X_reg_test[:3, 0], y_reg_test[:3], reg_preds_dist[:3]):
        print(f"  Hours: {h:4.1f} -> True Score: {true_s:5.1f} | Predicted: {pred_s:5.1f}")

    assert r2_uni > 0.85 and r2_dst > 0.85, "Regression R² should exceed 0.85"
    print("\nClassification and Regression tasks successfully compared!")

