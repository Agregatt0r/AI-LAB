# Practice: Impact of Feature Scaling on KNN Classification
# Demonstrates how unscaled features with large numerical magnitudes
# drown out informative smaller-scale features, causing KNN accuracy to collapse.

import numpy as np


def pairwise_distances(X_query, X_train):
    """Vectorized Euclidean distance."""
    dists_sq = (
        np.sum(X_query ** 2, axis=1, keepdims=True)
        - 2.0 * np.dot(X_query, X_train.T)
        + np.sum(X_train ** 2, axis=1, keepdims=True).T
    )
    return np.sqrt(np.maximum(dists_sq, 0.0))


def knn_predict(X_train, y_train, X_query, k=5):
    """KNN majority vote prediction."""
    dists = pairwise_distances(X_query, X_train)
    k_indices = np.argsort(dists, axis=1)[:, :k]
    neighbor_labels = y_train[k_indices]

    predictions = []
    for labels in neighbor_labels:
        classes, counts = np.unique(labels, return_counts=True)
        predictions.append(classes[np.argmax(counts)])
    return np.array(predictions)


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


if __name__ == "__main__":
    # Generate Synthetic Dataset: Customer Loan Approval
    # Feature 1: Annual Income in USD ($20,000 to $120,000) -> Large magnitude
    # Feature 2: Credit Rating Score (1.0 to 10.0)         -> Small magnitude
    # True Approval Rule: Approved (1) if Credit Rating >= 6.5 (regardless of income)
    rng = np.random.default_rng(42)
    n = 200

    income = rng.uniform(20000.0, 120000.0, n)
    credit_score = rng.uniform(1.0, 10.0, n)

    # Class 1 if credit_score > 6.0, else 0
    y = (credit_score > 6.0).astype(int)

    X_unscaled = np.column_stack([income, credit_score])

    # Train / Test split
    split = 140
    X_train_raw, X_test_raw = X_unscaled[:split], X_unscaled[split:]
    y_train, y_test = y[:split], y[split:]

    # 1. Evaluate KNN on Raw (Unscaled) Features
    unscaled_preds = knn_predict(X_train_raw, y_train, X_test_raw, k=5)
    unscaled_acc = accuracy_score(y_test, unscaled_preds)

    print("--- 1. Evaluating KNN on Raw (Unscaled) Features ---")
    print(f"Income Range:       [${income.min():,.0f}, ${income.max():,.0f}]")
    print(f"Credit Score Range: [{credit_score.min():.1f}, {credit_score.max():.1f}]")
    print(f"Unscaled KNN Accuracy: {unscaled_acc * 100:.2f}%\n")

    # 2. Evaluate KNN on Manually Standardized Features
    mean = np.mean(X_train_raw, axis=0)
    std = np.std(X_train_raw, axis=0)

    X_train_scaled = (X_train_raw - mean) / std
    X_test_scaled = (X_test_raw - mean) / std

    scaled_preds = knn_predict(X_train_scaled, y_train, X_test_scaled, k=5)
    scaled_acc = accuracy_score(y_test, scaled_preds)

    print("--- 2. Evaluating KNN on Standardized Features ---")
    print(f"Scaled Mean (Train): {np.round(np.mean(X_train_scaled, axis=0), 2).tolist()}")
    print(f"Scaled Std  (Train): {np.round(np.std(X_train_scaled, axis=0), 2).tolist()}")
    print(f"Scaled KNN Accuracy:   {scaled_acc * 100:.2f}%\n")

    # Inspect a single test sample to see why unscaled failed
    query_sample = X_test_raw[0]
    dists_raw = pairwise_distances(query_sample[np.newaxis, :], X_train_raw)[0]
    nearest_raw_idx = np.argsort(dists_raw)[:3]

    print("--- 3. Diagnosis: Why Unscaled Distances Fail ---")
    print(f"Query Point: Income = ${query_sample[0]:,.0f}, Credit Score = {query_sample[1]:.2f}")
    print("3 Closest Neighbors in Unscaled Space:")
    for rank, idx in enumerate(nearest_raw_idx, 1):
        sample = X_train_raw[idx]
        print(f"  #{rank}: Income = ${sample[0]:,.0f}, Credit Score = {sample[1]:.2f} (Label: {y_train[idx]})")
    print("-> Notice the neighbors have almost identical income, but completely random credit scores!")

    # Assertions
    assert scaled_acc >= 0.95, f"Scaled KNN should achieve >95% accuracy, got {scaled_acc}"
    assert unscaled_acc < 0.70, f"Unscaled KNN should suffer low accuracy, got {unscaled_acc}"
    print(f"\nVerification successful: Feature scaling improved accuracy by +{(scaled_acc - unscaled_acc)*100:.1f}%!")

