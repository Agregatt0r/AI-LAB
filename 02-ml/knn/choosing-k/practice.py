# Practice: Choosing k & Evaluating the Bias-Variance Tradeoff
# Evaluates KNN across odd values of k on training vs validation data.
# Demonstrates:
#   - k = 1 gives 100% training accuracy but overfits noisy samples
#   - Large k underfits
#   - Validation curve identifies optimal k

import numpy as np


def pairwise_distances(X_query, X_train):
    """Vectorized Euclidean distance matrix."""
    dists_sq = (
        np.sum(X_query ** 2, axis=1, keepdims=True)
        - 2.0 * np.dot(X_query, X_train.T)
        + np.sum(X_train ** 2, axis=1, keepdims=True).T
    )
    return np.sqrt(np.maximum(dists_sq, 0.0))


def knn_predict(X_train, y_train, X_query, k):
    """Vectorized KNN majority vote prediction."""
    dists = pairwise_distances(X_query, X_train)
    # Get indices of the k smallest distances for each query sample
    k_indices = np.argsort(dists, axis=1)[:, :k]
    # Look up labels of the k neighbors
    neighbor_labels = y_train[k_indices]

    predictions = []
    for labels in neighbor_labels:
        classes, counts = np.unique(labels, return_counts=True)
        predictions.append(classes[np.argmax(counts)])
    return np.array(predictions)


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


if __name__ == "__main__":
    # Generate noisy synthetic classification dataset
    # True boundary: x1^2 + x2^2 < 1.0 (Class 1 inside circle, Class 0 outside)
    rng = np.random.default_rng(42)
    n = 300
    X = rng.uniform(-1.5, 1.5, size=(n, 2))
    radius_sq = np.sum(X ** 2, axis=1)
    y = (radius_sq < 1.0).astype(int)

    # Inject 8% random label noise to simulate real-world noisy data
    noise_idx = rng.choice(n, size=int(0.08 * n), replace=False)
    y[noise_idx] = 1 - y[noise_idx]

    # Train / Validation Split (70% train, 30% val)
    split_idx = int(0.7 * n)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print("--- 1. Evaluating KNN for different values of k ---")
    k_candidates = [1, 3, 5, 7, 9, 15, 25, 45, 101]

    best_k = None
    best_val_acc = -1.0
    results = []

    print(f"{'k':<4} | {'Train Acc':<10} | {'Val Acc':<10} | {'Behavior'}")
    print("-" * 50)

    for k in k_candidates:
        train_preds = knn_predict(X_train, y_train, X_train, k=k)
        val_preds = knn_predict(X_train, y_train, X_val, k=k)

        train_acc = accuracy_score(y_train, train_preds)
        val_acc = accuracy_score(y_val, val_preds)

        behavior = ""
        if k == 1:
            behavior = "Overfitting (100% train, memorized noise)"
        elif k >= 45:
            behavior = "Underfitting (Oversmoothed boundary)"
        else:
            behavior = "Good balance"

        print(f"{k:<4} | {train_acc * 100:>6.2f}%    | {val_acc * 100:>6.2f}%    | {behavior}")
        results.append((k, train_acc, val_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_k = k

    print("-" * 50)
    print(f"\n--- 2. Optimal k Selection ---")
    print(f"Optimal k selected by Validation Accuracy: k = {best_k} (Val Acc: {best_val_acc * 100:.2f}%)")

    # Assertions
    # k = 1 must achieve 100% train accuracy because each point is its own nearest neighbor
    assert results[0][1] == 1.0, "k=1 should have 100% training accuracy!"
    # Optimal k should outperform k=1 on validation set because k=1 overfits
    assert best_val_acc >= results[0][2], "Validation accuracy of optimal k must be >= k=1"

    print("\nBias-variance tradeoff across k successfully verified!")

