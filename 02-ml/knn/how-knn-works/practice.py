# Practice: How KNN Works (Step-by-Step Neighbor Search)
# 1. Store training samples
# 2. Compute Euclidean distance from query point to all stored points
# 3. Sort distances and pick top-k indices
# 4. Predict by majority vote

import numpy as np


def euclidean_distance(a, b):
    """Compute Euclidean distance between two 1D vectors."""
    return np.sqrt(np.sum((a - b) ** 2))


def knn_step_by_step(X_train, y_train, x_query, k=3):
    """Perform KNN classification with step-by-step diagnostic prints."""
    distances = []

    # Step 1 & 2: Calculate distance to each training point
    for i in range(len(X_train)):
        d = euclidean_distance(x_query, X_train[i])
        distances.append((d, y_train[i], i))

    # Step 3: Sort by distance
    distances.sort(key=lambda item: item[0])

    # Select the k closest neighbors
    k_nearest = distances[:k]

    # Step 4: Majority vote
    neighbor_labels = [label for _, label, _ in k_nearest]
    classes, counts = np.unique(neighbor_labels, return_counts=True)
    predicted_class = classes[np.argmax(counts)]

    return predicted_class, k_nearest


if __name__ == "__main__":
    # 2D Training Dataset: [Feature 1, Feature 2]
    # Class 0: Blue (circles), Class 1: Orange (triangles)
    X_train = np.array([
        [1.0, 2.0],  # 0
        [1.5, 1.8],  # 1
        [2.0, 2.5],  # 2
        [5.0, 5.0],  # 3
        [5.5, 6.0],  # 4
        [6.0, 5.5],  # 5
    ])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    # Query point close to Class 0 cluster
    query_point = np.array([1.8, 2.0])

    print("--- 1. Training Set Stored in Memory ---")
    for i, (point, label) in enumerate(zip(X_train, y_train)):
        print(f"Sample #{i}: x = {point.tolist()} -> Class {label}")

    print(f"\nQuery Point: {query_point.tolist()}")

    # Run KNN with k = 3
    k = 3
    pred, neighbors = knn_step_by_step(X_train, y_train, query_point, k=k)

    print(f"\n--- 2. Finding {k} Nearest Neighbors ---")
    for rank, (dist, label, idx) in enumerate(neighbors, 1):
        print(f"Rank #{rank}: Sample #{idx} (x={X_train[idx].tolist()}), dist={dist:.4f}, class={label}")

    print(f"\n--- 3. Prediction ---")
    print(f"Neighbors classes: {[lbl for _, lbl, _ in neighbors]}")
    print(f"Predicted Class:   {pred} (Expected: 0)")

    assert pred == 0, f"Expected class 0, got {pred}"
    assert len(neighbors) == k, f"Expected {k} neighbors"

    # Query point close to Class 1 cluster
    query_point_2 = np.array([5.2, 5.8])
    pred_2, neighbors_2 = knn_step_by_step(X_train, y_train, query_point_2, k=k)
    print(f"\nSecond Query: {query_point_2.tolist()} -> Predicted Class: {pred_2} (Expected: 1)")
    assert pred_2 == 1, f"Expected class 1, got {pred_2}"

    print("\nKNN neighborhood search successfully verified!")

