# Practice: Supervised vs Unsupervised Learning from Scratch
# Shows the difference between using labeled pairs (X, y) vs unlabeled data (X only)

import numpy as np


# -------------------------------------------------------------
# 1. Supervised Learning: Nearest Centroid Classifier
# Uses known labels y to calculate class centers and classify new points
# -------------------------------------------------------------
class SupervisedNearestCentroid:
    def __init__(self):
        self.centroids = {}

    def fit(self, X, y):
        """Compute the average feature center for each labeled class."""
        unique_classes = np.unique(y)
        for c in unique_classes:
            self.centroids[c] = np.mean(X[y == c], axis=0)

    def predict(self, X):
        """Assign each point to the closest class center by Euclidean distance."""
        preds = []
        for point in X:
            # Distance from point to each class center
            distances = {c: np.linalg.norm(point - center) for c, center in self.centroids.items()}
            closest_class = min(distances, key=distances.get)
            preds.append(closest_class)
        return np.array(preds)


# -------------------------------------------------------------
# 2. Unsupervised Learning: Minimal K-Means Clustering
# Has NO labels; discovers natural groupings from X coordinates alone
# -------------------------------------------------------------
class UnsupervisedKMeans:
    def __init__(self, k=2, max_iters=20):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X):
        """Iteratively partition points into k clusters without any labels."""
        np.random.seed(42)
        # Randomly initialize k centroids from data points
        random_indices = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[random_indices].copy()

        for _ in range(self.max_iters):
            # 1. Assign points to nearest centroid
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            cluster_assignments = np.argmin(distances, axis=1)

            # 2. Recompute centroids as mean of assigned points
            new_centroids = np.array([X[cluster_assignments == i].mean(axis=0) for i in range(self.k)])

            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        return cluster_assignments


if __name__ == "__main__":
    # 2D Synthetic Data: Two natural clusters
    # Cluster A (center ~ [2, 2]), Cluster B (center ~ [8, 8])
    np.random.seed(42)
    cluster_a = np.random.normal(loc=[2.0, 2.0], scale=0.8, size=(10, 2))
    cluster_b = np.random.normal(loc=[8.0, 8.0], scale=0.8, size=(10, 2))

    X = np.vstack([cluster_a, cluster_b])
    # Known ground-truth labels for Supervised Learning (0 = Group A, 1 = Group B)
    y_true = np.array([0] * 10 + [1] * 10)

    # --- Part 1: Supervised Learning (Requires X AND y) ---
    print("--- 1. Supervised Learning (Learns with Labels) ---")
    sup_model = SupervisedNearestCentroid()
    sup_model.fit(X, y_true)
    for c, center in sup_model.centroids.items():
        print(f"Class {c} Center: {np.round(center, 2)}")

    unseen_sample = np.array([[2.5, 2.2], [7.8, 8.2]])
    sup_preds = sup_model.predict(unseen_sample)
    print(f"Predictions for unseen points: {sup_preds} (Expected: [0, 1])")

    # --- Part 2: Unsupervised Learning (Requires ONLY X, no y) ---
    print("\n--- 2. Unsupervised Learning (Discovers Groups without Labels) ---")
    unsup_model = UnsupervisedKMeans(k=2)
    discovered_clusters = unsup_model.fit(X)
    print(f"Discovered Cluster Centroids:\n{np.round(unsup_model.centroids, 2)}")
    print(f"Cluster Assignments:\n{discovered_clusters}")

    assert np.array_equal(sup_preds, [0, 1])
    print("\nBoth paradigms successfully demonstrated!")
