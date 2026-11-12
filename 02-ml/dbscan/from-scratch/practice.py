"""DBSCAN from Scratch in Pure NumPy.

This module provides an object-oriented, first-principles implementation of the
DBSCAN algorithm and benchmarks it on diverse topologies:
1. Multi-cluster Gaussian Blobs with Noise.
2. Non-convex Intertwining Crescent Moons.
3. Nested Concentric Circles.

Zero hard external dependencies: operates completely in pure Python and NumPy.
If scikit-learn is installed, it runs a side-by-side exact parity verification.
"""

from collections import deque
from typing import List, Optional, Tuple
import numpy as np

# Optional scikit-learn import for comparative validation
try:
    from sklearn.cluster import DBSCAN as SklearnDBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# =====================================================================
# 1. Self-Contained Metric: Adjusted Rand Index in Pure NumPy
# =====================================================================
def adjusted_rand_score_scratch(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Compute the Adjusted Rand Index (ARI) from first principles in pure NumPy."""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    n = len(labels_true)
    if n <= 1:
        return 1.0

    # Map arbitrary label sets to contiguous indices 0..K-1
    _, true_idx = np.unique(labels_true, return_inverse=True)
    _, pred_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = int(np.max(true_idx)) + 1
    n_clusters = int(np.max(pred_idx)) + 1

    # Contingency matrix
    contingency = np.zeros((n_classes, n_clusters), dtype=np.int64)
    for t, p in zip(true_idx, pred_idx):
        contingency[t, p] += 1

    # Binomial coefficient n_choose_2 = n * (n - 1) / 2
    sum_comb_c = int(np.sum(contingency * (contingency - 1)) // 2)
    a = np.sum(contingency, axis=1)
    b = np.sum(contingency, axis=0)
    sum_comb_a = int(np.sum(a * (a - 1)) // 2)
    sum_comb_b = int(np.sum(b * (b - 1)) // 2)
    comb_total = (n * (n - 1)) // 2

    expected_index = (sum_comb_a * sum_comb_b) / comb_total
    max_index = (sum_comb_a + sum_comb_b) / 2.0
    denominator = max_index - expected_index

    if denominator == 0:
        return 1.0
    return float((sum_comb_c - expected_index) / denominator)


# =====================================================================
# 2. Self-Contained Synthetic Dataset Generators in Pure NumPy
# =====================================================================
def make_blobs_scratch(
    n_samples: int = 300,
    centers: int = 3,
    cluster_std: float = 0.6,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Gaussian blobs using pure NumPy."""
    rng = np.random.RandomState(random_state)
    center_coords = np.array([[-4.0, -4.0], [4.0, 4.0], [-4.0, 4.0]])[:centers]
    n_per = n_samples // centers
    X_parts = []
    y_parts = []
    for i, c in enumerate(center_coords):
        X_parts.append(rng.normal(loc=c, scale=cluster_std, size=(n_per, 2)))
        y_parts.extend([i] * n_per)
    return np.vstack(X_parts), np.array(y_parts)


def make_moons_scratch(
    n_samples: int = 300,
    noise: float = 0.08,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate intertwining crescent moons using pure NumPy."""
    rng = np.random.RandomState(random_state)
    n_half = n_samples // 2
    theta1 = np.linspace(0, np.pi, n_half)
    moon1 = np.column_stack([np.cos(theta1), np.sin(theta1)]) + rng.normal(0, noise, size=(n_half, 2))
    theta2 = np.linspace(0, np.pi, n_half)
    moon2 = np.column_stack([1.0 - np.cos(theta2), 1.0 - np.sin(theta2) - 0.5]) + rng.normal(0, noise, size=(n_half, 2))
    X = np.vstack([moon1, moon2])
    y = np.array([0] * n_half + [1] * n_half)
    return X, y


def make_circles_scratch(
    n_samples: int = 300,
    factor: float = 0.4,
    noise: float = 0.06,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate nested concentric circles using pure NumPy."""
    rng = np.random.RandomState(random_state)
    n_half = n_samples // 2
    theta1 = rng.uniform(0, 2 * np.pi, n_half)
    c_out = np.column_stack([np.cos(theta1), np.sin(theta1)]) + rng.normal(0, noise, size=(n_half, 2))
    theta2 = rng.uniform(0, 2 * np.pi, n_half)
    c_in = factor * np.column_stack([np.cos(theta2), np.sin(theta2)]) + rng.normal(0, noise, size=(n_half, 2))
    X = np.vstack([c_out, c_in])
    y = np.array([0] * n_half + [1] * n_half)
    return X, y


# =====================================================================
# 3. Object-Oriented DBSCAN from Scratch
# =====================================================================
class DBSCANScratch:
    """Density-Based Spatial Clustering of Applications with Noise (from scratch).

    Parameters:
        eps: The maximum distance between two samples for one to be considered
             as in the neighborhood of the other.
        min_samples: The number of samples in a neighborhood for a point to be
                     considered as a core point (including the point itself).
        metric: Distance metric ('euclidean').
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5, metric: str = "euclidean") -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

        # Fitted attributes matching scikit-learn conventions
        self.labels_: np.ndarray = np.array([])
        self.core_sample_indices_: np.ndarray = np.array([], dtype=int)
        self.components_: np.ndarray = np.array([])
        self.n_clusters_: int = 0
        self.n_noise_: int = 0

    def _compute_pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance matrix using NumPy vectorization."""
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=-1))

    def fit(self, X: np.ndarray) -> "DBSCANScratch":
        """Perform DBSCAN clustering from features.

        Args:
            X: Array-like of shape (n_samples, n_features).

        Returns:
            self: Fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        n_samples = len(X)

        # 1. Compute pairwise distance matrix
        dist_matrix = self._compute_pairwise_distances(X)

        # 2. Extract epsilon-neighborhoods: N_eps(i) = { j | dist(i, j) <= eps }
        neighborhoods: List[np.ndarray] = [
            np.where(dist_matrix[i] <= self.eps)[0] for i in range(n_samples)
        ]

        # 3. Identify Core Points: |N_eps(i)| >= min_samples
        is_core = np.array([len(nb) >= self.min_samples for nb in neighborhoods], dtype=bool)
        self.core_sample_indices_ = np.where(is_core)[0]
        self.components_ = X[self.core_sample_indices_] if len(self.core_sample_indices_) > 0 else np.empty((0, X.shape[1]))

        # 4. Initialize labels (-1 for noise) and visited tracking
        labels = np.full(n_samples, -1, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0

        # 5. BFS Cluster Expansion
        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True

            # Only core points can initiate a new cluster
            if not is_core[i]:
                continue

            labels[i] = cluster_id

            # BFS queue for density-reachable points
            queue = deque(neighborhoods[i])
            in_queue = set(neighborhoods[i])

            while queue:
                q = queue.popleft()

                if not visited[q]:
                    visited[q] = True
                    # If neighbor q is also a core point, it expands the cluster boundary
                    if is_core[q]:
                        for neighbor_of_q in neighborhoods[q]:
                            if neighbor_of_q not in in_queue:
                                in_queue.add(neighbor_of_q)
                                queue.append(neighbor_of_q)

                # If neighbor q was unassigned or marked as noise, adopt it into the current cluster
                if labels[q] == -1:
                    labels[q] = cluster_id

            cluster_id += 1

        self.labels_ = labels
        self.n_clusters_ = cluster_id
        self.n_noise_ = int(np.sum(labels == -1))
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit the model and return cluster labels."""
        self.fit(X)
        return self.labels_


# =====================================================================
# 4. Verification Benchmark Runner
# =====================================================================
def run_benchmark_test(
    name: str,
    X: np.ndarray,
    y_true: np.ndarray,
    eps: float,
    min_samples: int,
) -> Tuple[int, int, float, Optional[float], bool]:
    """Run DBSCANScratch alongside ground truth and scikit-learn (if available)."""
    # 1. Scratch implementation
    scratch = DBSCANScratch(eps=eps, min_samples=min_samples)
    scratch_labels = scratch.fit_predict(X)

    # Metrics against ground truth
    ari_vs_true = adjusted_rand_score_scratch(y_true, scratch_labels)

    # 2. Scikit-learn reference (if available)
    ari_vs_sklearn = None
    exact_cores_match = False

    if HAS_SKLEARN:
        sklearn_db = SklearnDBSCAN(eps=eps, min_samples=min_samples)
        sklearn_labels = sklearn_db.fit_predict(X)
        exact_cores_match = np.array_equal(scratch.core_sample_indices_, sklearn_db.core_sample_indices_)
        ari_vs_sklearn = adjusted_rand_score_scratch(scratch_labels, sklearn_labels)

    return scratch.n_clusters_, scratch.n_noise_, ari_vs_true, ari_vs_sklearn, exact_cores_match


def main() -> None:
    print("=" * 70)
    print("7. DBSCAN FROM SCRATCH: NUMPY IMPLEMENTATION")
    print("=" * 70)

    # Generate 3 distinct benchmark topologies
    rng = np.random.RandomState(42)

    # 1. Gaussian Blobs with Noise
    X_blobs, y_blobs = make_blobs_scratch(n_samples=300, centers=3, cluster_std=0.6, random_state=42)
    noise_blobs = rng.uniform(low=-10, high=10, size=(30, 2))
    X_blobs = np.vstack([X_blobs, noise_blobs])
    y_blobs = np.concatenate([y_blobs, np.full(30, -1)])

    # 2. Interlocking Crescent Moons
    X_moons, y_moons = make_moons_scratch(n_samples=300, noise=0.08, random_state=42)

    # 3. Concentric Circles
    X_circles, y_circles = make_circles_scratch(n_samples=300, factor=0.4, noise=0.06, random_state=42)

    benchmarks = [
        ("Gaussian Blobs + Noise", X_blobs, y_blobs, 1.0, 5),
        ("Intertwining Moons", X_moons, y_moons, 0.20, 5),
        ("Concentric Circles", X_circles, y_circles, 0.22, 5),
    ]

    print(f"{'Topology':<24} | {'Clusters':<9} | {'Noise':<7} | {'ARI (Ground Truth)':<18} | {'ARI (vs Sklearn)':<16}")
    print("-" * 84)

    for name, X_data, y_truth, eps_val, ms_val in benchmarks:
        n_clusters, n_noise, ari_true, ari_sk, core_match = run_benchmark_test(
            name, X_data, y_truth, eps=eps_val, min_samples=ms_val
        )
        sk_str = f"{ari_sk:.4f} (EXACT)" if ari_sk is not None else "N/A (no sklearn)"
        print(f"{name:<24} | {n_clusters:<9} | {n_noise:<7} | {ari_true:<18.4f} | {sk_str:<16}")

        # Verification Assertions
        assert ari_true >= 0.95, f"ARI vs ground truth must be >= 0.95, got {ari_true}"
        if HAS_SKLEARN:
            assert ari_sk == 1.0, f"ARI between scratch and sklearn must be 1.0, got {ari_sk}"
            assert core_match, "Core sample indices must be 100% identical to scikit-learn!"

    print("-" * 84)
    print("\nBenchmark Takeaways:")
    print("  1. Pure NumPy self-containment: runs seamlessly without any third-party ML library.")
    print("  2. Discovers complex non-convex cluster topologies (moons, rings) with ARI > 0.95.")
    print("  3. Identifies and isolates background noise points without polluting cluster cores.")
    if HAS_SKLEARN:
        print("  4. 100% mathematical parity with scikit-learn's C-accelerated DBSCAN implementation.")

    print("\n✓ DBSCANScratch validated successfully from first principles!")


if __name__ == "__main__":
    main()
