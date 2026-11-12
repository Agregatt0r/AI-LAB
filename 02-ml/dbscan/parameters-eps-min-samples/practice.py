"""Hyperparameter Selection in DBSCAN: eps and min_samples.

This module implements:
1. Vectorized k-distance computation for density estimation.
2. Geometric elbow/knee detection algorithm using maximum perpendicular distance.
3. Parameter sensitivity sweep demonstrating:
   - eps too small -> cluster shattering / excessive noise
   - eps optimal -> correct discovery of natural clusters
   - eps too large -> cluster merging / noise absorption
"""

from typing import Tuple
import numpy as np

try:
    from sklearn.cluster import DBSCAN as SklearnDBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def dbscan_predict(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Pure NumPy DBSCAN prediction."""
    if HAS_SKLEARN:
        return SklearnDBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)

    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    n_samples = len(X)
    neighborhoods = [np.where(dist[i] <= eps)[0] for i in range(n_samples)]
    is_core = np.array([len(nb) >= min_samples for nb in neighborhoods], dtype=bool)

    labels = np.full(n_samples, -1, dtype=int)
    visited = np.zeros(n_samples, dtype=bool)
    cluster_id = 0

    for i in range(n_samples):
        if visited[i] or not is_core[i]:
            continue
        visited[i] = True
        labels[i] = cluster_id
        queue = list(neighborhoods[i])
        idx = 0
        while idx < len(queue):
            q = queue[idx]
            idx += 1
            if not visited[q]:
                visited[q] = True
                if is_core[q]:
                    for nb in neighborhoods[q]:
                        if nb not in queue:
                            queue.append(nb)
            if labels[q] == -1:
                labels[q] = cluster_id
        cluster_id += 1

    return labels


def compute_pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Compute NxN Euclidean distance matrix."""
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def compute_k_distances(X: np.ndarray, k: int) -> np.ndarray:
    """Compute distance to k-th nearest neighbor for all points, sorted ascending.

    Args:
        X: Dataset of shape (N, d).
        k: Neighbor rank (typically min_samples - 1).

    Returns:
        sorted_k_dist: 1D array of shape (N,) with k-distances sorted in ascending order.
    """
    dist_matrix = compute_pairwise_distances(X)
    # Sort each row: index 0 is distance to self (0.0), index k is distance to k-th neighbor
    sorted_row_dists = np.sort(dist_matrix, axis=1)
    k_dists = sorted_row_dists[:, k]
    return np.sort(k_dists)


def find_elbow_point(sorted_k_dist: np.ndarray) -> Tuple[int, float]:
    """Find the knee/elbow of maximum curvature in the sorted k-distance curve.

    Uses maximum perpendicular distance from the line connecting (0, y0) to (N-1, y_end).

    Returns:
        knee_idx: Index of maximum curvature.
        optimal_eps: Recommended epsilon value.
    """
    n_points = len(sorted_k_dist)
    p1 = np.array([0, sorted_k_dist[0]])
    p2 = np.array([n_points - 1, sorted_k_dist[-1]])

    # Vector along the baseline
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    unit_line = line_vec / line_len

    # Compute perpendicular distance for each point
    distances = []
    for i in range(n_points):
        p = np.array([i, sorted_k_dist[i]])
        vec_to_point = p - p1
        # Projection length onto line
        proj_len = np.dot(vec_to_point, unit_line)
        # Vector perpendicular from line to point
        perp_vec = vec_to_point - proj_len * unit_line
        distances.append(np.linalg.norm(perp_vec))

    knee_idx = int(np.argmax(distances))
    optimal_eps = float(sorted_k_dist[knee_idx])
    return knee_idx, optimal_eps


def generate_two_blobs_with_noise(
    n_per_blob: int = 100, n_noise: int = 25, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 2 Gaussian clusters plus background noise."""
    rng = np.random.RandomState(random_state)
    blob1 = rng.normal(loc=[-2.5, 0.0], scale=0.4, size=(n_per_blob, 2))
    blob2 = rng.normal(loc=[2.5, 0.0], scale=0.4, size=(n_per_blob, 2))
    noise = rng.uniform(low=[-5.0, -3.0], high=[5.0, 3.0], size=(n_noise, 2))

    X = np.vstack([blob1, blob2, noise])
    labels = np.array([0] * n_per_blob + [1] * n_per_blob + [-1] * n_noise)
    return X, labels


def main() -> None:
    print("=" * 70)
    print("3. HYPERPARAMETERS: eps AND min_samples SELECTION")
    print("=" * 70)

    # 1. Dataset setup
    X, true_labels = generate_two_blobs_with_noise(n_per_blob=100, n_noise=25, random_state=42)
    n_samples, n_features = X.shape
    print(f"Dataset generated: {n_samples} points in {n_features}D space.")
    print(f"  - Blob 1: 100 points around (-2.5, 0.0)")
    print(f"  - Blob 2: 100 points around (+2.5, 0.0)")
    print(f"  - Noise: 25 background points\n")

    # 2. Rule of thumb for min_samples: 2 * d
    min_samples = 2 * n_features  # 4 for 2D
    k = min_samples - 1            # 3-rd nearest neighbor
    print(f"Rule of thumb for min_samples: 2 * d = {min_samples}")
    print(f"Evaluating k-distance graph with k = min_samples - 1 = {k}")

    # 3. Compute sorted k-distances & find elbow
    sorted_k_dist = compute_k_distances(X, k=k)
    knee_idx, optimal_eps = find_elbow_point(sorted_k_dist)

    print(f"\nk-Distance Elbow Detection:")
    print(f"  - Minimum 3-NN distance: {sorted_k_dist[0]:.4f}")
    print(f"  - Maximum 3-NN distance: {sorted_k_dist[-1]:.4f}")
    print(f"  - Knee detected at index: {knee_idx} / {n_samples}")
    print(f"  - Recommended optimal eps: {optimal_eps:.4f}\n")

    # 4. Parameter sensitivity sweep: eps_small vs eps_optimal vs eps_large
    eps_small = optimal_eps * 0.4
    eps_large = optimal_eps * 4.0

    configs = [
        ("eps too small", eps_small),
        ("eps optimal (from elbow)", optimal_eps),
        ("eps too large", eps_large),
    ]

    print("Parameter Sensitivity Comparison:")
    print("-" * 70)
    print(f"{'Configuration':<26} | {'eps':<8} | {'# Clusters':<12} | {'# Noise Pts':<12}")
    print("-" * 70)

    results = {}
    for desc, eps_val in configs:
        pred_labels = dbscan_predict(X, eps=eps_val, min_samples=min_samples)
        n_clusters = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
        n_noise = np.sum(pred_labels == -1)
        results[desc] = (n_clusters, n_noise)
        print(f"{desc:<26} | {eps_val:<8.4f} | {n_clusters:<12} | {n_noise:<12}")

    print("-" * 70)

    # 5. Assertions validating expected behaviors:
    # Under-clustering (eps too small): many points lost to noise, or shattered
    small_clusters, small_noise = results["eps too small"]
    opt_clusters, opt_noise = results["eps optimal (from elbow)"]
    large_clusters, large_noise = results["eps too large"]

    assert small_noise > opt_noise, "eps too small must produce more noise than optimal eps"
    assert opt_clusters == 2, f"Optimal eps must recover exactly 2 clusters, got {opt_clusters}"
    assert large_clusters == 1, f"eps too large must merge everything into 1 cluster, got {large_clusters}"
    assert large_noise < opt_noise, "eps too large absorbs noise points into the cluster"

    print("\nBehavioral Observations:")
    print("  1. eps too small -> Fragile clusters: valid points rejected as noise.")
    print("  2. eps optimal   -> Clean recovery of the 2 true clusters with noise isolated.")
    print("  3. eps too large -> Premature merging: 2 clusters + noise joined into 1 mega-cluster.")

    print("\n✓ Parameter selection and k-distance elbow curve verified successfully!")


if __name__ == "__main__":
    main()

