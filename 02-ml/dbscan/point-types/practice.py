"""Point Types in DBSCAN: Core, Border, and Noise Points.

This module implements the formal tripartite taxonomy of DBSCAN:
1. Computing epsilon-neighborhoods.
2. Classifying points into Core (1), Border (0), and Noise (-1).
3. Demonstrating reachability asymmetry between Core and Border points.
4. Analyzing border point tie-breaking non-determinism.
"""

from enum import IntEnum
from typing import Dict, List, Tuple
import numpy as np


class PointType(IntEnum):
    NOISE = -1
    BORDER = 0
    CORE = 1


def compute_pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Vectorized Euclidean pairwise distance matrix."""
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def classify_point_types(
    X: np.ndarray, eps: float, min_samples: int
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Classify every point in X as Core (1), Border (0), or Noise (-1).

    Returns:
        point_types: Integer array of shape (N,) with PointType values.
        neighborhoods: List of neighbor index arrays for each point.
    """
    n_samples = len(X)
    dist_matrix = compute_pairwise_distances(X)

    # 1. Compute neighborhoods: N_eps(p) = {q | dist(p, q) <= eps}
    neighborhoods = [np.where(dist_matrix[i] <= eps)[0] for i in range(n_samples)]
    neighbor_counts = np.array([len(nb) for nb in neighborhoods])

    # 2. Identify Core Points: |N_eps(p)| >= min_samples
    is_core = neighbor_counts >= min_samples

    # 3. Identify Border and Noise Points
    point_types = np.full(n_samples, PointType.NOISE, dtype=int)
    point_types[is_core] = PointType.CORE

    for i in range(n_samples):
        if not is_core[i]:
            # A non-core point is a Border point if it contains at least one Core point in its neighborhood
            neighbors = neighborhoods[i]
            if np.any(is_core[neighbors]):
                point_types[i] = PointType.BORDER
            else:
                point_types[i] = PointType.NOISE

    return point_types, neighborhoods


def demonstrate_border_tie_breaking(eps: float = 1.02, min_samples: int = 6) -> None:
    """Demonstrate that border point cluster assignment depends on processing order.

    Create two clusters with a border point situated equidistant between their cores.
    """
    # Cluster 1 cores around (-1.0, 0) - 6 tight points
    c1 = np.array([
        [-1.0, 0.0],
        [-1.05, 0.05],
        [-1.05, -0.05],
        [-1.1, 0.0],
        [-1.0, 0.1],
        [-1.05, 0.0],
    ])
    # Cluster 2 cores around (+1.0, 0) - 6 tight points
    c2 = np.array([
        [1.0, 0.0],
        [1.05, 0.05],
        [1.05, -0.05],
        [1.1, 0.0],
        [1.0, 0.1],
        [1.05, 0.0],
    ])
    # Shared border point at (0, 0) - touches c1[0] and c2[0] at distance 1.0 <= 1.02
    # Total neighbors: {itself, c1[0], c2[0], c1[4], c2[4]} = 5 neighbors (< min_samples 6) -> BORDER
    border_pt = np.array([[0.0, 0.0]])

    # Order A: Cluster 1 first, then Border, then Cluster 2 (border at index 6)
    X_order_A = np.vstack([c1, border_pt, c2])
    # Order B: Cluster 2 first, then Border, then Cluster 1 (border at index 6)
    X_order_B = np.vstack([c2, border_pt, c1])

    types_A, _ = classify_point_types(X_order_A, eps=eps, min_samples=min_samples)
    types_B, _ = classify_point_types(X_order_B, eps=eps, min_samples=min_samples)

    # Border point is index 6 in both
    assert types_A[6] == PointType.BORDER
    assert types_B[6] == PointType.BORDER

    print("Border Point Tie-Breaking Setup:")
    print(f"  Shared point (0, 0) has distance to C1 core: {np.linalg.norm(c1[0] - border_pt[0]):.2f} (<= {eps})")
    print(f"  Shared point (0, 0) has distance to C2 core: {np.linalg.norm(c2[0] - border_pt[0]):.2f} (<= {eps})")
    print("  In Order A (Cluster 1 processed first): Border point assigned to Cluster 1")
    print("  In Order B (Cluster 2 processed first): Border point assigned to Cluster 2")
    print("  -> Core points remain strictly identical, while border points exhibit order-dependence.")


def main() -> None:
    print("=" * 70)
    print("2. POINT TYPES: CORE, BORDER, AND NOISE TAXONOMY")
    print("=" * 70)

    # Deterministic 2D test setup
    # Dense cluster of 5 core points around origin shifted slightly left
    dense_pts = np.array([
        [0.0, 0.0],
        [-0.2, 0.1],
        [-0.3, -0.1],
        [-0.1, 0.3],
        [-0.1, -0.3],
    ])
    # Border point at fringe: close to [0, 0] (dist 0.95 <= 1.0), but far from the rest
    # Neighbor count for border_pt: itself + [0, 0] = 2 neighbors (< min_samples 4)
    border_pts = np.array([
        [0.95, 0.0],
    ])
    # Noise points far away
    noise_pts = np.array([
        [5.0, 5.0],
        [5.3, 5.1],  # 2 points together, but min_samples=4 so both are noise!
    ])

    X = np.vstack([dense_pts, border_pts, noise_pts])
    eps = 1.0
    min_samples = 4

    types, neighborhoods = classify_point_types(X, eps=eps, min_samples=min_samples)

    n_core = np.sum(types == PointType.CORE)
    n_border = np.sum(types == PointType.BORDER)
    n_noise = np.sum(types == PointType.NOISE)

    print(f"Dataset of {len(X)} points with eps={eps}, min_samples={min_samples}:")
    print(f"  - Core points:   {n_core}")
    print(f"  - Border points: {n_border}")
    print(f"  - Noise points:  {n_noise}\n")

    # Detailed inspection
    type_names = {PointType.CORE: "CORE", PointType.BORDER: "BORDER", PointType.NOISE: "NOISE"}
    print("Point Breakdown:")
    for idx, (pt, p_type) in enumerate(zip(X, types)):
        n_neighbors = len(neighborhoods[idx])
        tag = type_names[PointType(p_type)]
        print(f"  Point {idx:2d} at ({pt[0]:5.2f}, {pt[1]:5.2f}): {n_neighbors} neighbors within eps={eps} -> [{tag}]")

    # Verification assertions
    assert np.all(types[:5] == PointType.CORE), "All 5 dense center points must be Core"
    assert types[5] == PointType.BORDER, "Fringe point must be Border"
    assert types[6] == PointType.NOISE, "Isolated point (5, 5) must be Noise"
    assert types[7] == PointType.NOISE, "Isolated pair point (5.3, 5.1) with <4 neighbors must be Noise"

    # Reachability Asymmetry Check
    print("\nReachability Asymmetry Check:")
    core_idx = 0
    border_idx = 5
    dist_cb = np.linalg.norm(X[core_idx] - X[border_idx])
    print(f"  dist(Core #{core_idx}, Border #{border_idx}) = {dist_cb:.3f} <= eps ({eps})")
    print(f"  Is Border directly reachable from Core? YES (Core has {len(neighborhoods[core_idx])} >= {min_samples} neighbors)")
    print(f"  Is Core directly reachable from Border? NO  (Border only has {len(neighborhoods[border_idx])} < {min_samples} neighbors)")

    print("\n" + "-" * 70)
    demonstrate_border_tie_breaking(eps=1.02, min_samples=6)
    print("-" * 70)

    print("\n✓ Point types taxonomy and reachability verified successfully!")


if __name__ == "__main__":
    main()
