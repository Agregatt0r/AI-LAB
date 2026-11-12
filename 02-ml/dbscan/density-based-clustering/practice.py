"""Density-Based Clustering: Core Concepts & Spatial Density.

This module demonstrates the foundational principles of density-based clustering:
1. Generating complex non-spherical geometries (concentric rings + noise).
2. Vectorized pairwise distance matrix computation in pure NumPy.
3. Estimating local spatial density via epsilon-neighborhood cardinality.
4. Demonstrating density-reachability across chained paths vs Euclidean centroid distance.
"""

from typing import List, Tuple
import numpy as np


def generate_concentric_rings_with_noise(
    n_inner: int = 100,
    n_outer: int = 200,
    n_noise: int = 30,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate two non-convex concentric rings with uniform random noise.

    Returns:
        X: Coordinates array of shape (N, 2).
        true_labels: Ground truth labels (0 for inner, 1 for outer, -1 for noise).
    """
    rng = np.random.RandomState(random_state)

    # 1. Inner circle (radius ~ 1.0)
    theta_inner = rng.uniform(0, 2 * np.pi, n_inner)
    r_inner = rng.normal(1.0, 0.1, n_inner)
    x_inner = np.column_stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)])

    # 2. Outer ring (radius ~ 3.0)
    theta_outer = rng.uniform(0, 2 * np.pi, n_outer)
    r_outer = rng.normal(3.0, 0.15, n_outer)
    x_outer = np.column_stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)])

    # 3. Background noise points (sparse outside or between rings)
    noise_points = []
    while len(noise_points) < n_noise:
        pt = rng.uniform(-4.5, 4.5, size=2)
        r = np.linalg.norm(pt)
        # Avoid sitting directly on top of the two ring bands
        if not (0.7 <= r <= 1.3 or 2.7 <= r <= 3.3):
            noise_points.append(pt)
    x_noise = np.array(noise_points)

    X = np.vstack([x_inner, x_outer, x_noise])
    labels = np.array([0] * n_inner + [1] * n_outer + [-1] * n_noise)
    return X, labels


def compute_pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Compute NxN Euclidean distance matrix using NumPy broadcasting.

    dist[i, j] = ||X[i] - X[j]||_2
    """
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def get_epsilon_neighborhood(
    point_idx: int, dist_matrix: np.ndarray, eps: float
) -> np.ndarray:
    """Find all neighbor indices within distance epsilon of point_idx (including itself)."""
    return np.where(dist_matrix[point_idx] <= eps)[0]


def find_density_path(
    start_idx: int,
    target_idx: int,
    dist_matrix: np.ndarray,
    eps: float,
    core_mask: np.ndarray,
) -> List[int]:
    """Find a density-reachable chain of core points connecting start_idx to target_idx.

    Returns the path of indices, or empty list if not density-reachable.
    """
    visited = {start_idx}
    parent = {start_idx: None}
    queue = [start_idx]

    while queue:
        curr = queue.pop(0)
        if curr == target_idx:
            # Reconstruct path
            path = []
            node = curr
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1]

        # Only core points can expand reachability to neighbors
        if core_mask[curr]:
            neighbors = get_epsilon_neighborhood(curr, dist_matrix, eps)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = curr
                    queue.append(neighbor)

    return []


def main() -> None:
    print("=" * 70)
    print("1. DENSITY-BASED CLUSTERING: CONCENTRIC GEOMETRIES & SPATIAL DENSITY")
    print("=" * 70)

    # Generate synthetic dataset
    X, true_labels = generate_concentric_rings_with_noise(
        n_inner=100, n_outer=200, n_noise=30, random_state=42
    )
    n_samples = len(X)
    print(f"Generated dataset with {n_samples} points:")
    print(f"  - Inner ring: 100 points (r ~ 1.0)")
    print(f"  - Outer ring: 200 points (r ~ 3.0)")
    print(f"  - Uniform noise: 30 points (in [-4, 4] x [-4, 4])\n")

    # Compute pairwise distance matrix
    dist_matrix = compute_pairwise_distances(X)
    assert dist_matrix.shape == (n_samples, n_samples)
    assert np.allclose(np.diag(dist_matrix), 0.0)

    # Local density estimation
    eps = 0.5
    min_samples = 5
    local_densities = np.array([len(get_epsilon_neighborhood(i, dist_matrix, eps)) for i in range(n_samples)])

    avg_inner_density = np.mean(local_densities[true_labels == 0])
    avg_outer_density = np.mean(local_densities[true_labels == 1])
    avg_noise_density = np.mean(local_densities[true_labels == -1])

    print(f"Local Spatial Density (epsilon={eps}):")
    print(f"  - Mean density in Inner Ring: {avg_inner_density:.2f} neighbors")
    print(f"  - Mean density in Outer Ring: {avg_outer_density:.2f} neighbors")
    print(f"  - Mean density in Noise:      {avg_noise_density:.2f} neighbors")
    assert avg_inner_density > avg_noise_density * 3.0
    assert avg_outer_density > avg_noise_density * 3.0

    # Core point identification
    core_mask = local_densities >= min_samples
    print(f"\nCore Points (min_samples={min_samples}):")
    print(f"  - Inner ring core points: {np.sum(core_mask[true_labels == 0])} / 100")
    print(f"  - Outer ring core points: {np.sum(core_mask[true_labels == 1])} / 200")
    print(f"  - Noise core points:      {np.sum(core_mask[true_labels == -1])} / 30 (filtered out!)")
    assert np.sum(core_mask[true_labels == -1]) <= 5

    # Density-Reachable Path demonstration
    # Pick two points on opposite sides of outer ring
    outer_indices = np.where(true_labels == 1)[0]
    p_start = outer_indices[np.argmin(X[outer_indices, 0])]  # leftmost outer point
    p_end = outer_indices[np.argmax(X[outer_indices, 0])]    # rightmost outer point

    direct_dist = dist_matrix[p_start, p_end]
    print(f"\nOuter Ring Opposite Points:")
    print(f"  - Direct Euclidean distance: {direct_dist:.2f} (Far apart, would split in K-Means!)")

    path = find_density_path(p_start, p_end, dist_matrix, eps=eps, core_mask=core_mask)
    print(f"  - Density-connected path exists? {'YES' if len(path) > 0 else 'NO'}")
    print(f"  - Path length (chain of core neighbors): {len(path)} hops")
    assert len(path) > 0, "Points on the outer ring must be density-connected!"

    # Verify that a noise point is NOT reachable from the outer ring
    noise_idx = np.where(true_labels == -1)[0][0]
    noise_path = find_density_path(p_start, noise_idx, dist_matrix, eps=eps, core_mask=core_mask)
    print(f"  - Can reach random noise point #{noise_idx}? {'YES' if len(noise_path) > 0 else 'NO (correctly isolated)'}")
    assert len(noise_path) == 0

    print("\n✓ Density-based clustering fundamentals verified successfully!")


if __name__ == "__main__":
    main()
