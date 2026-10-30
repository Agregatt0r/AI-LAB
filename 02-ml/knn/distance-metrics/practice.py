# Practice: Distance Metrics & Vectorized Pairwise Distances
# Implements:
#   - Euclidean (L2)
#   - Manhattan (L1)
#   - Minkowski (Lp)
#   - Cosine Distance
#   - Vectorized Pairwise Distance Matrix

import time
import numpy as np


def euclidean_distance(u, v):
    """Euclidean (L2) distance."""
    return np.sqrt(np.sum((u - v) ** 2))


def manhattan_distance(u, v):
    """Manhattan (L1) distance."""
    return np.sum(np.abs(u - v))


def minkowski_distance(u, v, p=3):
    """Minkowski (Lp) distance."""
    return np.sum(np.abs(u - v) ** p) ** (1.0 / p)


def cosine_distance(u, v):
    """Cosine distance = 1 - cosine_similarity."""
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 1.0
    cosine_sim = np.dot(u, v) / (norm_u * norm_v)
    return 1.0 - cosine_sim


def pairwise_distances_loop(X_test, X_train, metric="euclidean"):
    """Compute pairwise distance matrix using Python loops."""
    m, n = len(X_test), len(X_train)
    D = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if metric == "euclidean":
                D[i, j] = euclidean_distance(X_test[i], X_train[j])
            elif metric == "manhattan":
                D[i, j] = manhattan_distance(X_test[i], X_train[j])
    return D


def pairwise_distances_vectorized(X_test, X_train):
    """Vectorized Euclidean pairwise distance matrix:
    ||u - v||^2 = ||u||^2 - 2 u·v + ||v||^2
    """
    # ||u||^2 for each test row (m, 1)
    test_sq = np.sum(X_test ** 2, axis=1, keepdims=True)
    # ||v||^2 for each train row (1, n)
    train_sq = np.sum(X_train ** 2, axis=1, keepdims=True).T
    # -2 * X_test @ X_train.T (m, n)
    dists_sq = test_sq - 2.0 * np.dot(X_test, X_train.T) + train_sq
    # Numerical safety: clip negative zeros
    dists_sq = np.maximum(dists_sq, 0.0)
    return np.sqrt(dists_sq)


if __name__ == "__main__":
    # 1. Compare Distance Values on Two Vectors
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0, 0.0, -1.0])

    d_euc = euclidean_distance(u, v)
    d_man = manhattan_distance(u, v)
    d_mink = minkowski_distance(u, v, p=3)
    d_cos = cosine_distance(u, v)

    print("--- 1. Distance Metrics on Vectors u and v ---")
    print(f"Vector u: {u.tolist()}")
    print(f"Vector v: {v.tolist()}")
    print(f"Euclidean (L2): {d_euc:.4f}  (sqrt(3^2 + 2^2 + 4^2) = sqrt(29) ≈ 5.3852)")
    print(f"Manhattan (L1): {d_man:.4f}  (|3| + |2| + |4| = 9.0000)")
    print(f"Minkowski (p=3):{d_mink:.4f} ((3^3 + 2^3 + 4^3)^(1/3) = 99^(1/3) ≈ 4.6261)")
    print(f"Cosine Distance:{d_cos:.4f}")

    assert np.isclose(d_euc, np.sqrt(29)), "Euclidean distance calculation mismatch!"
    assert np.isclose(d_man, 9.0), "Manhattan distance calculation mismatch!"
    assert np.isclose(d_mink, 99.0 ** (1.0 / 3.0)), "Minkowski distance calculation mismatch!"

    # 2. Benchmark Vectorized Distance Matrix
    print("\n--- 2. Pairwise Distance Matrix Benchmark ---")
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((500, 10))
    X_test = rng.standard_normal((100, 10))

    # Loop version
    t0 = time.perf_counter()
    D_loop = pairwise_distances_loop(X_test, X_train, metric="euclidean")
    t_loop = time.perf_counter() - t0

    # Vectorized version
    t0 = time.perf_counter()
    D_vec = pairwise_distances_vectorized(X_test, X_train)
    t_vec = time.perf_counter() - t0

    print(f"Matrix shape: {D_vec.shape} (100 test x 500 train)")
    print(f"Loop time:       {t_loop * 1000:.2f} ms")
    print(f"Vectorized time: {t_vec * 1000:.2f} ms")
    print(f"Speedup:         {t_loop / t_vec:.1f}x faster!")

    # Verify numerical equivalence
    assert np.allclose(D_loop, D_vec, atol=1e-5), "Vectorized distances do not match loop output!"
    print("\nVectorized pairwise distance matrix verified accurately!")

