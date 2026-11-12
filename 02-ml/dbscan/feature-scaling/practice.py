"""The Impact of Feature Scaling on DBSCAN.

This module demonstrates:
1. How unscaled features distort the spherical epsilon-neighborhood into a needle.
2. A pure NumPy implementation of StandardScaler from scratch.
3. The failure of DBSCAN on unscaled customer data (features on disparate scales).
4. The restoration of optimal cluster separation and noise isolation after standardization.
"""

from typing import Tuple
import numpy as np

try:
    from sklearn.cluster import DBSCAN as SklearnDBSCAN
    from sklearn.metrics import adjusted_rand_score as sklearn_ari
    from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def adjusted_rand_score_scratch(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Compute the Adjusted Rand Index (ARI) in pure NumPy."""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    n = len(labels_true)
    if n <= 1:
        return 1.0
    _, true_idx = np.unique(labels_true, return_inverse=True)
    _, pred_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = int(np.max(true_idx)) + 1
    n_clusters = int(np.max(pred_idx)) + 1
    contingency = np.zeros((n_classes, n_clusters), dtype=np.int64)
    for t, p in zip(true_idx, pred_idx):
        contingency[t, p] += 1
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


class StandardScalerScratch:
    """Standardize features by removing the mean and scaling to unit variance (pure NumPy)."""

    def __init__(self) -> None:
        self.mean_: np.ndarray = np.array([])
        self.scale_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray) -> "StandardScalerScratch":
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Avoid division by zero for constant features
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def generate_customer_segments_unscaled(
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate customer segmentation dataset with widely disparate scales.

    Feature 0: Annual Income in USD ($20,000 to $120,000, std ~ 25,000)
    Feature 1: Loyalty Score (0 to 100, std ~ 25)

    3 Genuine Segments:
      - Cluster 0: Low Income ($30k), High Loyalty (85)
      - Cluster 1: High Income ($100k), High Loyalty (85)
      - Cluster 2: High Income ($100k), Low Loyalty (15)
    """
    rng = np.random.RandomState(random_state)
    n_per_cluster = 100

    # Cluster 0: Low Income, High Loyalty
    inc_0 = rng.normal(30_000, 2_000, n_per_cluster)
    loy_0 = rng.normal(85, 4, n_per_cluster)
    c0 = np.column_stack([inc_0, loy_0])

    # Cluster 1: High Income, High Loyalty
    inc_1 = rng.normal(100_000, 2_000, n_per_cluster)
    loy_1 = rng.normal(85, 4, n_per_cluster)
    c1 = np.column_stack([inc_1, loy_1])

    # Cluster 2: High Income, Low Loyalty
    inc_2 = rng.normal(100_000, 2_000, n_per_cluster)
    loy_2 = rng.normal(15, 4, n_per_cluster)
    c2 = np.column_stack([inc_2, loy_2])

    # Noise points
    inc_noise = rng.uniform(20_000, 120_000, 30)
    loy_noise = rng.uniform(0, 100, 30)
    c_noise = np.column_stack([inc_noise, loy_noise])

    X = np.vstack([c0, c1, c2, c_noise])
    y = np.array([0] * n_per_cluster + [1] * n_per_cluster + [2] * n_per_cluster + [-1] * 30)
    return X, y


def main() -> None:
    print("=" * 70)
    print("6. THE IMPACT OF FEATURE SCALING ON DBSCAN")
    print("=" * 70)

    # 1. Dataset setup
    X_raw, y_true = generate_customer_segments_unscaled(random_state=42)
    print("Raw Unscaled Dataset Statistics:")
    print(f"  - Feature 0 (Income):  mean=${np.mean(X_raw[:, 0]):,.0f}, std=${np.std(X_raw[:, 0]):,.0f}")
    print(f"  - Feature 1 (Loyalty): mean={np.mean(X_raw[:, 1]):.1f},  std={np.std(X_raw[:, 1]):.1f}")
    print("  -> Income scale is ~1,000x larger than Loyalty score!\n")

    # 2. Experiment A: Clustering WITHOUT Feature Scaling
    # To bridge the within-cluster income standard deviation of $2,000, eps must be around $3,000-$5,000.
    # But with eps=$4,000, the 70-point difference in Loyalty score is completely submerged!
    labels_unscaled = dbscan_predict(X_raw, eps=4000.0, min_samples=6)
    ari_unscaled = adjusted_rand_score_scratch(y_true, labels_unscaled)
    n_clusters_unscaled = len(set(labels_unscaled)) - (1 if -1 in labels_unscaled else 0)

    print("Experiment A: DBSCAN on Raw Unscaled Data (eps=4000):")
    print(f"  - Discovered clusters: {n_clusters_unscaled} (Failed: merged Cluster 1 & Cluster 2 together!)")
    print(f"  - Adjusted Rand Index (ARI): {ari_unscaled:.4f}")
    print("  - Diagnosis: High-loyalty ($100k) and Low-loyalty ($100k) customers were merged into")
    print("    one single group because loyalty differences are negligible compared to $4,000 income distance.\n")

    # 3. Experiment B: Clustering WITH StandardScaler from Scratch
    scaler = StandardScalerScratch()
    X_scaled = scaler.fit_transform(X_raw)

    # Confirm unit variance
    assert np.allclose(np.mean(X_scaled, axis=0), 0.0, atol=1e-7)
    assert np.allclose(np.std(X_scaled, axis=0), 1.0, atol=1e-7)

    # Validate against scikit-learn's StandardScaler (if available)
    if HAS_SKLEARN:
        sklearn_scaler = SklearnStandardScaler()
        X_sklearn_scaled = sklearn_scaler.fit_transform(X_raw)
        assert np.allclose(X_scaled, X_sklearn_scaled), "Scratch scaler must match sklearn StandardScaler exactly"

    # With standardized features, eps around 0.35 cleanly isolates clusters
    labels_scaled = dbscan_predict(X_scaled, eps=0.35, min_samples=6)
    ari_scaled = adjusted_rand_score_scratch(y_true, labels_scaled)
    n_clusters_scaled = len(set(labels_scaled)) - (1 if -1 in labels_scaled else 0)
    n_noise_scaled = np.sum(labels_scaled == -1)

    print("Experiment B: DBSCAN on Standardized Data (eps=0.35):")
    print(f"  - Discovered clusters: {n_clusters_scaled} (Correct: 3 distinct customer groups recovered!)")
    print(f"  - Noise points isolated: {n_noise_scaled}")
    print(f"  - Adjusted Rand Index (ARI): {ari_scaled:.4f}")
    print("  - Diagnosis: Standardizing features restores an isotropic circular epsilon-ball,")
    print("    allowing both Income and Loyalty to contribute equally to spatial density.\n")

    # 4. Summary Table
    print(f"{'Condition':<25} | {'Clusters Found':<16} | {'ARI Score':<12} | {'Result':<10}")
    print("-" * 70)
    print(f"{'Without Scaling':<25} | {n_clusters_unscaled:<16} | {ari_unscaled:<12.4f} | {'FAILED':<10}")
    print(f"{'With StandardScaler':<25} | {n_clusters_scaled:<16} | {ari_scaled:<12.4f} | {'SUCCESS':<10}")
    print("-" * 70)

    # 5. Assertions
    assert ari_unscaled < 0.60, "Unscaled DBSCAN must suffer from feature dominance failure"
    assert ari_scaled > 0.90, f"Scaled DBSCAN should achieve ARI > 0.90, got {ari_scaled:.3f}"
    assert n_clusters_scaled == 3, f"Scaled DBSCAN must recover all 3 customer clusters, got {n_clusters_scaled}"

    print("\n✓ Feature scaling impact and StandardScalerScratch verified successfully!")


if __name__ == "__main__":
    main()

