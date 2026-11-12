"""Anomaly and Outlier Detection with DBSCAN.

This module implements:
1. Multi-modal normal distribution dataset with injected low-density anomalies.
2. Density-based anomaly detection via DBSCAN noise labeling (pred = 1 if label == -1).
3. Pure NumPy calculation of Confusion Matrix, Precision, Recall, and F1-Score.
4. Benchmark comparison against Global Z-Score to highlight density-based superiority.
"""

from typing import Dict, Tuple
import numpy as np

try:
    from sklearn.cluster import DBSCAN as SklearnDBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def generate_multimodal_fraud_data(
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate multi-modal operational transaction data with rare fraud anomalies.

    Returns:
        X: Feature matrix of shape (N, 2).
        y_true_anomaly: Binary indicator (1 = anomaly/fraud, 0 = normal transaction).
    """
    rng = np.random.RandomState(random_state)

    # Mode 1: Normal consumer transactions (300 samples around [10, 10])
    mode1 = rng.normal(loc=[10.0, 10.0], scale=1.2, size=(300, 2))

    # Mode 2: Normal corporate transactions (150 samples around [35, 35])
    mode2 = rng.normal(loc=[35.0, 35.0], scale=1.8, size=(150, 2))

    # Injected anomalies: 25 sparse outliers in empty gaps and extremes
    anomalies_gap = rng.uniform(low=[18.0, 18.0], high=[26.0, 26.0], size=(15, 2))
    anomalies_extreme = rng.uniform(low=[0.0, 45.0], high=[5.0, 50.0], size=(10, 2))
    anomalies = np.vstack([anomalies_gap, anomalies_extreme])

    X = np.vstack([mode1, mode2, anomalies])
    y_anomaly = np.array([0] * 450 + [1] * 25)
    return X, y_anomaly


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute Confusion Matrix, Precision, Recall, and F1-score from first principles."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def zscore_anomaly_detector(X: np.ndarray, threshold: float = 2.5) -> np.ndarray:
    """Global Z-score outlier detector: flags point if any feature |z| > threshold."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    z_scores = np.abs((X - mean) / std)
    return (np.max(z_scores, axis=1) > threshold).astype(int)


def dbscan_anomaly_detector(
    X: np.ndarray, eps: float = 2.0, min_samples: int = 6
) -> np.ndarray:
    """DBSCAN outlier detector: flags point as anomaly if assigned noise label -1."""
    if HAS_SKLEARN:
        db = SklearnDBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)
        return (labels == -1).astype(int)

    # Pure NumPy fallback
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

    return (labels == -1).astype(int)


def main() -> None:
    print("=" * 70)
    print("4. ANOMALY DETECTION WITH DBSCAN (NOISE LABEL -1)")
    print("=" * 70)

    # 1. Dataset setup
    X, y_true = generate_multimodal_fraud_data(random_state=42)
    n_samples = len(X)
    n_anomalies = int(np.sum(y_true == 1))
    print(f"Generated dataset with {n_samples} total events:")
    print(f"  - Normal transactions: {n_samples - n_anomalies} (across 2 separate operational regimes)")
    print(f"  - Fraudulent anomalies: {n_anomalies} (in sparse intermediate gaps)\n")

    # 2. Run DBSCAN Anomaly Detector
    eps = 2.0
    min_samples = 6
    y_pred_dbscan = dbscan_anomaly_detector(X, eps=eps, min_samples=min_samples)
    metrics_dbscan = compute_classification_metrics(y_true, y_pred_dbscan)

    # 3. Run Baseline Global Z-Score Anomaly Detector
    y_pred_zscore = zscore_anomaly_detector(X, threshold=2.2)
    metrics_zscore = compute_classification_metrics(y_true, y_pred_zscore)

    # 4. Display Comparison
    print(f"{'Metric':<18} | {'DBSCAN (Density-Based)':<22} | {'Global Z-Score (Parametric)':<25}")
    print("-" * 70)
    print(f"{'True Positives (TP)':<18} | {metrics_dbscan['tp']:<22} | {metrics_zscore['tp']:<25}")
    print(f"{'False Positives(FP)':<18} | {metrics_dbscan['fp']:<22} | {metrics_zscore['fp']:<25}")
    print(f"{'False Negatives(FN)':<18} | {metrics_dbscan['fn']:<22} | {metrics_zscore['fn']:<25}")
    print(f"{'Precision':<18} | {metrics_dbscan['precision']:<22.4f} | {metrics_zscore['precision']:<25.4f}")
    print(f"{'Recall':<18} | {metrics_dbscan['recall']:<22.4f} | {metrics_zscore['recall']:<25.4f}")
    print(f"{'F1-Score':<18} | {metrics_dbscan['f1']:<22.4f} | {metrics_zscore['f1']:<25.4f}")
    print("-" * 70)

    # 5. Assertions verifying density-based superiority
    assert metrics_dbscan["recall"] >= 0.90, f"DBSCAN recall should be >= 0.90, got {metrics_dbscan['recall']:.2f}"
    assert metrics_dbscan["precision"] >= 0.85, f"DBSCAN precision should be >= 0.85, got {metrics_dbscan['precision']:.2f}"
    assert metrics_dbscan["f1"] > metrics_zscore["f1"] + 0.30, (
        "DBSCAN must substantially outperform global Z-score on multimodal anomaly data!"
    )

    print("\nKey Takeaways:")
    print("  1. DBSCAN detects subtle anomalies located in low-density regions between clusters.")
    print("  2. Global Z-Score fails because the multimodal distribution creates a wide variance,")
    print("     causing intermediate anomalies to look 'close to the global mean'.")
    print("  3. Density-based outlier detection makes zero distributional assumptions.")

    print("\n✓ Anomaly detection via DBSCAN noise isolation verified successfully!")


if __name__ == "__main__":
    main()
