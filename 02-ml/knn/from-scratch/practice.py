# Practice: Complete K-Nearest Neighbors From Scratch
# Implements:
#   - KNNClassifier (predict, predict_proba, score)
#   - KNNRegressor (predict with uniform & distance weights, score via R^2)
#   - Metrics: Euclidean & Manhattan
#   - StandardScaler integration
#   - Side-by-side verification against scikit-learn

import numpy as np

try:
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# -------------------------------------------------------------
# 1. Custom StandardScaler
# -------------------------------------------------------------
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_ = np.where(self.scale_ == 0.0, 1.0, self.scale_)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# -------------------------------------------------------------
# 2. Base KNN Class
# -------------------------------------------------------------
class _BaseKNN:
    def __init__(self, n_neighbors=5, metric="euclidean", weights="uniform"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X, y):
        """Memorize the training set."""
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y)
        return self

    def _compute_distances(self, X):
        """Compute pairwise distance matrix between query X and stored X_train_."""
        X = np.asarray(X, dtype=float)
        if self.metric == "euclidean":
            # Vectorized ||u - v||^2 = ||u||^2 - 2u·v + ||v||^2
            dists_sq = (
                np.sum(X ** 2, axis=1, keepdims=True)
                - 2.0 * np.dot(X, self.X_train_.T)
                + np.sum(self.X_train_ ** 2, axis=1, keepdims=True).T
            )
            return np.sqrt(np.maximum(dists_sq, 0.0))
        elif self.metric == "manhattan":
            # Broadcasting Manhattan distance: |X[:, None, :] - X_train_[None, :, :]|
            return np.sum(np.abs(X[:, np.newaxis, :] - self.X_train_[np.newaxis, :, :]), axis=2)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def kneighbors(self, X):
        """Return distances and indices of the k nearest neighbors for each query."""
        dists = self._compute_distances(X)
        k_indices = np.argsort(dists, axis=1)[:, : self.n_neighbors]
        k_dists = np.take_along_axis(dists, k_indices, axis=1)
        return k_dists, k_indices


# -------------------------------------------------------------
# 3. KNN Classifier
# -------------------------------------------------------------
class KNNClassifier(_BaseKNN):
    def fit(self, X, y):
        super().fit(X, y)
        self.classes_ = np.unique(self.y_train_)
        return self

    def predict_proba(self, X):
        """Return class probability distribution for each query sample."""
        k_dists, k_indices = self.kneighbors(X)
        neighbor_y = self.y_train_[k_indices]
        n_queries = len(X)
        probabilities = np.zeros((n_queries, len(self.classes_)))

        for i in range(n_queries):
            labels = neighbor_y[i]
            if self.weights == "uniform":
                for c_idx, c in enumerate(self.classes_):
                    probabilities[i, c_idx] = np.mean(labels == c)
            elif self.weights == "distance":
                dists = k_dists[i]
                weights = 1.0 / (dists + 1e-10)
                total_w = np.sum(weights)
                for c_idx, c in enumerate(self.classes_):
                    mask = (labels == c)
                    probabilities[i, c_idx] = np.sum(weights[mask]) / total_w
            else:
                raise ValueError(f"Unknown weights: {self.weights}")

        return probabilities

    def predict(self, X):
        """Predict the most probable class for each query sample."""
        probs = self.predict_proba(X)
        best_class_idx = np.argmax(probs, axis=1)
        return self.classes_[best_class_idx]

    def score(self, X, y):
        """Return mean accuracy score."""
        return np.mean(self.predict(X) == y)


# -------------------------------------------------------------
# 4. KNN Regressor
# -------------------------------------------------------------
class KNNRegressor(_BaseKNN):
    def predict(self, X):
        """Predict continuous target values."""
        k_dists, k_indices = self.kneighbors(X)
        neighbor_y = self.y_train_[k_indices]
        n_queries = len(X)
        predictions = np.zeros(n_queries)

        for i in range(n_queries):
            y_vals = neighbor_y[i]
            if self.weights == "uniform":
                predictions[i] = np.mean(y_vals)
            elif self.weights == "distance":
                dists = k_dists[i]
                weights = 1.0 / (dists + 1e-10)
                predictions[i] = np.sum(weights * y_vals) / np.sum(weights)
            else:
                raise ValueError(f"Unknown weights: {self.weights}")

        return predictions

    def score(self, X, y):
        """Return R^2 coefficient of determination."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - (ss_res / ss_tot)


if __name__ == "__main__":
    # =========================================================
    # Task A: Multi-Class Classification (Iris-like 3-Class Data)
    # =========================================================
    print("--- 1. Testing KNNClassifier from Scratch ---")
    rng = np.random.default_rng(42)
    n_samples_per_class = 60

    # 3 clusters in 3D feature space
    c0 = rng.normal(loc=[1.0, 1.0, 50.0], scale=[0.6, 0.6, 5.0], size=(n_samples_per_class, 3))
    c1 = rng.normal(loc=[3.5, 1.5, 80.0], scale=[0.6, 0.6, 5.0], size=(n_samples_per_class, 3))
    c2 = rng.normal(loc=[2.0, 4.0, 110.0], scale=[0.6, 0.6, 5.0], size=(n_samples_per_class, 3))

    X_clf = np.vstack([c0, c1, c2])
    y_clf = np.array([0] * n_samples_per_class + [1] * n_samples_per_class + [2] * n_samples_per_class)

    # Train / Test split
    indices = rng.permutation(len(X_clf))
    train_idx, test_idx = indices[:130], indices[130:]
    X_clf_train, X_clf_test = X_clf[train_idx], X_clf[test_idx]
    y_clf_train, y_clf_test = y_clf[train_idx], y_clf[test_idx]

    # Preprocessing: StandardScaler
    scaler_clf = StandardScaler()
    X_clf_train_scaled = scaler_clf.fit_transform(X_clf_train)
    X_clf_test_scaled = scaler_clf.transform(X_clf_test)

    # Train Custom KNN Classifier
    clf_custom = KNNClassifier(n_neighbors=5, metric="euclidean", weights="distance")
    clf_custom.fit(X_clf_train_scaled, y_clf_train)

    train_acc = clf_custom.score(X_clf_train_scaled, y_clf_train)
    test_acc = clf_custom.score(X_clf_test_scaled, y_clf_test)

    print(f"Custom Classifier Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Custom Classifier Test Accuracy:  {test_acc * 100:.2f}%")

    if HAS_SKLEARN:
        clf_sk = KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="distance")
        clf_sk.fit(X_clf_train_scaled, y_clf_train)
        sk_test_acc = clf_sk.score(X_clf_test_scaled, y_clf_test)
        print(f"scikit-learn Test Accuracy:       {sk_test_acc * 100:.2f}%")
        assert np.isclose(test_acc, sk_test_acc), "Custom classifier accuracy does not match scikit-learn!"
        assert np.array_equal(clf_custom.predict(X_clf_test_scaled), clf_sk.predict(X_clf_test_scaled)), "Predictions mismatch!"
        print("-> Exact prediction match with scikit-learn confirmed!")

    assert test_acc > 0.95, "Test accuracy should exceed 95%"

    # =========================================================
    # Task B: Multi-Feature Regression (House Price Prediction)
    # =========================================================
    print("\n--- 2. Testing KNNRegressor from Scratch ---")
    n_reg = 200
    size = rng.uniform(40.0, 150.0, n_reg)
    rooms = rng.integers(1, 5, n_reg)
    age = rng.uniform(1.0, 25.0, n_reg)

    X_reg = np.column_stack([size, rooms, age])
    # Ground truth non-linear target: Price ($)
    y_reg = 2500.0 * size + 15000.0 * rooms - 1200.0 * (age ** 1.1) + rng.normal(0, 3000, n_reg)

    reg_train_idx, reg_test_idx = indices[:150], indices[150:]
    X_reg_train, X_reg_test = X_reg[reg_train_idx], X_reg[reg_test_idx]
    y_reg_train, y_reg_test = y_reg[reg_train_idx], y_reg[reg_test_idx]

    # Preprocessing
    scaler_reg = StandardScaler()
    X_reg_train_scaled = scaler_reg.fit_transform(X_reg_train)
    X_reg_test_scaled = scaler_reg.transform(X_reg_test)

    # Train Custom KNN Regressor
    reg_custom = KNNRegressor(n_neighbors=5, metric="euclidean", weights="distance")
    reg_custom.fit(X_reg_train_scaled, y_reg_train)

    r2_custom = reg_custom.score(X_reg_test_scaled, y_reg_test)
    y_pred_custom = reg_custom.predict(X_reg_test_scaled)
    rmse_custom = np.sqrt(np.mean((y_reg_test - y_pred_custom) ** 2))

    print(f"Custom Regressor Test R² Score: {r2_custom:.4f}")
    print(f"Custom Regressor Test RMSE:     ${rmse_custom:,.2f}")

    if HAS_SKLEARN:
        reg_sk = KNeighborsRegressor(n_neighbors=5, metric="euclidean", weights="distance")
        reg_sk.fit(X_reg_train_scaled, y_reg_train)
        r2_sk = reg_sk.score(X_reg_test_scaled, y_reg_test)
        print(f"scikit-learn Test R² Score:     {r2_sk:.4f}")
        assert np.isclose(r2_custom, r2_sk, atol=1e-3), "Custom regressor R2 does not match scikit-learn!"
        print("-> Exact matching performance with scikit-learn confirmed!")

    assert r2_custom > 0.90, "Regressor R² should exceed 0.90"

    print("\nComplete KNN implementations from scratch verified successfully!")

