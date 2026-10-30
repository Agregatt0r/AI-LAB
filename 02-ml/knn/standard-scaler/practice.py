# Practice: StandardScaler from Scratch & Comparison with scikit-learn
# Implements:
#   - Z-score normalization: z = (x - μ) / σ
#   - Inverse transform: x = z * σ + μ
#   - Zero variance protection
#   - Verification against sklearn.preprocessing.StandardScaler (if installed)

import numpy as np

try:
    from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class StandardScaler:
    """Standardizes features by removing the mean and scaling to unit variance."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None  # Standard deviation

    def fit(self, X):
        """Compute mean and std on training data."""
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        # Population standard deviation (ddof=0 to match scikit-learn)
        self.scale_ = np.std(X, axis=0)
        # Protect against division by zero for constant features
        self.scale_ = np.where(self.scale_ == 0.0, 1.0, self.scale_)
        return self

    def transform(self, X):
        """Apply standardization using stored mean and std."""
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        """Revert standardized features back to original scale."""
        X_scaled = np.asarray(X_scaled, dtype=float)
        return X_scaled * self.scale_ + self.mean_


if __name__ == "__main__":
    # 1. Test Dataset with widely differing scales and one constant feature
    # Feature 1: House size [40 - 150]
    # Feature 2: Bedrooms [1 - 4]
    # Feature 3: Constant feature [5.0 everywhere]
    X_train = np.array([
        [50.0, 1.0, 5.0],
        [85.0, 2.0, 5.0],
        [120.0, 3.0, 5.0],
        [65.0, 2.0, 5.0],
        [110.0, 3.0, 5.0],
    ])

    X_test = np.array([
        [70.0, 2.0, 5.0],
        [130.0, 4.0, 5.0],
    ])

    print("--- 1. Training Set Before Standardization ---")
    print(X_train)

    # 2. Fit and Transform using Custom Scaler
    custom_scaler = StandardScaler()
    X_train_custom = custom_scaler.fit_transform(X_train)
    X_test_custom = custom_scaler.transform(X_test)

    print("\n--- 2. Custom Scaler Parameters ---")
    print(f"Learned Mean: {np.round(custom_scaler.mean_, 4)}")
    print(f"Learned Std:  {np.round(custom_scaler.scale_, 4)}")

    print("\n--- 3. Standardized Training Set ---")
    print(np.round(X_train_custom, 4))
    print(f"Transformed Mean (approx 0): {np.round(np.mean(X_train_custom, axis=0), 4)}")
    print(f"Transformed Std  (approx 1): {np.round(np.std(X_train_custom, axis=0), 4)}")

    # 3. Verify against scikit-learn if available
    print("\n--- 4. Verification against scikit-learn ---")
    if HAS_SKLEARN:
        sk_scaler = SklearnStandardScaler()
        X_train_sk = sk_scaler.fit_transform(X_train)
        X_test_sk = sk_scaler.transform(X_test)

        assert np.allclose(custom_scaler.mean_, sk_scaler.mean_), "Means do not match sklearn!"
        assert np.allclose(X_train_custom, X_train_sk), "Train transforms do not match sklearn!"
        assert np.allclose(X_test_custom, X_test_sk), "Test transforms do not match sklearn!"
        print("Exact match with sklearn.preprocessing.StandardScaler confirmed!")
    else:
        print("scikit-learn not in current environment; verified against NumPy analytical standard.")

    # Mathematical assertions
    # Transformed non-constant columns must have mean 0 and std 1
    assert np.allclose(np.mean(X_train_custom[:, :2], axis=0), [0.0, 0.0]), "Mean must be 0!"
    assert np.allclose(np.std(X_train_custom[:, :2], axis=0), [1.0, 1.0]), "Std must be 1!"
    # Constant column should be 0 everywhere without exploding into NaN
    assert np.all(X_train_custom[:, 2] == 0.0), "Constant column should be 0"

    # 4. Verify Inverse Transform
    X_recovered = custom_scaler.inverse_transform(X_train_custom)
    assert np.allclose(X_recovered, X_train), "Inverse transform failed to reconstruct original data!"
    print("Inverse transformation accurately restored original features!")

    print("\nStandardScaler implementation from scratch successfully verified!")
