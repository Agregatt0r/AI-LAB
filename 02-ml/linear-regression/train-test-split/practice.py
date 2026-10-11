# Practice: Train/Test Split & Generalization Metrics from Scratch
# Metrics:
#   MSE  = (1/n) * Σ(y - ŷ)^2
#   RMSE = sqrt(MSE)
#   MAE  = (1/n) * Σ|y - ŷ|
#   R2   = 1 - (SS_res / SS_tot) = 1 - Σ(y - ŷ)^2 / Σ(y - y_mean)^2

import numpy as np


def train_test_split_scratch(X, y, test_size=0.2, random_state=42):
    """Split arrays into random train and test subsets without sklearn."""
    np.random.seed(random_state)
    n = len(X)
    indices = np.random.permutation(n)

    test_count = int(n * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def r2_score_scratch(y_true, y_pred):
    """Compute R^2 (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / ss_tot)


def fit_linear_regression(X_train, y_train):
    """Solve weights and bias via the Normal Equation."""
    n = len(X_train)
    X_aug = np.column_stack([np.ones(n), X_train])
    theta = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y_train)
    return theta[0], theta[1:]


def predict(X, b, w):
    """Compute ŷ = X @ w + b."""
    return X @ w + b


if __name__ == "__main__":
    # Generate noisy synthetic linear dataset
    np.random.seed(42)
    n_samples = 250
    X = np.random.uniform(10, 100, (n_samples, 1))
    noise = np.random.normal(0, 15.0, (n_samples, 1))
    y = 2.5 * X + 15.0 + noise
    y = y.ravel()

    # 1. Split into 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split_scratch(X, y, test_size=0.2, random_state=42)

    print("--- 1. Dataset Partition ---")
    print(f"Total Samples: {len(X)}")
    print(f"Train Samples: {len(X_train)} ({len(X_train)/len(X)*100:.0f}%)")
    print(f"Test Samples:  {len(X_test)}  ({len(X_test)/len(X)*100:.0f}%)")

    # 2. Train model ONLY on training data
    b, w = fit_linear_regression(X_train, y_train)
    print("\n--- 2. Learned Model (Fitted on Train Set) ---")
    print(f"Weight w: {w[0]:.4f} (True: 2.5000)")
    print(f"Bias b:   {b:.4f} (True: 15.0000)")

    # 3. Evaluate on unseen Test Set
    y_test_pred = predict(X_test, b, w)

    test_mse = np.mean((y_test - y_test_pred) ** 2)
    test_rmse = np.sqrt(test_mse)
    test_mae = np.mean(np.abs(y_test - y_test_pred))
    test_r2 = r2_score_scratch(y_test, y_test_pred)

    print("\n--- 3. Evaluation on Unseen Test Set ---")
    print(f"Test MSE:   {test_mse:.4f}")
    print(f"Test RMSE:  {test_rmse:.4f}")
    print(f"Test MAE:   {test_mae:.4f}")
    print(f"Test R²:    {test_r2:.4f} (1.0 = perfect explanation)")

    # Assert model learned robust generalizable pattern
    assert test_r2 > 0.90, f"R2 score {test_r2} is below expected threshold!"
    print("\nModel successfully generalized to unseen test data with high R²!")
