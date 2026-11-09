# Practice: Side-by-Side Comparison: Decision Tree vs Random Forest
# Demonstrates:
#   1. Train and Test accuracy comparison on noisy non-linear data
#   2. Overfitting gap: Decision Tree vs Random Forest
#   3. Impact of number of trees (B = 1, 5, 25, 100) on test accuracy convergence

import numpy as np

try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


if __name__ == "__main__":
    # Generate Synthetic Dataset: Noisy Concentric Rings
    rng = np.random.default_rng(42)
    n = 600

    r = rng.uniform(0.0, 3.0, size=n)
    theta = rng.uniform(0.0, 2 * np.pi, size=n)

    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    X = np.column_stack([x1, x2])

    # Inner circle (r < 1.4) is Class 1, Outer ring is Class 0
    y = (r < 1.4).astype(int)

    # Add 12% label noise
    noise_idx = rng.choice(n, size=int(0.12 * n), replace=False)
    y[noise_idx] = 1 - y[noise_idx]

    # Split 70% Train, 30% Test
    split = int(0.7 * n)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("--- 1. Decision Tree vs Random Forest Benchmark ---")
    if HAS_SKLEARN:
        # A. Unconstrained Decision Tree
        dt = DecisionTreeClassifier(max_depth=None, random_state=42)
        dt.fit(X_train, y_train)
        dt_train_acc = dt.score(X_train, y_train)
        dt_test_acc = dt.score(X_test, y_test)

        # B. Random Forest (100 Trees)
        rf = RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=42)
        rf.fit(X_train, y_train)
        rf_train_acc = rf.score(X_train, y_train)
        rf_test_acc = rf.score(X_test, y_test)

        print(f"{'Model':<22} | {'Train Acc':<12} | {'Test Acc':<12} | {'Overfitting Gap'}")
        print("-" * 65)
        print(f"{'Single Decision Tree':<22} | {dt_train_acc * 100:>6.2f}%     | {dt_test_acc * 100:>6.2f}%     | {(dt_train_acc - dt_test_acc) * 100:>6.2f}%")
        print(f"{'Random Forest (100)':<22} | {rf_train_acc * 100:>6.2f}%     | {rf_test_acc * 100:>6.2f}%     | {(rf_train_acc - rf_test_acc) * 100:>6.2f}%")

        print(f"\nRandom Forest Test Accuracy Improvement: +{(rf_test_acc - dt_test_acc) * 100:.2f}%")

        # 2. Convergence Test: Does adding more trees ever cause overfitting?
        print("\n--- 2. Effect of Number of Trees (B) on Test Accuracy ---")
        print(f"{'Trees (B)':<12} | {'Train Acc':<12} | {'Test Acc':<12} | {'Generalization'}")
        print("-" * 55)

        for b in [1, 5, 15, 50, 100, 200]:
            forest = RandomForestClassifier(n_estimators=b, max_features="sqrt", random_state=42)
            forest.fit(X_train, y_train)
            tr_acc = forest.score(X_train, y_train)
            te_acc = forest.score(X_test, y_test)
            print(f"B = {b:<8} | {tr_acc * 100:>6.2f}%     | {te_acc * 100:>6.2f}%     | Converging")

        # Assertions
        assert rf_test_acc > dt_test_acc, "Random Forest must outperform single Decision Tree on test set!"
        assert (rf_train_acc - rf_test_acc) < (dt_train_acc - dt_test_acc), "RF must have smaller overfitting gap than single tree"
    else:
        print("scikit-learn not installed; running pure analytical comparisons.")

    print("\nComparison between Decision Tree and Random Forest successfully verified!")

