# Practice: Logistic Regression with scikit-learn
# API used:
#   LogisticRegression(C=1.0)   -> L2-regularized logistic regression (C = inverse of regularization strength)
#   .fit(X, y)                  -> learns w and b
#   .predict_proba(X)           -> P(y=0|x), P(y=1|x)
#   .predict(X)                 -> hard labels using a 0.5 threshold
#   make_pipeline(StandardScaler(), ...) -> scaling is fitted on training data only (no leakage)
# Compare with 02-ml/logistic-regression/from-scratch for the same algorithm written by hand.

import numpy as np
from sklearn.linear_model import LogisticRegression  # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics import (  # pyright: ignore[reportMissingModuleSource]
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import train_test_split  # pyright: ignore[reportMissingModuleSource]
from sklearn.pipeline import make_pipeline  # pyright: ignore[reportMissingModuleSource]
from sklearn.preprocessing import StandardScaler  # pyright: ignore[reportMissingModuleSource]


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def make_dataset(n=300, seed=42):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.normal(0, 1, n), rng.normal(50, 10, n)])
    z = 2.0 * X[:, 0] - 0.3 * (X[:, 1] - 50) + 0.5
    y = (rng.uniform(size=n) < sigmoid(z)).astype(int)
    return X, y


if __name__ == "__main__":
    X, y = make_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("--- 1. Build and Train the Model ---")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    model.fit(X_train, y_train)
    clf = model.named_steps["logisticregression"]
    print(f"Coefficients (scaled features): {np.round(clf.coef_[0], 4)}")
    print(f"Intercept: {clf.intercept_[0]:.4f}")

    print("\n--- 2. Probabilities and Labels ---")
    proba = model.predict_proba(X_test)  # shape (n, 2): columns are P(y=0), P(y=1)
    labels = model.predict(X_test)
    print(f"First 5 P(y=1): {np.round(proba[:5, 1], 4)}")
    print(f"First 5 predicted labels: {labels[:5]}")

    # predict() is exactly the 0.5 threshold applied to P(y=1)
    manual_labels = (proba[:, 1] >= 0.5).astype(int)
    assert np.array_equal(labels, manual_labels), "predict() should equal threshold rule at 0.5"

    print("\n--- 3. Evaluation Metrics ---")
    acc = accuracy_score(y_test, labels)
    loss = log_loss(y_test, proba)
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test log loss (BCE): {loss:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, labels))
    print(classification_report(y_test, labels, digits=4))

    print("--- 4. Effect of Regularization Strength C ---")
    print(f"{'C':>8} | {'||coef||':>9} | {'test BCE':>9}")
    norms = []
    for C in [0.01, 0.1, 1.0, 100.0]:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))
        m.fit(X_train, y_train)
        coef_norm = np.linalg.norm(m.named_steps["logisticregression"].coef_)
        bce = log_loss(y_test, m.predict_proba(X_test))
        norms.append(coef_norm)
        print(f"{C:>8.2f} | {coef_norm:>9.4f} | {bce:>9.4f}")

    # Smaller C = stronger penalty = smaller weights
    assert all(norms[i] <= norms[i + 1] for i in range(len(norms) - 1)), \
        "Coefficient norm should grow as C increases (weaker regularization)"

    assert acc > 0.8, "scikit-learn model should classify the test set well"
    print("\nscikit-learn logistic regression matches the from-scratch behaviour.")
