# Logistic Regression with scikit-learn

After implementing logistic regression by hand, this module shows the same model through **scikit-learn**, the standard production library. Knowing the from-scratch version lets you understand what each option actually changes.

---

## 1. Core API

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train, y_train)

model.predict_proba(X_test)   # shape (n, 2): [P(y=0), P(y=1)]
model.predict(X_test)         # hard labels at threshold 0.5
model.coef_, model.intercept_ # learned w and b
```

| Argument | Meaning |
|---|---|
| `C` | Inverse regularization strength. Smaller $C$ = stronger penalty |
| `penalty` | Regularization type (default L2) |
| `solver` | Optimizer (default `lbfgs`, a quasi-Newton method, not plain gradient descent) |
| `max_iter` | Iteration cap; increase it if you see convergence warnings |

---

## 2. Mapping to the From-Scratch Version

| From scratch | scikit-learn |
|---|---|
| `sigmoid(X @ w + b)` | `predict_proba(X)[:, 1]` |
| `p >= threshold` | `predict(X)` (threshold 0.5) |
| BCE loss | `sklearn.metrics.log_loss` |
| Gradient descent loop | `solver="lbfgs"` (or `"saga"`, `"liblinear"`) |
| Standardize with train stats | `StandardScaler` inside a `Pipeline` |

The objective is the same BCE, plus an L2 penalty on $\mathbf{w}$ by default:

$$\min_{\mathbf{w}, b}\; \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n}\ell(y_i, p_i)$$

---

## 3. Pipelines Prevent Leakage

`make_pipeline(StandardScaler(), LogisticRegression())` fits the scaler on the training data inside `fit`, and reuses those statistics in `predict`. This is the same rule the from-scratch class follows.

---

## 4. Regularization Strength $C$

- Small $C$ (strong penalty): smaller weights, simpler boundary, may underfit.
- Large $C$ (weak penalty): larger weights, may overfit.

The practice script shows the coefficient norm rising as $C$ grows. This is the same idea as Ridge regularization in `../../linear-regression/regularization`.

---

## 5. Evaluation Tools

- `accuracy_score`, `confusion_matrix`, `classification_report` for labels.
- `log_loss` for probabilistic quality (BCE).
- Adjust the threshold on `predict_proba` when the default 0.5 is not appropriate (see `../probability-and-threshold`).

---

## Summary

- `LogisticRegression` implements the same model, loss, and boundary as the from-scratch version.
- Wrap it in a `Pipeline` with `StandardScaler` to avoid data leakage.
- `C` controls regularization; `predict_proba` gives the probabilities that the threshold acts on.

**Previous:** [From Scratch](../from-scratch/README.md)
