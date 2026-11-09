# 02-ml: Machine Learning Algorithms & Foundations from Scratch

This directory contains practical implementations, mathematical foundations, and intuitive explanations of core Machine Learning algorithms and concepts implemented from scratch in Python and NumPy.

Each topic builds directly on the mathematical principles explored in `01-math` (linear algebra, calculus, probability, and statistics).

---

## 🗺️ Roadmap & Modules

| # | Module | Core Focus | Why it matters |
|---|---|---|---|
| 1 | **[ML Foundations](ml-basics/README.md)** | Core concepts, paradigms & pipeline | What ML is, Supervised vs Unsupervised learning, Features ($X$) vs Labels ($y$), Regression vs Classification, Training vs Inference, and the end-to-end ML Pipeline. |
| 2 | **[Linear Regression](linear-regression/README.md)** | Supervised Learning for continuous targets | The foundational regression model: hypothesis, MSE, gradient descent, vectorization, multi-variable modeling, the Normal Equation, and regularization. |
| 3 | **[Logistic Regression](logistic-regression/README.md)** | Supervised Learning for binary classification | The foundational classification model: sigmoid, probabilities and thresholds, decision boundary, Binary Cross-Entropy, gradient descent, and scikit-learn. |
| 4 | **[K-Nearest Neighbors (KNN)](knn/README.md)** | Instance-based learning for classification & regression | The foundational distance-based model: lazy learning, Euclidean/Manhattan metrics, choosing $k$, feature scaling, StandardScaler, and from-scratch estimators. |
| 5 | **[Decision Trees](decision-trees/README.md)** | Non-linear tree-based learning | Tree architecture, nodes/branches/leaves, Gini impurity, Entropy & Information Gain, split finding, hyperparameters, overfitting & pruning, and from-scratch decision tree classifier. |
| 6 | **[Random Forest](random-forest/README.md)** | Ensemble learning & variance reduction | Wisdom of crowds, bagging, bootstrap sampling, random feature selection ($m \approx \sqrt{d}$), prediction aggregation, feature importance (MDI), and scikit-learn. |

---

## 📂 Structure

Each topic folder is organized into focused submodules, each containing:
- `README.md`: Mathematical theory, intuition, geometric interpretation, and formulas.
- `practice.py`: Hands-on Python and NumPy implementations from first principles without black-box machine learning libraries.
