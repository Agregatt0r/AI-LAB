# Regression vs Classification

Within Supervised Learning, tasks are fundamentally divided into two major problem types depending on the mathematical nature of the target variable $y$: **Regression** and **Classification**.

---

## 1. Regression

The target $y$ is a **continuous real number** ($y \in \mathbb{R}$).

$$\hat{y} = f(\mathbf{x}) \in (-\infty, +\infty)$$

- **Question Answered**: *"How much?"* or *"How many?"*
- **Geometric View**: Fitting a continuous curve or hyperplane that passes as close as possible to the data points.
- **Real-World Examples**:
  - Predicting property sales price based on location and size.
  - Forecasting tomorrow's temperature in Celsius.
  - Estimating a vehicle's remaining battery life in kilometers.
- **Common Evaluation Metrics**:
  - **MSE** (Mean Squared Error): $\frac{1}{n} \sum (y - \hat{y})^2$
  - **RMSE** (Root Mean Squared Error): $\sqrt{\text{MSE}}$
  - **MAE** (Mean Absolute Error): $\frac{1}{n} \sum |y - \hat{y}|$
  - **$R^2$** (Coefficient of Determination)

---

## 2. Classification

The target $y$ is a **discrete category or class label** ($y \in \{0, 1, \dots, K-1\}$).

$$\hat{y} \in \{c_1, c_2, \dots, c_K\} \quad \text{or} \quad P(y = c \mid \mathbf{x}) \in [0, 1]$$

- **Question Answered**: *"Which category does this belong to?"*
- **Geometric View**: Drawing a **Decision Boundary** through the feature space that separates different classes.
- **Problem Variations**:
  - **Binary Classification**: Exactly 2 classes (e.g. Email: `Spam (1)` vs `Not Spam (0)`).
  - **Multi-class Classification**: More than 2 mutually exclusive classes (e.g. Handwritten digit: $0, 1, 2, \dots, 9$).
  - **Multi-label Classification**: A single sample can belong to multiple categories simultaneously (e.g. Article tags: `[AI, Science]`).
- **Common Evaluation Metrics**:
  - **Accuracy**: $\frac{\text{Correct Predictions}}{\text{Total Predictions}}$
  - **Precision**: Of all predicted positives, how many were actually positive?
  - **Recall**: Of all actual positives, how many did the model identify?
  - **F1-Score**: Harmonic mean of Precision and Recall.
  - **Confusion Matrix**: Table of True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN).

---

## 3. Comparison Matrix

| Property | Regression | Classification |
|---|---|---|
| **Target Type ($y$)** | Continuous numerical ($y \in \mathbb{R}$) | Discrete categories ($y \in \{0, 1, \dots\}$) |
| **Model Output** | Scalar value ($\hat{y} = 245000.50$) | Class label ($\hat{y} = 1$) or probability ($0.92$) |
| **Core Goal** | Minimize distance to points | Maximize separation between classes |
| **Error Measure** | Residual differences $(y - \hat{y})$ | Misclassification counts / Log Loss |
| **Core Algorithms** | Linear Regression, Ridge, Polynomial Reg. | Logistic Regression, Decision Trees, SVM |
