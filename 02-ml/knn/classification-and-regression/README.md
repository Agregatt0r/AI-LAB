# KNN for Classification and Regression

Although K-Nearest Neighbors is most commonly taught as a classification algorithm, the exact same neighborhood mechanism applies naturally to **regression**. The only difference is how the targets of the $k$ neighbors are combined.

---

## 1. Classification with KNN

In classification, targets are discrete class labels ($y \in \{0, 1, \dots, C-1\}$).

### 1. Majority Voting (Mode)
The predicted class is the most frequent label among the $k$ neighbors:

$$\hat{y} = \arg\max_{c} \sum_{i \in N_k} \mathbb{1}[y_i = c]$$

### 2. Class Probability Estimation
KNN can output a calibrated probability for each class $c$:

$$P(y = c \mid \mathbf{x}) = \frac{1}{k} \sum_{i \in N_k} \mathbb{1}[y_i = c]$$

For example, if $k = 5$ and 4 of the neighbors are class 1 and 1 is class 0:
$$P(y = 1 \mid \mathbf{x}) = \frac{4}{5} = 0.80$$

### 3. Evaluation Metrics:
- **Accuracy**: $\frac{\text{Correct}}{\text{Total}}$
- **Precision, Recall, F1-Score**
- **Confusion Matrix**

---

## 2. Regression with KNN

In regression, targets are continuous real numbers ($y \in \mathbb{R}$, e.g. price, temperature, energy consumption).

### 1. Uniform Average
The prediction is the arithmetic mean of the targets of the $k$ neighbors:

$$\hat{y} = \frac{1}{k} \sum_{i \in N_k} y_i$$

### 2. Distance-Weighted Average (`weights='distance'`)
Neighbors that are extremely close to the query point should exert more influence than neighbors farther away:

$$w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i) + \epsilon}$$

$$\hat{y} = \frac{\sum_{i \in N_k} w_i y_i}{\sum_{i \in N_k} w_i}$$

where $\epsilon$ is a small constant (e.g. $10^{-5}$) to prevent division by zero when a query point coincides with a training point.

### 3. Evaluation Metrics:
- **MSE (Mean Squared Error)**: $\frac{1}{n}\sum (y - \hat{y})^2$
- **RMSE (Root Mean Squared Error)**: $\sqrt{\text{MSE}}$
- **MAE (Mean Absolute Error)**: $\frac{1}{n}\sum |y - \hat{y}|$
- **$R^2$ (Coefficient of Determination)**: $1 - \frac{\sum(y - \hat{y})^2}{\sum(y - \bar{y})^2}$

---

## 3. Side-by-Side Comparison

| Dimension | KNN Classifier | KNN Regressor |
|---|---|---|
| **Target Variable ($y$)** | Discrete categories ($y \in \{0, 1, \dots\}$) | Continuous values ($y \in \mathbb{R}$) |
| **Neighborhood Aggregation** | **Majority Vote** (Mode) | **Average** (Mean) |
| **Probability Support** | Yes: fraction of neighbors in class $c$ | No: outputs point estimate $\hat{y}$ |
| **Distance Weighting** | Weighted voting | Weighted averaging |
| **Prediction Shape** | Piecewise constant decision regions | Step-like (uniform) or smooth interpolation (weighted) |
| **Key Evaluation Metrics** | Accuracy, F1-Score, Confusion Matrix | MSE, RMSE, MAE, $R^2$ |

---

## Summary

- KNN Classification aggregates neighbors via **mode / majority vote** and outputs class probabilities.
- KNN Regression aggregates neighbors via **mean / weighted average** and outputs continuous estimates.
- Distance weighting ($w_i = 1/d_i$) gives closer points higher influence in both tasks.

**Previous:** [StandardScaler](../standard-scaler/README.md) · **Next:** [Complete KNN From Scratch](../from-scratch/README.md)

