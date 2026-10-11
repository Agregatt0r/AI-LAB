# Train/Test Split & Model Evaluation

If we train a model on all available data and evaluate it on that exact same data, we cannot tell whether the model has truly learned the underlying relationship or simply memorized the training examples.

The true goal of Machine Learning is **generalization**—the ability to make accurate predictions on new, unseen data.

---

## 1. Generalization, Underfitting, and Overfitting

```
    Underfitting (High Bias)       Good Fit (Balanced)         Overfitting (High Variance)
        y ^    /                     y ^     _---_                 y ^   _/\_
          |   /                        |   /       \                 |  /    \/\_
          |  /   *   *                 |  /  *   *  \                | /*  *  \  *
          | /  *       *               | /*         *\               |/       \
          +--------------> x           +--------------> x            +--------------> x
       Too simple to capture         Captures true underlying       Memorizes noise; fails
          the true pattern                   signal                     on new data
```

---

## 2. Dataset Splitting

We partition the raw dataset into distinct subsets:
- **Training Set (typically 80%)**: Used by the optimization algorithm to update parameters ($w, b$).
- **Test Set (typically 20%)**: Held out completely during training. Evaluated only once to estimate performance on unseen future data.

> **Important**: Always shuffle the dataset randomly before splitting to prevent ordering bias (e.g., if data is sorted by date or price).

---

## 3. Evaluation Metrics for Regression

### Mean Squared Error (MSE)
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

### Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\text{MSE}}$$
Interpretable in the original units of $y$ (e.g. dollars, kilograms).

### Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
The average magnitude of errors, robust against outliers.

### $R^2$ Score (Coefficient of Determination)
Quantifies what percentage of variance in the target $y$ is explained by the model:

$$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

where $\bar{y}$ is the mean of true targets.
- $R^2 = 1.0$: Perfect predictions.
- $R^2 = 0.0$: Model performs no better than simply predicting the baseline average $\bar{y}$.
- $R^2 < 0.0$: Model performs worse than the simple average baseline.
