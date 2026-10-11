# Loss Function: Mean Squared Error (MSE)

When a model makes predictions $\hat{y}$, they rarely match actual targets $y$ perfectly on day one. A **loss function** (or cost function) quantifies how wrong our predictions are by returning a single non-negative penalty number.

---

## 1. Residuals

The error on a single data point $i$ is called the **residual**:

$$e_i = y_i - \hat{y}_i$$

- If $e_i > 0$, the model under-predicted.
- If $e_i < 0$, the model over-predicted.
- If $e_i = 0$, the prediction was exact.

---

## 2. Mean Squared Error (MSE)

Instead of averaging simple residuals (which would cancel each other out because positives and negatives cancel to 0), we square the differences:

$$J(w, b) = \text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \frac{1}{n} \sum_{i=1}^n (y_i - (wx_i + b))^2$$

where:
- $n$ is the total number of training samples.
- $(y_i - \hat{y}_i)^2$ squares every residual.

### Why square the errors?
1. **Always positive**: Errors in either direction contribute positively to total loss.
2. **Penalizes large errors heavily**: An error of $10$ adds $100$ to the loss, whereas an error of $2$ adds only $4$.
3. **Smooth and differentiable**: Squaring creates a continuous, bowl-shaped (convex) parabolic surface whose derivative exists everywhere, making it ideal for gradient descent calculus.

---

## 3. Related Metrics

- **RMSE (Root Mean Squared Error)**:
  $$\text{RMSE} = \sqrt{\text{MSE}}$$
  Brings the error back to the original units of $y$ (e.g., dollars instead of dollars squared).
- **MAE (Mean Absolute Error)**:
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
  Linear penalty, less sensitive to outliers.

---

## 4. Parameter Evaluation Example

Given $x = [1, 2, 3]$ and $y = [3, 5, 7]$:

| Candidate $(w, b)$ | Predictions $\hat{y}$ | Residuals $(y - \hat{y})$ | Squared Errors | MSE |
|---|---|---|---|---|
| $(1, 1)$ | $[2, 3, 4]$ | $[1, 2, 3]$ | $[1, 4, 9]$ | $\frac{14}{3} \approx 4.67$ |
| $(2, 1)$ | $[3, 5, 7]$ | $[0, 0, 0]$ | $[0, 0, 0]$ | **$0.0$** (Optimal!) |
| $(3, 0)$ | $[3, 6, 9]$ | $[0, -1, -2]$ | $[0, 1, 4]$ | $\frac{5}{3} \approx 1.67$ |

The parameter pair with the lowest MSE provides the line of best fit.
