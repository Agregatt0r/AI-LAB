# Residuals and Gradients: The Mathematical Core of GBM

Gradient Boosting (Friedman, 2001) unifies the entire boosting framework under a single elegant principle: **gradient descent in function space**. Instead of updating model *parameters* as in standard gradient descent, Gradient Boosting updates the *prediction function* $F$ itself, one weak learner at a time.

---

## 🧮 Function Space Gradient Descent

Standard parameter-space gradient descent minimizes a loss $L(\theta)$ over model parameters $\theta$:

$$\theta^{(t+1)} = \theta^{(t)} - \eta \cdot \nabla_\theta L(\theta^{(t)})$$

Gradient Boosting applies the same logic but minimizes the **empirical risk** over prediction functions $F$:

$$\mathcal{L}[F] = \sum_{i=1}^{N} L\left(y_i, F(x_i)\right)$$

The "gradient" is now a vector of partial derivatives with respect to the current prediction at each training point:

$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{m-1}}$$

This vector $\mathbf{r}_m = (r_{1m}, r_{2m}, \dots, r_{Nm})$ is called the **pseudo-residuals** at stage $m$.

---

## 📐 The GBM Algorithm (Friedman's Original)

```
Algorithm: GradientBoostingMachine(X, y, loss L, M steps, learning rate nu)
------------------------------------------------------------------------
1. Initialize with the optimal constant prediction:
      F_0(x) = argmin_gamma sum_i L(y_i, gamma)
      e.g. for MSE: F_0 = mean(y)
      e.g. for log-loss: F_0 = log(p / (1 - p)) where p = mean(y)

2. For m = 1 to M:
      a. Compute pseudo-residuals (negative gradient):
            r_im = -[dL(y_i, F(x_i)) / dF(x_i)] evaluated at F = F_{m-1}

      b. Fit a weak learner h_m to the residuals:
            h_m = argmin_h sum_i (r_im - h(x_i))^2

      c. Find optimal step size (leaf-wise for trees):
            gamma_m = argmin_gamma sum_i L(y_i, F_{m-1}(x_i) + gamma * h_m(x_i))

      d. Update the ensemble:
            F_m(x) = F_{m-1}(x) + nu * gamma_m * h_m(x)

3. Return F_M
```

---

## 🔬 Pseudo-Residuals for Common Loss Functions

### 1. MSE Loss (Regression)

$$L(y, F) = \frac{1}{2}(y - F)^2$$

$$r_i = -\frac{\partial L}{\partial F} = -(F - y_i) = y_i - F(x_i)$$

For MSE, pseudo-residuals are simply the **ordinary residuals** $y - \hat{y}$. Each new tree fits the remaining unexplained portion of the target.

---

### 2. Log-Loss / Deviance (Binary Classification)

$$L(y, F) = -y \cdot \log p - (1-y) \cdot \log(1-p), \quad p = \sigma(F) = \frac{1}{1 + e^{-F}}$$

$$r_i = -\frac{\partial L}{\partial F} = y_i - p_i = y_i - \sigma(F(x_i))$$

For log-loss, pseudo-residuals equal the **difference between label and predicted probability** — equivalent to fitting each tree to the probability residuals.

---

### 3. Absolute Error / MAE (Robust Regression)

$$L(y, F) = |y - F|$$

$$r_i = -\frac{\partial L}{\partial F} = \text{sign}(y_i - F(x_i))$$

Pseudo-residuals become $\pm 1$ signs — this produces a **robust** variant insensitive to outliers.

---

## 🌉 Why Fitting to Pseudo-Residuals Works

The fundamental insight: **fitting to the negative gradient is equivalent to taking one step in function space towards lower loss**:

$$F_m = F_{m-1} + \nu \cdot h_m \approx F_{m-1} - \nu \cdot \nabla_F \mathcal{L}$$

where $h_m(x)$ approximates $r_{im}$ at training points and **generalizes** the gradient to unseen test points via the tree's learned splits.

This is exactly what makes GBM so general: by swapping the loss function $L$, you change which residuals are computed, and the algorithm automatically adapts to minimize that custom objective — without changing the tree-fitting procedure at all.

