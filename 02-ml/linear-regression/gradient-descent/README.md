# Gradient Descent for Linear Regression

Instead of randomly guessing values for $w$ and $b$, **Gradient Descent** provides an iterative algorithm to systematically find the parameters that minimize the Cost Function $\text{MSE}(w, b)$.

---

## 1. Mathematical Derivation of Gradients

Recall the loss function:
$$J(w, b) = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2 = \frac{1}{n} \sum_{i=1}^n (wx_i + b - y_i)^2$$

Using the **Chain Rule** from calculus:
$$\frac{\partial}{\partial u} u^2 = 2u \quad \text{where } u = (wx_i + b - y_i) = (\hat{y}_i - y_i)$$

### Gradient with respect to $w$:
$$\frac{\partial J}{\partial w} = \frac{1}{n} \sum_{i=1}^n 2(\hat{y}_i - y_i) \cdot \frac{\partial (wx_i + b - y_i)}{\partial w}$$
$$\frac{\partial J}{\partial w} = \frac{2}{n} \sum_{i=1}^n (\hat{y}_i - y_i) x_i$$

### Gradient with respect to $b$:
$$\frac{\partial J}{\partial b} = \frac{1}{n} \sum_{i=1}^n 2(\hat{y}_i - y_i) \cdot \frac{\partial (wx_i + b - y_i)}{\partial b}$$
$$\frac{\partial J}{\partial b} = \frac{2}{n} \sum_{i=1}^n (\hat{y}_i - y_i)$$

---

## 2. Parameter Update Rule

At each step (epoch), we update $w$ and $b$ in the direction of the **negative gradient** (steepest decrease):

$$w \leftarrow w - \alpha \frac{\partial J}{\partial w}$$
$$b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$

where $\alpha$ (alpha) is the **learning rate**.

---

## 3. The Role of the Learning Rate $\alpha$

- **Too large ($\alpha > 1.0$)**: The step overshoots the minimum and the cost oscillates or diverges ($\text{MSE} \to \infty$).
- **Too small ($\alpha = 0.000001$)**: The model makes tiny adjustments and requires millions of iterations to converge.
- **Well tuned ($\alpha \approx 0.01 - 0.1$)**: Steady, monotonic decrease in MSE toward the global minimum.

```
       MSE ^
           |     \               /
           |      \   *         /
           |       \   \       /
           |        \   *     /
           |         \   \   /
           |          \___*_/
           +---------------------> w, b
               Stepping downhill
```

---

## 4. The Algorithm Loop

1. Initialize $w = 0$ (or random), $b = 0$.
2. For each epoch:
   a. Compute predictions $\hat{y}_i = wx_i + b$.
   b. Compute prediction errors $e_i = \hat{y}_i - y_i$.
   c. Compute gradients $\frac{\partial J}{\partial w}$ and $\frac{\partial J}{\partial b}$.
   d. Update $w \leftarrow w - \alpha \frac{\partial J}{\partial w}$ and $b \leftarrow b - \alpha \frac{\partial J}{\partial b}$.
3. Stop when the maximum number of epochs is reached or when the gradient magnitude is smaller than a tolerance threshold $\epsilon$.
