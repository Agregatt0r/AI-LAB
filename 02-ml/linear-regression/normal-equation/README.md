# The Normal Equation (Analytical Solution)

Gradient Descent is an iterative optimization algorithm. However, for linear regression with Mean Squared Error, there is an exact, **closed-form analytical solution** known as the **Normal Equation**.

---

## 1. The Bias Trick (Design Matrix)

To combine weights $w$ and bias $b$ into a single vector $\boldsymbol{\theta}$, we prepend a column of $1$s to our feature matrix $\mathbf{X}$:

$$\mathbf{X}_{\text{aug}} = \begin{bmatrix} 1 & x_{1,1} & x_{1,2} & \dots & x_{1,d} \\ 1 & x_{2,1} & x_{2,2} & \dots & x_{2,d} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n,1} & x_{n,2} & \dots & x_{n,d} \end{bmatrix}, \quad \boldsymbol{\theta} = \begin{bmatrix} b \\ w_1 \\ w_2 \\ \vdots \\ w_d \end{bmatrix}$$

Now predictions are simply:
$$\hat{\mathbf{y}} = \mathbf{X}_{\text{aug}} \boldsymbol{\theta}$$

---

## 2. Derivation: Setting Gradients to Zero

The Mean Squared Error in matrix form is:
$$J(\boldsymbol{\theta}) = \frac{1}{n} (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})^T (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$$

Taking the matrix derivative with respect to $\boldsymbol{\theta}$ and setting it to $\mathbf{0}$:
$$\nabla_{\boldsymbol{\theta}} J = \frac{2}{n} \mathbf{X}^T (\mathbf{X}\boldsymbol{\theta} - \mathbf{y}) = \mathbf{0}$$

$$\mathbf{X}^T \mathbf{X} \boldsymbol{\theta} = \mathbf{X}^T \mathbf{y}$$

Multiplying both sides by the matrix inverse $(\mathbf{X}^T \mathbf{X})^{-1}$ (from `01-math/matrix/inverse`):

$$\boldsymbol{\theta}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

This formula gives the global minimum in a single algebraic calculation!

---

## 3. Normal Equation vs Gradient Descent

| Feature | Normal Equation | Gradient Descent |
|---|---|---|
| **Iterations** | None (solves in 1 step) | Many epochs ($10^2 - 10^5$) |
| **Learning Rate $\alpha$** | No hyperparameter to tune | Requires tuning $\alpha$ |
| **Feature Scaling** | Not needed | Mandatory for multi-features |
| **Computational Complexity** | $O(d^3)$ due to $(X^T X)^{-1}$ | $O(k \cdot n \cdot d)$ |
| **Scalability** | Slow when $d > 10,000$ features | Scales to billions of samples & features |
| **ML Use Case** | Small/medium tabular data | Deep learning, large-scale ML |
