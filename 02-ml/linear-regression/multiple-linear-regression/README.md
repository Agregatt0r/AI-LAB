# Multiple Linear Regression

Real-world problems involve more than a single input feature. For example, predicting a house price usually relies on multiple attributes: size, number of bedrooms, age of the property, and distance to city center.

---

## 1. Mathematical Formulation

For $d$ input features $(x_1, x_2, \dots, x_d)$:

$$\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b = \mathbf{w}^T \mathbf{x} + b$$

where:
- $\mathbf{x} = [x_1, x_2, \dots, x_d]^T \in \mathbb{R}^d$ is the feature vector of a single sample.
- $\mathbf{w} = [w_1, w_2, \dots, w_d]^T \in \mathbb{R}^d$ is the weight vector.
- $b \in \mathbb{R}$ is the scalar bias (intercept).

### Matrix Form for $n$ Samples:
$$\mathbf{X} \in \mathbb{R}^{n \times d}, \quad \mathbf{y} \in \mathbb{R}^n, \quad \mathbf{w} \in \mathbb{R}^d$$

$$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$$

---

## 2. Vectorized Gradients

The cost function is:
$$J(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2 = \frac{1}{n} \|\hat{\mathbf{y}} - \mathbf{y}\|^2$$

Taking matrix derivatives:
$$\nabla_{\mathbf{w}} J = \frac{2}{n} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y}) \in \mathbb{R}^d$$
$$\frac{\partial J}{\partial b} = \frac{2}{n} \sum_{i=1}^n (\hat{y}_i - y_i) \in \mathbb{R}$$

Parameters are updated simultaneously:
$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \nabla_{\mathbf{w}} J$$
$$b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$

---

## 3. Why Feature Scaling is Essential

When features have different scales (e.g., $x_1 \in [50, 200]\text{ m}^2$ and $x_2 \in [1, 5]\text{ bedrooms}$):
- The loss surface becomes a narrow, stretched elliptical valley.
- Gradient descent oscillates perpendicular to the gradient and takes thousands of epochs to converge.

By standardizing features ($Z$-score normalization):
$$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$
the loss contours become symmetrical circles, allowing gradient descent to march directly downhill in far fewer epochs.
