# Gradient

The **gradient** is a vector containing all the partial derivatives of a multivariable function.

In Machine Learning, the gradient is the core of **Gradient Descent**:
- The gradient $\nabla f$ points in the direction of **steepest increase** (steepest ascent).
- The negative gradient $-\nabla f$ points in the direction of **steepest decrease** (steepest descent).
- To minimize a loss function, model parameters are updated via:
  $$\theta \leftarrow \theta - \alpha \nabla L(\theta)$$
  where $\alpha$ is the learning rate.

## Formula

For a function of 2 variables $f(x, y)$:
$$\nabla f(x, y) = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix}$$

For a function of $n$ variables $f(x_1, x_2, \dots, x_n)$:
$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

## Example

Given:
$$f(x, y) = x^2 + 3xy + y^2$$

1. Compute partial derivatives:
   $$\frac{\partial f}{\partial x} = 2x + 3y$$
   $$\frac{\partial f}{\partial y} = 3x + 2y$$

2. Construct the gradient vector:
   $$\nabla f(x, y) = \begin{bmatrix} 2x + 3y \\ 3x + 2y \end{bmatrix}$$

At $(x=2, y=3)$:
$$\nabla f(2, 3) = \begin{bmatrix} 2(2) + 3(3) \\ 3(2) + 2(3) \end{bmatrix} = \begin{bmatrix} 13 \\ 12 \end{bmatrix}$$
