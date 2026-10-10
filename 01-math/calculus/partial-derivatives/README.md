# Partial Derivatives

In Machine Learning, loss functions almost always depend on many variables (e.g. weights $w_1, w_2, \dots, w_n$ and bias $b$).

A **partial derivative** measures how a multivariable function changes as one variable changes while all other variables are kept constant.

## Notation

For a function $f(x, y)$:
- Partial derivative with respect to $x$: $\frac{\partial f}{\partial x}$ or $f_x$
- Partial derivative with respect to $y$: $\frac{\partial f}{\partial y}$ or $f_y$

## The Rule

- When computing $\frac{\partial f}{\partial x}$: treat $y$ as a constant number.
- When computing $\frac{\partial f}{\partial y}$: treat $x$ as a constant number.

## Example

Given:
$$f(x, y) = x^2 + 3xy + y^2$$

1. **With respect to $x$:**
   $$\frac{\partial f}{\partial x} = \frac{\partial}{\partial x}(x^2) + \frac{\partial}{\partial x}(3xy) + \frac{\partial}{\partial x}(y^2) = 2x + 3y + 0 = 2x + 3y$$

2. **With respect to $y$:**
   $$\frac{\partial f}{\partial y} = \frac{\partial}{\partial y}(x^2) + \frac{\partial}{\partial y}(3xy) + \frac{\partial}{\partial y}(y^2) = 0 + 3x + 2y = 3x + 2y$$
