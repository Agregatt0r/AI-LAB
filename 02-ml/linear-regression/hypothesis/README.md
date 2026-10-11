# Linear Hypothesis

The **hypothesis** in Linear Regression is the function that models the relationship between the independent input variable ($x$) and the dependent target variable ($y$).

In Simple Linear Regression (one feature), the hypothesis is a straight line:

$$\hat{y} = wx + b$$

where:
- $x$ is the input feature (e.g., house size in $\text{m}^2$).
- $w$ is the **weight** (or slope), determining the angle of the line and how strongly $x$ influences $y$.
- $b$ is the **bias** (or intercept), shifting the line up or down and representing the base predicted value when $x = 0$.
- $\hat{y}$ (read "y-hat") is the **predicted value**.
- $y$ is the **actual ground-truth target** value.

---

## Geometric Intuition

A straight line is completely determined by two parameters:
- Changing $w$ rotates the line around the vertical intercept.
- Changing $b$ shifts the line up or down vertically without changing its slope.

```
       y (Price)
       ^             /  (ŷ = wx + b)
       |            /
       |           /
       |          /
       |   b ----/
       |        /
       +-------------------> x (Size)
```

---

## Example Calculation

Suppose we model house prices:
- $w = 3500$ (each square meter adds \$3,500 to the price)
- $b = 10000$ (base plot value is \$10,000)

For a house of size $x = 60\text{ m}^2$:

$$\hat{y} = (3500 \times 60) + 10000 = 210000 + 10000 = \$220,000$$

---

## Summary for ML

The hypothesis function is the **forward pass** of linear regression:
$$\text{Input } x \xrightarrow{\hat{y} = wx + b} \text{Prediction } \hat{y}$$
Training a model is nothing more than finding the optimal values of $w$ and $b$ so that $\hat{y}$ is as close to actual $y$ as possible across all training examples.
