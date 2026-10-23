# Decision Boundary

The **decision boundary** is the set of points where the model is exactly undecided between the two classes. For logistic regression it is a straight line (in 2D), a plane (in 3D), or a hyperplane (in higher dimensions).

---

## 1. Where the Boundary Is

The boundary occurs where the probability equals the threshold. With the default threshold $t = 0.5$:

$$\sigma(z) = 0.5 \iff z = 0$$

So the decision boundary is the set of points satisfying:

$$\mathbf{w}^\top \mathbf{x} + b = 0$$

---

## 2. Two Features

With $\mathbf{x} = (x_1, x_2)$ and $\mathbf{w} = (w_1, w_2)$:

$$w_1 x_1 + w_2 x_2 + b = 0$$

Solving for $x_2$:

$$x_2 = -\frac{w_1 x_1 + b}{w_2}$$

This is a straight line with slope $-\dfrac{w_1}{w_2}$ and intercept $-\dfrac{b}{w_2}$.

---

## 3. Geometry

- The weight vector $\mathbf{w}$ is **perpendicular** to the boundary. It points toward the class 1 side.
- The bias $b$ shifts the boundary away from the origin. Changing $b$ moves the line without rotating it.
- The distance from a point $\mathbf{x}$ to the boundary is $\dfrac{|\mathbf{w}^\top \mathbf{x} + b|}{\|\mathbf{w}\|}$. The further a point is from the line, the larger $|z|$ and the more confident the prediction.

---

## 4. Linear Boundaries Are a Limitation

Because $z$ is linear in $\mathbf{x}$, plain logistic regression can only separate classes with a linear boundary. Data such as concentric circles cannot be separated this way. Remedies include adding polynomial or interaction features, or using non-linear models.

---

## 5. Probability Versus Boundary

Points far from the boundary get probabilities near $0$ or $1$. Points close to it get probabilities near $0.5$. The boundary is therefore the line of maximal uncertainty.

---

## Summary

- The boundary is $\mathbf{w}^\top \mathbf{x} + b = 0$, the set where $\sigma(z) = 0.5$.
- In 2D it is a line $x_2 = -(w_1 x_1 + b)/w_2$.
- $\mathbf{w}$ is normal to the boundary; $b$ shifts it.

**Previous:** [Probability & Threshold](../probability-and-threshold/README.md) · **Next:** [Binary Cross-Entropy](../binary-cross-entropy/README.md)
