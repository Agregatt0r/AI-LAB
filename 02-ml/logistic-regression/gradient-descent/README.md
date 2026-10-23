# Gradient Descent for Classification

Gradient descent minimizes the BCE loss $J(\mathbf{w}, b)$ by repeatedly stepping the parameters in the direction of steepest descent. The procedure matches the one used for linear regression (see `../../linear-regression/gradient-descent`), but the gradient takes a remarkably simple form.

---

## 1. Loss Recap

$$J(\mathbf{w}, b) = -\frac{1}{n}\sum_{i=1}^{n}\Big[y_i \log p_i + (1-y_i)\log(1-p_i)\Big], \qquad p_i = \sigma(z_i),\quad z_i = \mathbf{w}^\top\mathbf{x}_i + b$$

---

## 2. Derivation of the Gradient

**Step 1: derivative of the loss with respect to $p_i$.**

$$\frac{\partial \ell_i}{\partial p_i} = -\frac{y_i}{p_i} + \frac{1 - y_i}{1 - p_i}$$

**Step 2: derivative of the sigmoid.** Using $\sigma'(z) = \sigma(z)(1-\sigma(z))$:

$$\frac{\partial p_i}{\partial z_i} = p_i(1 - p_i)$$

**Step 3: chain rule.** Multiply the two terms:

$$\frac{\partial \ell_i}{\partial z_i} = \left(-\frac{y_i}{p_i} + \frac{1 - y_i}{1 - p_i}\right) p_i(1 - p_i)$$

Expanding and simplifying:

$$= -y_i(1 - p_i) + (1 - y_i)p_i = p_i - y_i$$

The sigmoid derivative cancels the denominators in the log derivative. This is why the result is so clean.

**Step 4: chain rule to the parameters.** Since $\partial z_i / \partial \mathbf{w} = \mathbf{x}_i$ and $\partial z_i / \partial b = 1$:

$$\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{n}\sum_{i=1}^{n}(p_i - y_i)\,\mathbf{x}_i = \frac{1}{n}\mathbf{X}^\top(\mathbf{p} - \mathbf{y})$$

$$\frac{\partial J}{\partial b} = \frac{1}{n}\sum_{i=1}^{n}(p_i - y_i)$$

Compare with linear regression: the gradient has the same form, with the error $(\hat{y} - y)$ replaced by $(p - y)$.

---

## 3. Update Rule

$$\mathbf{w} \leftarrow \mathbf{w} - \alpha\,\frac{1}{n}\mathbf{X}^\top(\mathbf{p} - \mathbf{y})$$

$$b \leftarrow b - \alpha\,\frac{1}{n}\sum_{i=1}^{n}(p_i - y_i)$$

where $\alpha$ is the learning rate.

---

## 4. Algorithm

1. Initialize $\mathbf{w} = \mathbf{0}$, $b = 0$.
2. Compute probabilities $\mathbf{p} = \sigma(\mathbf{X}\mathbf{w} + b)$.
3. Compute errors $\mathbf{p} - \mathbf{y}$.
4. Compute gradients $\nabla_{\mathbf{w}} J$ and $\partial_b J$.
5. Update parameters.
6. Repeat until the loss stops decreasing or the epoch limit is reached.

---

## 5. Practical Notes

- **Feature scaling matters.** Features on very different scales make the gradient step size uneven across dimensions. Standardize features first (see `../from-scratch`).
- **Learning rate.** Too large causes oscillation; too small is slow. Track BCE per epoch to diagnose this.
- **Gradient check.** Compare the analytical gradient with a finite-difference estimate $\frac{J(\theta + h) - J(\theta - h)}{2h}$ to catch bugs. The practice script does this.

---

## Summary

- The gradient of BCE with respect to $z$ is $p - y$.
- $\nabla_{\mathbf{w}} J = \frac{1}{n}\mathbf{X}^\top(\mathbf{p} - \mathbf{y})$ and $\partial_b J = \frac{1}{n}\sum(p_i - y_i)$.
- Same update structure as linear regression, with $\hat{y}$ replaced by $p$.

**Previous:** [Binary Cross-Entropy](../binary-cross-entropy/README.md) · **Next:** [From Scratch](../from-scratch/README.md)
