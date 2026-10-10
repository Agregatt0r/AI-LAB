# Derivative

A **derivative** measures the instantaneous rate of change of a function at a specific point. Geometrically, it represents the **slope of the tangent line** to the function's graph at that point.

## Geometric Interpretation

- **Secant line:** Slope between two distinct points $(x_1, y_1)$ and $(x_2, y_2)$:
  $$m_{\text{secant}} = \frac{f(x_2) - f(x_1)}{x_2 - x_1} = \frac{\Delta y}{\Delta x}$$

- **Tangent line:** When the distance between the two points approaches zero ($h \to 0$), the secant line becomes the tangent line at point $x$:
  $$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

## What the Derivative Tells Us

- $f'(x) > 0$: The function is **increasing** at $x$ (slope is positive).
- $f'(x) < 0$: The function is **decreasing** at $x$ (slope is negative).
- $f'(x) = 0$: Stationary point (potential local minimum, maximum, or saddle point).

## Examples

For $f(x) = x^2 \implies f'(x) = 2x$:
- At $x = 2$: $f'(2) = 2 \cdot 2 = 4$ $\to$ at $x = 2$, the function $x^2$ is increasing with a slope of $4$.
- At $x = 5$: $f'(5) = 2 \cdot 5 = 10$ $\to$ at $x = 5$, the function $x^2$ is increasing with a slope of $10$.

For $f(x) = x^3 \implies f'(x) = 3x^2$:
- At $x = 2$: $f'(2) = 3 \cdot 2^2 = 12$.
