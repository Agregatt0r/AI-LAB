# Chain Rule

The **Chain Rule** is a formula for computing the derivative of composite functions (functions inside functions).

In Deep Learning, **Backpropagation is the Chain Rule applied repeatedly** from the output layer back to the input layer to compute gradients with respect to every weight and bias.

## Formula

If $y = f(u)$ and $u = g(x)$, then:
$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

Or in prime notation:
$$(f(g(x)))' = f'(g(x)) \cdot g'(x)$$

### Power of a Function Rule
$$\frac{d}{dx}\left[ (g(x))^n \right] = n \cdot (g(x))^{n-1} \cdot g'(x)$$

## Example

Given:
$$f(x) = (3x^2 + 1)^4$$

1. **Outer function:** $u^4 \implies \text{derivative} = 4u^3 = 4(3x^2 + 1)^3$
2. **Inner function:** $u = 3x^2 + 1 \implies \text{derivative} = 6x$
3. **Multiply outer by inner:**
   $$f'(x) = 4(3x^2 + 1)^3 \cdot 6x = 24x(3x^2 + 1)^3$$
