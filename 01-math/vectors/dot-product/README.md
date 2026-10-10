# Dot product

Multiply corresponding components, then sum. The result is **one number**.

```
a = [a1, a2, a3]
b = [b1, b2, b3]

a · b = a1 * b1 + a2 * b2 + a3 * b3
```

Example:

```
a = [2, 3, 4]
b = [5, 1, 2]

a · b = 2 * 5 + 3 * 1 + 4 * 2 = 21
```

Prefer `A · B` or `dot(A, B)` over `A * B` (`*` looks like element-wise multiply).

Dot product is very important for AI, because it turns a set of features and weights
into 1 number:

```
features = [x1, x2, x3]
weights = [w1, w2, w3]

dot(features, weights) = x1 * w1 + x2 * w2 + x3 * w3
```

It is a foundation of Linear Regression, Logistic Regression, and neural networks.
