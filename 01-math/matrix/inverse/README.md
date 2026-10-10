# Matrix inverse

The inverse of A is written `A⁻¹`. It satisfies:

```
A · A⁻¹ = I
```

where `I` is the **identity matrix** (1s on the diagonal, 0s elsewhere):

```
I = [1  0]        I = [1  0  0]
    [0  1]            [0  1  0]
                      [0  0  1]
```

Multiplying by `I` leaves a matrix unchanged: `A * I = A`.

## Determinant (2×2)

```
A = [a  b]
    [c  d]

det(A) = ad - bc
```

A square matrix has an inverse **only if** `det(A) ≠ 0`.
If `det(A) = 0`, the matrix is **singular** and has no inverse.

Example:

```
A = [2  3]
    [1  4]

det(A) = 8 - 3 = 5 ≠ 0  → inverse exists
```

## Inverse formula (2×2)

```
A⁻¹ = (1 / (ad - bc)) * [ d  -b]
                         [-c   a]
```

Example:

```
A = [2  0]          A⁻¹ = [0.5   0   ]
    [0  4]                [0    0.25]

A * A⁻¹ = I
```

## In ML

Used for example in linear regression via the normal equation:

```
θ = (Xᵀ X)⁻¹ Xᵀ y
```
