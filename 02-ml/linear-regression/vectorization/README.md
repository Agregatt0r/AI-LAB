# Vectorization with NumPy

In machine learning, datasets contain thousands or millions of samples. Writing Python `for` loops over each individual row is too slow for real-world models.

**Vectorization** is the practice of replacing explicit loops with array and matrix operations executed in optimized C/Fortran code via libraries like **NumPy**.

---

## 1. Loop vs Vectorized Comparison

### Forward Predictions ($\hat{\mathbf{y}} = w\mathbf{x} + b$)

```python
# Slow: Pure Python loop
y_pred = []
for val in x:
    y_pred.append(w * val + b)

# Fast: NumPy vectorization (SIMD parallelized)
y_pred = w * x + b
```

### Mean Squared Error ($\text{MSE}$)

```python
# Slow: Pure Python
mse = sum((y[i] - y_pred[i]) ** 2 for i in range(n)) / n

# Fast: NumPy
mse = np.mean((y - y_pred) ** 2)
```

---

## 2. Vectorized Gradients

For a dataset represented as NumPy vectors:
- Input vector: $\mathbf{x} \in \mathbb{R}^n$
- Target vector: $\mathbf{y} \in \mathbb{R}^n$
- Predictions: $\hat{\mathbf{y}} = w\mathbf{x} + b \in \mathbb{R}^n$
- Error vector: $\mathbf{e} = \hat{\mathbf{y}} - \mathbf{y} \in \mathbb{R}^n$

The partial derivatives can be computed using vector dot products and sums:

$$\frac{\partial J}{\partial w} = \frac{2}{n} \mathbf{e} \cdot \mathbf{x} = \frac{2}{n} \sum_{i=1}^n e_i x_i = \frac{2}{n} \mathbf{x}^T (\hat{\mathbf{y}} - \mathbf{y})$$

$$\frac{\partial J}{\partial b} = \frac{2}{n} \sum_{i=1}^n e_i = \frac{2}{n} \text{np.sum}(\hat{\mathbf{y}} - \mathbf{y})$$

In NumPy code:
```python
errors = y_pred - y
dw = (2 / n) * np.dot(errors, x)
db = (2 / n) * np.sum(errors)
```

---

## 3. Why Vectorization Matters

1. **Massive Speedup**: NumPy arrays use contiguous C memory and hardware SIMD (Single Instruction Multiple Data) registers, yielding 10x - 100x speedups.
2. **Cleaner, Declarative Code**: Eliminates indexing boilerplate (`[i]`, `range(n)`) and index out-of-bounds bugs.
3. **Bridge to Linear Algebra**: Connects directly to the matrix operations studied in `01-math/matrix`.
