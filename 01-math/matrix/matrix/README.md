# Matrix

A **matrix** is a rectangular grid (array) of numbers arranged in rows and columns.

```
A = [
    [1, 2, 3],
    [4, 5, 6]
]
```

- 2 rows
- 3 columns
- shape = `(2, 3)` or `A ∈ R^(2×3)`

Without NumPy, a matrix in Python is just a list of lists:

```
A[0][0]  # 1
A[0][2]  # 3
A[1][1]  # 5

A[row][column]
```

## Matrix vs vector

- **Vector:** one row (or one column) of numbers
- **Matrix:** several rows

## Matrices in ML

In ML a matrix often represents a dataset:

```
X = [
    [height, weight, age],
    [height, weight, age],
    [height, weight, age],
    ...
]
```

- **rows:** observations / samples
- **columns:** features

Example:

```
X = [
    [176, 65, 20],
    [182, 80, 25],
    [170, 70, 22]
]
```

3 samples, 3 features → shape `(3, 3)`
