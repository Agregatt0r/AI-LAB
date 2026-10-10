# Matrix × vector

A vector here is one column of numbers. You can compute `A × x` when
**columns of A = length of x**.

The result is a vector: each entry is the dot product of one row of A with x.

```
A = [
    [1, 2],
    [3, 4]
]

x = [5, 6]

A × x = [1*5 + 2*6, 3*5 + 4*6] = [17, 39]
```

## In ML

One sample `x` and weights `w` give a prediction via a dot product
(same idea as one row of the feature matrix times `w`):

```
x = [height, weight, age]
w = [w1, w2, w3]

y = w1*x1 + w2*x2 + w3*x3
```
