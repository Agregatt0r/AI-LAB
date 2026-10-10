# Data Centering

Before applying PCA, the data must be **centered** so that the origin `(0, 0, ...)` is located at the center (mean) of the data cloud.

## Why center the data?

PCA measures variance around the origin. If data is not centered, the first principal component will point toward the mean of the data rather than the direction of maximum spread.

## Formula

1. **Calculate mean for each feature:**
   $$\mu_j = \frac{1}{n} \sum_{i=1}^{n} X_{ij}$$

2. **Subtract mean from every sample:**
   $$X_{\text{centered}, ij} = X_{ij} - \mu_j$$

## Example

```
Original X:
[2, 1]
[3, 2]
[4, 3]
[5, 4]

Mean: [3.5, 2.5]

Centered X:
[2 - 3.5, 1 - 2.5] = [-1.5, -1.5]
[3 - 3.5, 2 - 2.5] = [-0.5, -0.5]
[4 - 3.5, 3 - 2.5] = [ 0.5,  0.5]
[5 - 3.5, 4 - 2.5] = [ 1.5,  1.5]
```
