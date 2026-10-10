# Projection & Dimensionality Reduction

The final step of PCA is **projecting** the centered data points onto the chosen principal component(s).

## Projection Formula

For a centered data point $x_{\text{centered}} = [x_1, x_2, \dots, x_d]$ and a unit principal component vector $PC = [v_1, v_2, \dots, v_d]$:

$$z = x_{\text{centered}} \cdot PC = x_1 v_1 + x_2 v_2 + \dots + x_d v_d$$

Here $z$ is a single scalar representing the coordinate of the point along the new axis $PC$.

## Matrix Projection

To reduce an $n \times d$ centered matrix $X_{\text{centered}}$ to $k$ dimensions using top $k$ principal components $V_k = [PC_1, PC_2, \dots, PC_k]$ ($d \times k$ matrix):

$$X_{\text{reduced}} = X_{\text{centered}} \cdot V_k$$

The resulting matrix $X_{\text{reduced}}$ has shape $n \times k$.

## Reconstruction (Approximation)

To approximate original points back in original $d$-dimensional space:

$$X_{\text{reconstructed}} = X_{\text{reduced}} \cdot V_k^T + \mu$$
