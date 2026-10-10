# Principal Components & Explained Variance

Once eigenvalues and eigenvectors are found, we rank them:

$$\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d$$

## Principal Components

- **PC1 (First Principal Component):** The eigenvector corresponding to the largest eigenvalue $\lambda_1$. This is the direction of greatest variance in the data.
- **PC2 (Second Principal Component):** The eigenvector corresponding to $\lambda_2$. It is perpendicular (orthogonal) to PC1 and captures the second highest variance.

## Explained Variance Ratio

Each eigenvalue $\lambda_i$ represents the variance captured along that principal component.

The proportion of total variance explained by component $i$ is:

$$\text{Explained Variance Ratio}_i = \frac{\lambda_i}{\sum_{j=1}^d \lambda_j}$$

- If PC1 explains 95%+ of variance, we can often drop the remaining components with little information loss.
