# PCA Overview

**PCA (Principal Component Analysis)** finds the axes along which data varies the most.

Those directions are called **principal components**.

```
High-dimensional data
         ↓
Find axes of greatest variance (eigenvectors of covariance matrix)
         ↓
Keep top k strongest components
         ↓
Lower-dimensional data (minimal information loss)
```

## Why we need PCA

1. **Dimensionality reduction:** Compress data (e.g. 100 features → 10 features).
2. **Data visualization:** Project high-dimensional data onto 2D or 3D.
3. **Noise reduction:** Discard low-variance dimensions which often contain noise.
4. **Multicollinearity removal:** Principal components are strictly orthogonal (uncorrelated).

## The 5-Step Pipeline

1. **Center the data:** Subtract the mean of each feature ($\mu$).
2. **Compute the covariance matrix:** $C = \frac{1}{n-1} X_{\text{centered}}^T X_{\text{centered}}$.
3. **Find eigenvalues and eigenvectors:** Solve $\det(C - \lambda I) = 0$ and $(C - \lambda I)v = 0$.
4. **Rank components:** Sort eigenvectors by eigenvalues in descending order ($\lambda_1 \ge \lambda_2 \ge \dots$).
5. **Project data:** Compute $X_{\text{reduced}} = X_{\text{centered}} \cdot V_k$.
