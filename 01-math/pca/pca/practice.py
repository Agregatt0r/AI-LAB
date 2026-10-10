# Practice: Complete End-to-End PCA Pipeline (2D -> 1D)

# 0. Raw Dataset
X = [
    [2, 1],
    [3, 2],
    [4, 3],
    [5, 4],
]
n = len(X)

# 1. Mean
mean_x = sum(row[0] for row in X) / n
mean_y = sum(row[1] for row in X) / n

# 2. Centering
X_centered = [[row[0] - mean_x, row[1] - mean_y] for row in X]

# 3. Covariance Matrix
xs = [row[0] for row in X_centered]
ys = [row[1] for row in X_centered]

cov_xx = sum(xs[i] * xs[i] for i in range(n)) / (n - 1)
cov_xy = sum(xs[i] * ys[i] for i in range(n)) / (n - 1)
cov_yy = sum(ys[i] * ys[i] for i in range(n)) / (n - 1)
C = [
    [cov_xx, cov_xy],
    [cov_xy, cov_yy],
]

# 4. Eigenvalues & Eigenvectors
# det(C - λI) = 0 -> λ² - trace*λ + det = 0
trace = C[0][0] + C[1][1]
det = C[0][0] * C[1][1] - C[0][1] * C[1][0]
d = (trace**2 - 4 * det) ** 0.5

lambda1 = (trace + d) / 2  # 10/3 ≈ 3.3333
lambda2 = (trace - d) / 2  # 0.0

# PC1 corresponds to lambda1:
v1 = [1, 1]
norm1 = (v1[0]**2 + v1[1]**2) ** 0.5
PC1 = [v1[0] / norm1, v1[1] / norm1]  # [1/√2, 1/√2]

# 5. Projection (Dimensionality Reduction to 1D)
X_reduced = [point[0] * PC1[0] + point[1] * PC1[1] for point in X_centered]

print("Original 2D data:", X)
print("Mean:", [mean_x, mean_y])
print("Centered data:", X_centered)
print("Covariance Matrix:", C)
print(f"λ1 = {lambda1:.4f}, λ2 = {lambda2:.4f}")
print("PC1:", PC1)
print("1D Reduced projection:", [round(val, 3) for val in X_reduced])
