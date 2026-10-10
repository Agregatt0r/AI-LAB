# Practice: Principal Components and Explained Variance

# Eigenvalues and normalized eigenvectors
lambdas = [10/3, 0.0]
eigenvectors = [
    [0.70710678, 0.70710678],   # for λ = 10/3
    [-0.70710678, 0.70710678],  # for λ = 0
]

# 1. Rank principal components (largest lambda first)
pairs = sorted(zip(lambdas, eigenvectors), key=lambda x: x[0], reverse=True)

PC1 = pairs[0][1]
PC2 = pairs[1][1]

print("PC1 (primary direction):", PC1, f"with λ1 = {pairs[0][0]:.4f}")
print("PC2 (secondary direction):", PC2, f"with λ2 = {pairs[1][0]:.4f}")

# 2. Explained Variance Ratio
total_variance = sum(lambdas)
var_pc1 = pairs[0][0] / total_variance if total_variance > 0 else 0
var_pc2 = pairs[1][0] / total_variance if total_variance > 0 else 0

print(f"\nPC1 explains: {var_pc1 * 100:.1f}% of total variance")
print(f"PC2 explains: {var_pc2 * 100:.1f}% of total variance")
