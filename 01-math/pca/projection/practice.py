# Practice: Projecting Centered Data onto Principal Components

X_centered = [
    [-1.5, -1.5],
    [-0.5, -0.5],
    [ 0.5,  0.5],
    [ 1.5,  1.5]
]

# PC1 is unit eigenvector [1/√2, 1/√2]
PC1 = [0.70710678, 0.70710678]
PC2 = [-0.70710678, 0.70710678]

# 1. Project onto PC1 (Reduction 2D -> 1D)
X_reduced = [
    point[0] * PC1[0] + point[1] * PC1[1]
    for point in X_centered
]

# 2. Project onto PC2 (Should be ~0 since lambda2 = 0)
X_pc2 = [
    point[0] * PC2[0] + point[1] * PC2[1]
    for point in X_centered
]

print("1D coordinates along PC1:")
print([round(val, 3) for val in X_reduced])
# [-2.121, -0.707, 0.707, 2.121]

print("\n1D coordinates along PC2 (noise/orthogonal axis):")
print([round(val, 3) for val in X_pc2])
# [0.0, 0.0, 0.0, 0.0]
