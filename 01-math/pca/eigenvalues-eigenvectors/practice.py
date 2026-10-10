# Practice: Eigenvalues & Eigenvectors of Covariance Matrix

C = [
    [5/3, 5/3],
    [5/3, 5/3],
]

# 1. Characteristic equation: det(C - λI) = 0
# λ² - trace*λ + det = 0
trace = C[0][0] + C[1][1]
det = C[0][0] * C[1][1] - C[0][1] * C[1][0]

d = (trace**2 - 4 * 1 * det) ** 0.5
lambda1 = (trace + d) / 2
lambda2 = (trace - d) / 2

print(f"λ1 = {lambda1:.4f} (10/3)")
print(f"λ2 = {lambda2:.4f} (0)")

# 2. Eigenvectors
# For λ1 = 10/3: (C - λ1*I)v = 0  ->  v1 = v2  ->  v = [1, 1]
v1_raw = [1, 1]
norm1 = (v1_raw[0]**2 + v1_raw[1]**2) ** 0.5
v1 = [v1_raw[0] / norm1, v1_raw[1] / norm1]

# For λ2 = 0: (C - λ2*I)v = 0  ->  v1 = -v2  ->  v = [-1, 1]
v2_raw = [-1, 1]
norm2 = (v2_raw[0]**2 + v2_raw[1]**2) ** 0.5
v2 = [v2_raw[0] / norm2, v2_raw[1] / norm2]

print("v1 (normalized):", v1)  # [1/√2, 1/√2]
print("v2 (normalized):", v2)  # [-1/√2, 1/√2]

# 3. Verification Cv = λv
def mat_vec(matrix, v):
    return [
        matrix[0][0] * v[0] + matrix[0][1] * v[1],
        matrix[1][0] * v[0] + matrix[1][1] * v[1],
    ]

print("\nVerify Cv1 == λ1*v1:")
print("  Cv1   =", [round(x, 4) for x in mat_vec(C, v1)])
print("  λ1*v1 =", [round(lambda1 * x, 4) for x in v1])
