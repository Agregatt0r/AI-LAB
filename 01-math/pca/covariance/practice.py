# Practice: Computing Covariance Matrix

X_centered = [
    [-1.5, -1.5],
    [-0.5, -0.5],
    [ 0.5,  0.5],
    [ 1.5,  1.5]
]
n = len(X_centered)

xs = [row[0] for row in X_centered]
ys = [row[1] for row in X_centered]

# Cov(x, y) = sum(x_i * y_i) / (n - 1)
cov_xx = sum(xs[i] * xs[i] for i in range(n)) / (n - 1)
cov_xy = sum(xs[i] * ys[i] for i in range(n)) / (n - 1)
cov_yx = cov_xy
cov_yy = sum(ys[i] * ys[i] for i in range(n)) / (n - 1)

C = [
    [cov_xx, cov_xy],
    [cov_yx, cov_yy],
]

print("Cov(x,x):", cov_xx)  # 5/3 ≈ 1.6667
print("Cov(x,y):", cov_xy)  # 5/3 ≈ 1.6667
print("Cov(y,y):", cov_yy)  # 5/3 ≈ 1.6667

print("\nCovariance Matrix C:")
for row in C:
    print(row)
