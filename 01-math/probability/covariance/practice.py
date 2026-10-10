# Practice: Covariance between two variables
# Formula: Cov(X, Y) = (1/N) * sum((x_i - μ_X) * (y_i - μ_Y))

def mean(data):
    return sum(data) / len(data)


def covariance(x, y):
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    mu_x = mean(x)
    mu_y = mean(y)
    return sum((xi - mu_x) * (yi - mu_y) for xi, yi in zip(x, y)) / len(x)


x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

mu_x = mean(x)
mu_y = mean(y)
cov_xy = covariance(x, y)

print("X:", x)
print("Y:", y)
print(f"Mean X (μ_X): {mu_x}")
print(f"Mean Y (μ_Y): {mu_y}")
print(f"Covariance Cov(X, Y): {cov_xy}")  # 4.0
