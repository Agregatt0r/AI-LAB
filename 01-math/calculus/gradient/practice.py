# Practice: Gradient Vector
# Given f(x, y) = x^2 + 3xy + y^2
# ∇f(x, y) = [∂f/∂x, ∂f/∂y]^T = [2x + 3y, 3x + 2y]

def gradient(x, y):
    df_dx = 2 * x + 3 * y
    df_dy = 3 * x + 2 * y
    return [df_dx, df_dy]


if __name__ == "__main__":
    x, y = 2, 3
    grad = gradient(x, y)
    print(f"Gradient ∇f({x}, {y}) = {grad}")  # [13, 12]
