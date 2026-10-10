# Practice: Partial Derivatives
# Given f(x, y) = x^2 + 3xy + y^2

# 1. Partial derivative with respect to x: ∂f/∂x = 2x + 3y
def partial_derivative_x(x, y):
    return 2 * x + 3 * y


# 2. Partial derivative with respect to y: ∂f/∂y = 3x + 2y
def partial_derivative_y(x, y):
    return 3 * x + 2 * y


if __name__ == "__main__":
    x, y = 2, 3
    df_dx = partial_derivative_x(x, y)
    df_dy = partial_derivative_y(x, y)

    print(f"f(x, y) = x^2 + 3xy + y^2 at ({x}, {y}):")
    print(f"∂f/∂x = 2({x}) + 3({y}) = {df_dx}")  # 13
    print(f"∂f/∂y = 3({x}) + 2({y}) = {df_dy}")  # 12
