# Practice: Power Rule and Constant Rule

# 1. f(x) = x^4 -> f'(x) = 4x^3
def df1(x):
    return 4 * x**3


# 2. f(x) = 3x^2 -> f'(x) = 6x
def df2(x):
    return 6 * x


# 3. f(x) = 5x^3 + 2x -> f'(x) = 15x^2 + 2
def df3(x):
    return 15 * x**2 + 2


# 4. f(x) = 7 -> f'(x) = 0
def df4(x):
    return 0


# General Power Rule helper: d/dx (c * x^n) = c * n * x^(n-1)
def power_rule_derivative(c, n, x):
    if n == 0:
        return 0
    return c * n * (x ** (n - 1))


if __name__ == "__main__":
    x = 2
    print(f"f(x) = x^4           at x={x} -> f'({x}) = {df1(x)}")  # 32
    print(f"f(x) = 3x^2         at x={x} -> f'({x}) = {df2(x)}")  # 12
    print(f"f(x) = 5x^3 + 2x    at x={x} -> f'({x}) = {df3(x)}")  # 62
    print(f"f(x) = 7            at x={x} -> f'({x}) = {df4(x)}")  # 0
