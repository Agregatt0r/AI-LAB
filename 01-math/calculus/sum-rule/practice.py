# Practice: Polynomials and Sum Rule

# 1. Given f(x) = 4x^4 - 3x^2 + 7x
# Derivative f'(x) = 16x^3 - 6x + 7
def derivative_poly(x):
    return 16 * x**3 - 6 * x + 7


# 2. Given f(x) = x^3 + 5x^2 -> f'(x) = 3x^2 + 10x
def df_sum1(x):
    return 3 * x**2 + 10 * x


# 3. Given f(x) = x^5 + 2x^3 - 8 -> f'(x) = 5x^4 + 6x^2
def df_sum2(x):
    return 5 * x**4 + 6 * x**2


if __name__ == "__main__":
    x = 2
    print(f"f(x) = 4x^4 - 3x^2 + 7x at x={x} -> f'({x}) = {derivative_poly(x)}")  # 123
    print(f"f(x) = x^3 + 5x^2       at x={x} -> f'({x}) = {df_sum1(x)}")          # 32
    print(f"f(x) = x^5 + 2x^3 - 8   at x={x} -> f'({x}) = {df_sum2(x)}")          # 104
