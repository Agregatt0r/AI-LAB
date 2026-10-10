# Practice: Chain Rule
# Given f(x) = (3x^2 + 1)^4
# Find f'(x) = 4 * (3x^2 + 1)^3 * (6x) = 24x * (3x^2 + 1)^3

def chain_rule(x):
    outer_derivative = 4 * (3 * x**2 + 1)**3
    inner_derivative = 6 * x
    return outer_derivative * inner_derivative


if __name__ == "__main__":
    x = 2
    # At x = 2:
    # 3(2)^2 + 1 = 13
    # 4 * 13^3 * 12 = 48 * 2197 = 105456
    result = chain_rule(x)
    print(f"f(x) = (3x^2 + 1)^4 at x={x} -> f'({x}) = {result}")  # 105456
