# Practice: Derivative as Slope / Rate of Change

# 1. Given f(x) = x^3, find f'(2)
def derivative_cube(x):
    return 3 * x**2


print("f(x) = x^3 at x = 2 -> f'(2) =", derivative_cube(2))  # 12


# 2. Given f(x) = x^2, find f'(2) and f'(5)
def derivative_square(x):
    return 2 * x


slope_2 = derivative_square(2)  # 4
slope_5 = derivative_square(5)  # 10

print(f"f(x) = x^2 at x = 2 -> f'(2) = {slope_2}")  # 4
print(f"f(x) = x^2 at x = 5 -> f'(5) = {slope_5}")  # 10

# Meaning:
# f'(2) = 4 means at x = 2 the curve increases with slope 4
# f'(5) = 10 means at x = 5 the curve increases with slope 10
