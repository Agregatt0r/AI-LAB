# Task 1: Scalar multiplication
# Given vector a, calculate 3 * a
# Write a function without numpy


def scalar_multiplication(a, k):
    return [k * x for x in a]


a = [2, 4, 6]

scalar = scalar_multiplication(a, 3)

print("3 * a =", scalar)  # [6, 12, 18]
