# Task 1: Dot product
# Given two vectors a and b, calculate a · b
# Write a function without sum and without numpy


def dot_product(a, b):
    dot = 0
    for x, y in zip(a, b):
        dot += x * y
    return dot


a = [2, 4, 6]
b = [1, 3, 5]

dot = dot_product(a, b)

print("a · b =", dot)  # 44
