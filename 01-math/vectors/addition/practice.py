# Task 1: Vector Addition
# Given two vectors a and b, calculate a + b
# Write a function without sum and without numpy


def vector_addition(a, b):
    return [x + y for x, y in zip(a, b)]


a = [2, 4, 6]
b = [1, 3, 5]

add = vector_addition(a, b)

print("a + b =", add)  # [3, 7, 11]
