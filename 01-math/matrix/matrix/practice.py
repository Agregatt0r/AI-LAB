# Task 1: Matrix shape
# Given matrix A, return (rows, columns) without using numpy


def shape(matrix):
    return len(matrix), len(matrix[0])


A = [
    [1, 2, 3],
    [4, 5, 6]
]

print(shape(A))  # (2, 3)
