# Task 1: Scalar multiplication
# Given matrix A, calculate 2 * A
# Write a function without numpy


def scalar_multiplication(matrix, k):
    result = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix[0])):
            row.append(matrix[i][j] * k)
        result.append(row)
    return result


A = [
    [1, 2, 3],
    [4, 5, 6]
]

print("2 * A =", scalar_multiplication(A, 2))  # [[2, 4, 6], [8, 10, 12]]
