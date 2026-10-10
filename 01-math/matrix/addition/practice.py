# Task 1: Matrix addition
# Given two matrices A and B, calculate A + B
# Write a function without numpy


def matrix_addition(A, B):
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(A[0])):
            row.append(A[i][j] + B[i][j])
        result.append(row)
    return result


A = [
    [1, 2, 3],
    [4, 5, 6]
]

B = [
    [10, 20, 30],
    [40, 50, 60]
]

print("A + B =", matrix_addition(A, B))  # [[11, 22, 33], [44, 55, 66]]
