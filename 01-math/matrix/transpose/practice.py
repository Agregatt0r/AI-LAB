# Task 1: Matrix transpose
# Given matrix A, calculate A_T without using numpy


def transpose(matrix):
    result = []
    for j in range(len(matrix[0])):
        row = []
        for i in range(len(matrix)):
            row.append(matrix[i][j])
        result.append(row)
    return result


def shape(matrix):
    return len(matrix), len(matrix[0])


A = [
    [1, 2, 3],
    [4, 5, 6]
]

A_T = transpose(A)
print(A)           # [[1, 2, 3], [4, 5, 6]]
print(A_T)         # [[1, 4], [2, 5], [3, 6]]
print(shape(A))    # (2, 3)
print(shape(A_T))  # (3, 2)
