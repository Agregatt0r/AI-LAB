# Task 1: Matrix multiplication
# Given A and B, check shapes, explain if you can multiply them,
# what shape A*B will have, calculate c[0][0] manually,
# and write a multiplication function without numpy
#
# A shape (2, 3), B shape (3, 2) → yes (cols A == rows B), result shape (2, 2)
# c[0][0] = 1*7 + 2*9 + 3*11 = 58


def shape(matrix):
    return len(matrix), len(matrix[0])


def multiply(A, B):
    C = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            c = 0
            for k in range(len(B)):
                c += A[i][k] * B[k][j]
            row.append(c)
        C.append(row)
    return C


A = [
    [1, 2, 3],
    [4, 5, 6]
]

B = [
    [7, 8],
    [9, 10],
    [11, 12]
]

print(shape(A))  # (2, 3)
print(shape(B))  # (3, 2)

# You can multiply only if A columns == B rows
C = multiply(A, B)
print(C)         # [[58, 64], [139, 154]]
print(shape(C))  # (2, 2)

c00 = A[0][0] * B[0][0] + A[0][1] * B[1][0] + A[0][2] * B[2][0]
print(c00)       # 58
