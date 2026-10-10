# Task 1: Determinant and inverse
# Given matrix A, calculate det, say if inverse exists, find inverse, check A * A^-1 = I


A = [
    [2, 1],
    [1, 3]
]

# A = [a b] = [2 1]
#     [c d]   [1 3]
# det(A) = ad - bc
det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
print(det)        # 5
print(det != 0)   # True → inverse exists

# A^-1 = (1/det) * [ d -b]
#                  [-c  a]
A_inv = [
    [A[1][1] / det, -A[0][1] / det],
    [-A[1][0] / det, A[0][0] / det],
]
print(A_inv)  # [[0.6, -0.2], [-0.2, 0.4]]

# check A * A^-1 = I
I = [
    [
        A[0][0] * A_inv[0][0] + A[0][1] * A_inv[1][0],
        A[0][0] * A_inv[0][1] + A[0][1] * A_inv[1][1],
    ],
    [
        A[1][0] * A_inv[0][0] + A[1][1] * A_inv[1][0],
        A[1][0] * A_inv[0][1] + A[1][1] * A_inv[1][1],
    ],
]
print(I)  # [[1.0, 0.0], [0.0, 1.0]]
