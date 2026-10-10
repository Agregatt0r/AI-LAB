# Task 1: Matrix × vector
# Given matrix A and vector x, calculate A * x without numpy
# You can multiply when columns of A == length of x


def shape(matrix):
    return len(matrix), len(matrix[0])


def matrix_vector_multiply(matrix, vector):
    result = []
    for i in range(len(matrix)):
        total = 0
        for j in range(len(vector)):
            total += matrix[i][j] * vector[j]
        result.append(total)
    return result


A = [
    [2, 1, 3],
    [4, 2, 1]
]

x = [5, 10, 2]

print(shape(A))                      # (2, 3)
print(len(x))                        # 3
print(matrix_vector_multiply(A, x))  # [26, 42]


# Task 2: Prediction as a weighted sum (dot product)
# y = w0*x0 + w1*x1 + w2*x2 without numpy

features = [180, 70, 20]
weights = [0.5, 0.3, 0.2]


def calculate_y(x, w):
    y = 0
    for i in range(len(x)):
        y += w[i] * x[i]
    return y


print(calculate_y(features, weights))  # 115
# y = 0.5*180 + 0.3*70 + 0.2*20 = 90 + 21 + 4 = 115
