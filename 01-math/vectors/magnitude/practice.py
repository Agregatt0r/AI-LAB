# Task 3: Calculate the magnitude of a vector without using numpy
import math

def vector_magnitude(v):
    # return math.sqrt(sum(x ** 2 for x in v))

    m = 0
    for x in v:
        m += x ** 2
    return m ** 0.5

v = [6, 8]

print(vector_magnitude(v))  # 10

# 10 not 14 because its sqrt of (6^2 + 8^2)

# if each element doubled sqrt((12^2 + 16^2)) = 20 so it's 2x longer
