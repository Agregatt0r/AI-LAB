# Task 4: Calculate the Euclidean distance between two points in a plane
# or higher-dimensional space without using numpy
# A = [1, 2]
# B = [4, 6]

def euclidean_distance(a, b):
    distance = 0

    for x, y in zip(a, b):
        distance += (x - y) ** 2
    return distance ** 0.5

a = [1, 2]
b = [4, 6]

print(euclidean_distance(a, b))  # 5

# Euclidean distance is the distance between two points in a plane or higher-dimensional
# space while vector magnitude is the length of a vector from starting point to ending point
