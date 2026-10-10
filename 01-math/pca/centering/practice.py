# Practice: Mean and Data Centering

X = [
    [2, 1],
    [3, 2],
    [4, 3],
    [5, 4],
]
n = len(X)

# 1. Calculate mean for each feature
mean_x = sum(row[0] for row in X) / n
mean_y = sum(row[1] for row in X) / n

print("Mean x:", mean_x)  # 3.5
print("Mean y:", mean_y)  # 2.5

# 2. Center the dataset
X_centered = [
    [row[0] - mean_x, row[1] - mean_y]
    for row in X
]

print("\nCentered dataset:")
for row in X_centered:
    print(row)
# [[-1.5, -1.5], [-0.5, -0.5], [0.5, 0.5], [1.5, 1.5]]
