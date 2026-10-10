# Practice: Pearson Correlation Coefficient (r)

import math


def pearson_correlation(x, y):
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("x and y must have the same length >= 2.")

    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = sum((xi - mean_x) ** 2 for xi in x)
    denom_y = sum((yi - mean_y) ** 2 for yi in y)

    denominator = math.sqrt(denom_x * denom_y)
    if denominator == 0:
        return 0.0

    return numerator / denominator


# Example: Study hours vs Exam score
study_hours = [1, 2, 3, 4, 5, 6, 7]
exam_scores = [52, 55, 65, 70, 78, 85, 92]

r = pearson_correlation(study_hours, exam_scores)

print("Study Hours:", study_hours)
print("Exam Scores:", exam_scores)
print(f"Pearson Correlation (r): {r:.4f}")

if r > 0.7:
    print("Interpretation: Strong positive linear relationship.")
elif r < -0.7:
    print("Interpretation: Strong negative linear relationship.")
else:
    print("Interpretation: Moderate/weak relationship.")
