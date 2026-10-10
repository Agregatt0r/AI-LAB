# Measures of Central Tendency

Measures of central tendency summarize an entire data distribution with a single representative central value.

## 1. Sample Mean ($\bar{x}$)
The arithmetic average of all sample observations:
$$\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$$
- **Pros:** Uses all data points; mathematically tractable for gradient descent and loss functions.
- **Cons:** Extremely sensitive to **outliers**.

## 2. Median
The middle value when data is sorted in ascending order:
- If $n$ is odd: middle element at position $\frac{n+1}{2}$.
- If $n$ is even: average of the two middle elements.
- **Pros:** Robust against extreme outliers and skewed distributions.

## 3. Mode
The most frequently occurring value(s) in the dataset.
- Useful for categorical/discrete data.

## 4. Trimmed / Winsorized Mean
Calculates the mean after removing a fixed percentage (e.g. $5\%$ or $10\%$) of extreme values from both ends.

## Summary Comparison

| Metric | Calculation | Sensitivity to Outliers | Best For |
| :--- | :--- | :--- | :--- |
| **Mean** | Sum / Count | High | Symmetric, normal data |
| **Median** | Sorted middle value | Low (Robust) | Skewed data (e.g. incomes, house prices) |
| **Mode** | Most frequent value | None | Categorical features |
