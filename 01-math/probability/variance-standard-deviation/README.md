# Variance & Standard Deviation

While the **mean** ($\mu$) describes the central location of a dataset, **variance** ($\sigma^2$) and **standard deviation** ($\sigma$) describe the **spread or dispersion** of data around that mean.

## Formulas

1. **Mean ($\mu$):**
   $$\mu = \frac{1}{N} \sum_{i=1}^N x_i$$

2. **Population Variance ($\sigma^2$):**
   Average of the squared differences from the mean:
   $$\sigma^2 = \frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2$$

   *(Note: For sample variance $s^2$, divide by $N - 1$ to apply Bessel's correction)*.

3. **Standard Deviation ($\sigma$):**
   Square root of variance (measured in the original units of the data):
   $$\sigma = \sqrt{\sigma^2} = \sqrt{\frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2}$$

## Example

Dataset: $X = [2, 4, 6, 8, 10]$ ($N = 5$)

1. **Mean:**
   $$\mu = \frac{2 + 4 + 6 + 8 + 10}{5} = \frac{30}{5} = 6.0$$

2. **Squared Differences $(x_i - \mu)^2$:**
   $$(2-6)^2 = 16, \quad (4-6)^2 = 4, \quad (6-6)^2 = 0, \quad (8-6)^2 = 4, \quad (10-6)^2 = 16$$
   $$\text{Sum} = 16 + 4 + 0 + 4 + 16 = 40$$

3. **Variance:**
   $$\sigma^2 = \frac{40}{5} = 8.0$$

4. **Standard Deviation:**
   $$\sigma = \sqrt{8.0} \approx 2.8284$$

## Why it Matters in ML

- **Feature Standardization (Z-score normalization):** $z = \frac{x - \mu}{\sigma}$ centers data at $0$ with unit variance $1$.
- **Bias-Variance Tradeoff:** High variance leads to overfitting.
- **Normal (Gaussian) Distribution:** Completely parameterized by $\mu$ and $\sigma^2$.
