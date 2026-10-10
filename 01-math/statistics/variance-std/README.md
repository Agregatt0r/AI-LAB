# Sample Variance, Standard Deviation, and Standard Error

In statistics, when working with a sample of size $n$, sample variance and standard deviation must be corrected for **bias** when estimating population variance.

## 1. Bessel's Correction ($n - 1$)

Because sample data tends to be slightly closer to the sample mean $\bar{x}$ than to the true population mean $\mu$, dividing by $n$ underestimates the true variance. 

Dividing by $n - 1$ provides an **unbiased estimator**:

$$s^2 = \frac{1}{n - 1} \sum_{i=1}^n (x_i - \bar{x})^2$$

- **Population Variance ($\sigma^2$):** Divides by $N$ (entire population).
- **Sample Variance ($s^2$):** Divides by $n - 1$ (sample).

## 2. Sample Standard Deviation ($s$)

The square root of sample variance:
$$s = \sqrt{\frac{1}{n - 1} \sum_{i=1}^n (x_i - \bar{x})^2}$$

## 3. Standard Error of the Mean ($\text{SE}$)

Measures how much the sample mean $\bar{x}$ is expected to vary from the true population mean $\mu$ across different random samples:

$$\text{SE} = \frac{s}{\sqrt{n}}$$

- As sample size $n$ grows, $\text{SE} \to 0$ (estimates become more precise).
