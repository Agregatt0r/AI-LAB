# Sampling & Central Limit Theorem

In practice, collecting data from an entire **population** is often impossible or too expensive. Instead, we collect a representative **sample**.

## Population vs Sample

| Concept | Population | Sample |
| :--- | :--- | :--- |
| **Definition** | The entire group of interest | A subset drawn from the population |
| **Size** | $N$ | $n$ |
| **Values** | **Parameters** (fixed, true constants) | **Statistics** (estimates calculated from sample) |
| **Mean** | $\mu$ | $\bar{x}$ |
| **Variance** | $\sigma^2$ | $s^2$ |
| **Standard Deviation** | $\sigma$ | $s$ |

## Law of Large Numbers (LLN)
As sample size $n \to \infty$, the sample average $\bar{x}$ converges to the true population mean $\mu$:
$$\lim_{n \to \infty} \bar{x} = \mu$$

## Central Limit Theorem (CLT)
No matter what shape the underlying population distribution has (skewed, uniform, bimodal), the distribution of **sample means** $\bar{X}$ will approach a **Normal distribution** as sample size $n$ increases (typically $n \ge 30$):

$$\bar{X} \sim \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)$$

The standard deviation of this distribution is called the **Standard Error (SE)**:
$$\text{SE} = \frac{\sigma}{\sqrt{n}}$$

## Why Sampling Matters in ML
- **Train/Validation/Test Splits:** Samples must be representative and identically distributed (i.i.d.).
- **Mini-batch SGD:** Optimizes model weights using small random batches (samples) instead of the entire dataset.
- **Bootstrapping / Bagging:** Random sampling with replacement (e.g. Random Forests).
