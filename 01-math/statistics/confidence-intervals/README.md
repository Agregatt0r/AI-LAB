# Confidence Intervals

A **Confidence Interval (CI)** provides a range of plausible values for an unknown population parameter (such as population mean $\mu$), rather than just a single point estimate ($\bar{x}$).

## Definition

A $95\%$ Confidence Interval means:
> If we repeated the experiment and collected many random samples, $95\%$ of the calculated confidence intervals would contain the true population parameter $\mu$.

## Formula for Population Mean

$$\text{CI} = \bar{x} \pm \text{Margin of Error}$$
$$\text{CI} = \bar{x} \pm z^* \cdot \frac{s}{\sqrt{n}}$$

Where:
- $\bar{x}$: Sample mean (point estimate)
- $z^*$: Critical value from standard normal distribution corresponding to the chosen confidence level $(1 - \alpha)$:
  - **$90\%$ Confidence:** $z^* \approx 1.645$
  - **$95\%$ Confidence:** $z^* \approx 1.960$
  - **$99\%$ Confidence:** $z^* \approx 2.576$
- $\frac{s}{\sqrt{n}}$: Standard Error ($\text{SE}$)

*(Note: For small samples $n < 30$ with unknown $\sigma$, replace $z^*$ with the critical $t^*$ value from Student's $t$-distribution with $df = n - 1$)*.

## How to Narrow the Confidence Interval
1. **Increase sample size $n$:** $\text{SE} \propto \frac{1}{\sqrt{n}}$ (quadrupling sample size cuts interval width in half).
2. **Reduce variability / noise in data ($s$).**
3. **Decrease confidence level** (e.g. from $99\%$ to $90\%$).
