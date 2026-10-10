# Correlation

While covariance indicates the direction of a linear relationship between two variables, its value depends on the units of measurement.

**Correlation** standardizes covariance into a unitless value between $-1$ and $+1$.

## 1. Pearson Correlation Coefficient ($r$)

Measures the strength and direction of the **linear** relationship between $X$ and $Y$:

$$r = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}} = \frac{\text{Cov}(X, Y)}{s_X s_Y}$$

## Interpreting $r$

- $r = +1.0$: Perfect positive linear correlation
- $0.7 \le r < 1.0$: Strong positive correlation
- $0.3 \le r < 0.7$: Moderate positive correlation
- $-0.3 < r < +0.3$: Weak or no linear correlation
- $-1.0 < r \le -0.7$: Strong negative correlation
- $r = -1.0$: Perfect negative linear correlation

## Correlation vs Causation

> **"Correlation does not imply causation."**

Two variables can be strongly correlated due to:
1. Pure coincidence
2. A **confounding variable** (a common third cause)
3. Reverse causality

## Correlation in Machine Learning
- **Feature Selection:** Identify features with high correlation to the target variable $y$.
- **Multicollinearity Detection:** Drop or combine highly collinear features ($r > 0.9$) to improve model stability in Linear/Logistic Regression.
