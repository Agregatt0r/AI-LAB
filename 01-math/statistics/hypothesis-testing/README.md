# Hypothesis Testing & A/B Testing

**Hypothesis testing** is a statistical framework for determining whether an observed effect in data is real or likely occurred by random chance.

---

## 1. The 5 Steps of Hypothesis Testing

1. **State the Hypotheses:**
   - **Null Hypothesis ($H_0$):** Status quo / no effect / no difference (e.g. $\mu = \mu_0$ or $\mu_A = \mu_B$).
   - **Alternative Hypothesis ($H_a$):** The effect we want to test (e.g. $\mu > \mu_0$ or $\mu_A \ne \mu_B$).

2. **Choose Significance Level ($\alpha$):**
   - Typically $\alpha = 0.05$ ($5\%$ threshold for false alarms).

3. **Compute the Test Statistic:**
   - **One-Sample Z-Test Statistic:**
     $$Z = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}$$

4. **Compute the $p$-value:**
   - The probability of obtaining a test statistic at least as extreme as the observed one, assuming $H_0$ is true.

5. **Make a Decision:**
   - If $p\text{-value} \le \alpha$: **Reject $H_0$** (Statistically significant).
   - If $p\text{-value} > \alpha$: **Fail to reject $H_0$** (Not enough evidence).

---

## 2. Decision Errors

| Decision | $H_0$ is True | $H_0$ is False |
| :--- | :--- | :--- |
| **Reject $H_0$** | **Type I Error ($\alpha$)** (False Positive) | **Correct Decision (Power $1 - \beta$)** |
| **Fail to Reject $H_0$** | **Correct Decision ($1 - \alpha$)** | **Type II Error ($\beta$)** (False Negative) |

---

## 3. A/B Testing in Data Science & ML

In Machine Learning, hypothesis testing is standard for **A/B Testing**:
- **Model A (Baseline)** vs **Model B (New Model / Prompt)**.
- Metric: Conversion Rate, Click-Through Rate, Loss, or F1-Score.
- Hypothesis testing ensures the improvement observed in Model B is statistically significant and not random noise.
