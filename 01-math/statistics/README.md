# Statistics (Day 6)

**Statistics** is the science of collecting, exploring, summarizing, and drawing inferences from data.

While **Probability** starts with known rules and models to deduce what data might look like (*deductive*), **Statistics** starts with observed data samples and works backwards to estimate unknown parameters, test hypotheses, and uncover relationships (*inductive*).

In Machine Learning, statistics is essential for:
- Exploratory Data Analysis (EDA) and feature engineering
- Validating model performance and generalization (A/B testing, statistical significance)
- Quantifying confidence in model estimates (confidence intervals, standard errors)
- Feature selection via correlation analysis and hypothesis testing

Each subfolder has:
- `README.md` — theory for that topic
- `practice.py` — your exercises

## Structure

```
statistics/
├── sampling                    population vs sample, random sampling, Central Limit Theorem (CLT)
├── mean                        measures of central tendency (mean, median, mode, robust stats)
├── variance-std                sample variance (Bessel's correction n-1), sample std, Standard Error
├── correlation                 Pearson (r) & Spearman rank correlation, correlation vs causation
├── distributions               empirical data distributions, standardization (Z-scores), t-distribution
├── confidence-intervals        point estimates, margin of error, 95% confidence intervals (CI)
└── hypothesis-testing          null hypothesis (H0), p-values, z-test, t-test, Type I & II errors
```
