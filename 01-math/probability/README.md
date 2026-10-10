# Probability (Day 5)

**Probability** is the mathematical foundation for reasoning under uncertainty. In Machine Learning and AI, probability theory allows models to quantify uncertainty, make predictions (e.g. classification probabilities), update beliefs via Bayesian inference, and optimize objective functions.

This folder contains practice exercises and notes for the core probability topics.

Each subfolder has:
- `README.md` — theory for that topic
- `practice.py` — your exercises

## Structure

```
probability/
├── probability                 what probability is (sample space, events, complement)
├── conditional-probability     probability of A given that B occurred P(A|B)
├── bayes-theorem               updating probabilities with new evidence (prior, likelihood, posterior)
├── random-variables            discrete vs continuous variables, PMF, PDF, CDF
├── distributions               theoretical distributions (Bernoulli, Binomial, Uniform, Normal)
├── expected-value              weighted average of all possible values E[X]
├── variance-standard-deviation theoretical spread of a distribution (Var(X) and std)
└── covariance                  joint variability of two random variables Cov(X, Y)
```
