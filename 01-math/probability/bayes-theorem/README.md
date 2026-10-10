# Bayes' Theorem

**Bayes' Theorem** describes how to update the probability of a hypothesis ($A$) when given new information or observed evidence ($B$).

## Formula

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

Using the **Law of Total Probability**, the denominator $P(B)$ (the total probability of the evidence) can be expanded as:
$$P(B) = P(B \mid A) \cdot P(A) + P(B \mid \neg A) \cdot P(\neg A)$$

## The Four Components

1. **Prior Probability $P(A)$:** Our initial belief about hypothesis $A$ before seeing evidence $B$.
2. **Likelihood $P(B \mid A)$:** The probability of observing evidence $B$ given that hypothesis $A$ is true.
3. **Marginal Evidence $P(B)$:** The total probability of observing evidence $B$ across all possible scenarios.
4. **Posterior Probability $P(A \mid B)$:** The updated probability of hypothesis $A$ after observing evidence $B$.

## Example

Suppose a disease affects $20\%$ of the population ($P(A) = 0.20$).
A medical test has:
- True Positive Rate / Sensitivity (Likelihood): $P(B \mid A) = 0.80$
- False Positive Rate: $P(B \mid \neg A) = 0.10$

1. Total probability of a positive test $P(B)$:
   $$P(B) = (0.80 \times 0.20) + (0.10 \times 0.80) = 0.16 + 0.08 = 0.24$$

2. Probability of actually having the disease given a positive test $P(A \mid B)$:
   $$P(A \mid B) = \frac{0.80 \times 0.20}{0.24} = \frac{0.16}{0.24} \approx 0.6667 \ (66.67\%)$$

## Bayes' Theorem in ML

- **Naive Bayes Classifier:** Used in spam filtering, text classification, and sentiment analysis.
- **Bayesian Optimization:** Hyperparameter tuning in deep learning.
- **Bayesian Neural Networks:** Quantifying model uncertainty.
