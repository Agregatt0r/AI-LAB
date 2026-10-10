# Conditional Probability

**Conditional probability** measures the probability of an event $A$ occurring, given that another event $B$ has already occurred. It is denoted as $P(A \mid B)$ (read "probability of $A$ given $B$").

## Formula

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad \text{where } P(B) > 0$$

- $P(A \cap B)$: The joint probability that both events $A$ and $B$ occur.
- $P(B)$: The probability that condition $B$ occurs.

## Independent Events

Two events $A$ and $B$ are **independent** if the occurrence of $B$ does not affect the probability of $A$:
$$P(A \mid B) = P(A) \iff P(A \cap B) = P(A) \cdot P(B)$$

## Example

In a group of $20$ students:
- $12$ students know Python ($P(\text{Python}) = \frac{12}{20} = 0.60$).
- $8$ students know C# ($P(\text{C\#}) = \frac{8}{20} = 0.40$).
- $5$ students know both Python and C# ($P(\text{Python} \cap \text{C\#}) = \frac{5}{20} = 0.25$).

The probability that a randomly chosen Python student also knows C#:
$$P(\text{C\#} \mid \text{Python}) = \frac{P(\text{Python} \cap \text{C\#})}{P(\text{Python})} = \frac{5 / 20}{12 / 20} = \frac{5}{12} \approx 0.4167 \ (41.67\%)$$

## Conditional Probability in ML

- **Language Models:** Predict the next word given preceding words $P(w_t \mid w_1, w_2, \dots, w_{t-1})$.
- **Decision Trees:** Split nodes based on condition outcomes.
