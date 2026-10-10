# Basic Probability

**Probability** quantifies how likely an event is to occur. It is expressed as a real number between $0$ and $1$.

## Formula

For equally likely outcomes:
$$P(A) = \frac{\text{Number of favourable outcomes for event } A}{\text{Total number of possible outcomes}}$$

## Core Axioms and Rules

1. **Range:**
   $$0 \le P(A) \le 1$$
   - $P(A) = 0$: Impossible event
   - $P(A) = 1$: Certain (guaranteed) event

2. **Complement Rule:**
   The complement of event $A$ (denoted $\text{not } A$ or $A^c$) represents $A$ NOT occurring:
   $$P(A) + P(\text{not } A) = 1 \implies P(\text{not } A) = 1 - P(A)$$

## Example

Suppose an urn contains $3$ red marbles and $7$ blue marbles (total $= 10$).
- $P(\text{Red}) = \frac{3}{10} = 0.3$ ($30\%$)
- $P(\text{Blue}) = \frac{7}{10} = 0.7$ ($70\%$)
- $P(\text{not Red}) = 1 - 0.3 = 0.7$

## Probability in ML

- Classification models output probabilities (e.g. via Sigmoid or Softmax).
- A model might predict $P(y = \text{"cat"} \mid x) = 0.85$.
