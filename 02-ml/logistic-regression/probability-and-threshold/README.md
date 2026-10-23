# Probability & Decision Threshold

Logistic regression does not output a class directly. It outputs a **probability**, and a **decision threshold** converts that probability into a class label.

---

## 1. Predicted Probability

The model estimates the probability that a sample belongs to class 1:

$$p = P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b)$$

Because the sigmoid is bounded, $p \in (0, 1)$. The probability of class 0 is the complement:

$$P(y = 0 \mid \mathbf{x}) = 1 - p$$

---

## 2. Decision Rule

To get a hard label we compare $p$ with a threshold $t$:

$$\hat{y} = \begin{cases} 1 & \text{if } p \geq t \\ 0 & \text{if } p < t \end{cases}$$

The default is $t = 0.5$, which corresponds to the linear score $z \geq 0$.

---

## 3. Why the Threshold Is a Choice

The threshold is not fixed by the math. It encodes how costly each kind of error is:

- **Lower $t$** (e.g. $0.1$): predict class 1 more often. Catches more true positives (higher **recall**), but produces more false alarms (lower **precision**).
- **Higher $t$** (e.g. $0.9$): predict class 1 only when very sure. Fewer false alarms (higher precision), but misses more positives (lower recall).

Example: for a cancer screening test, a low threshold is preferred (missing a case is worse than a false alarm). For a spam filter that deletes mail, a high threshold is preferred (losing a real email is worse than letting spam through).

---

## 4. Confusion Matrix Metrics

|  | Predicted 1 | Predicted 0 |
|---|---|---|
| **Actual 1** | TP | FN |
| **Actual 0** | FP | TN |

$$\text{Precision} = \frac{TP}{TP + FP} \qquad \text{Recall} = \frac{TP}{TP + FN}$$

Moving the threshold trades one metric against the other. Recall never increases as $t$ rises, because fewer samples are predicted positive.

---

## 5. Choosing the Threshold

- Use the default $0.5$ when classes are balanced and errors cost the same.
- Otherwise, sweep thresholds on a **validation set** (never the test set) and pick the one that meets the business goal, for example the precision-recall curve or a target recall.

---

## Summary

- Logistic regression outputs $p = P(y=1 \mid \mathbf{x}) \in (0, 1)$.
- A threshold $t$ (default $0.5$) turns $p$ into a label.
- The threshold trades precision against recall.

**Previous:** [Sigmoid](../sigmoid/README.md) · **Next:** [Decision Boundary](../decision-boundary/README.md)
