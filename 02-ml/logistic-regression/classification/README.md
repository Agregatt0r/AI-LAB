# Classification & Binary Classification

**Classification** is a supervised learning task where the target $y$ is **discrete**: it takes one of a finite set of class labels. In **binary classification** there are exactly two classes, which we encode as $y \in \{0, 1\}$ (for example: spam / not spam, fraud / legitimate, pass / fail).

This folder explains why we cannot simply reuse Linear Regression for classification, and what kind of model output we need instead.

---

## 1. Problem Setup

Given a dataset of $n$ samples $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ with $\mathbf{x}_i \in \mathbb{R}^d$ and $y_i \in \{0, 1\}$, we want a model that predicts the class of a new input $\mathbf{x}$.

The model must answer a question of the form:

$$P(y = 1 \mid \mathbf{x})$$

That is: "given these features, what is the probability this sample belongs to class 1?"

---

## 2. The Naive Approach: Linear Regression on Labels

A tempting first attempt is to fit a linear model to the 0/1 labels:

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b$$

and then predict class 1 when $\hat{y} \geq 0.5$. This has two problems:

1. **Outputs are unbounded.** $\hat{y}$ can be negative or greater than 1, so it cannot be a probability.
2. **Outliers distort the fit.** A few very confident samples far from the boundary pull the line, shifting the threshold for everyone else.

The practice script shows the first problem directly: the raw linear outputs fall well outside $[0, 1]$.

---

## 3. What We Need

For a classifier to output a meaningful probability, the model must satisfy:

$$0 \leq P(y=1 \mid \mathbf{x}) \leq 1$$

The solution is to pass the linear score $z = \mathbf{w}^\top \mathbf{x} + b$ through a function that squashes any real number into $(0, 1)$. That function is the **sigmoid** (see `../sigmoid`), which gives **logistic regression**.

---

## 4. Output Types

| Output | Meaning | Example |
|---|---|---|
| Hard label $\hat{y} \in \{0, 1\}$ | Final decision | "spam" |
| Probability $p \in (0, 1)$ | Confidence in class 1 | $p = 0.93$ |
| Log-odds $z \in \mathbb{R}$ | Linear score before squashing | $z = 2.6$ |

Logistic regression produces all three, and they are related by the sigmoid.

---

## 5. Geometric Intuition

A linear classifier splits the feature space into two regions with a hyperplane. Points on one side are predicted as class 0, points on the other side as class 1. The learning problem is to find the hyperplane that separates the classes as well as possible under a probabilistic loss.

---

## 6. Real-World Context

- **Medicine:** disease present or absent, given test results.
- **Finance:** default or repay a loan.
- **Email:** spam or not spam.
- **Vision:** cat or dog in an image (after feature extraction).

---

## Summary

- Binary classification predicts $y \in \{0, 1\}$.
- Fitting a linear regression to labels gives unbounded outputs that are not probabilities.
- Logistic regression fixes this by passing a linear score through the sigmoid.

**Next:** [Sigmoid Function](../sigmoid/README.md)
