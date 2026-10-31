# Decision Tree Structure: Nodes, Branches & Leaves

A **Decision Tree** is a non-parametric supervised learning algorithm that makes predictions by recursively partitioning the feature space into rectangular regions using a sequence of simple binary conditional tests.

It mimics human decision-making: asking a hierarchy of "if-then-else" questions until reaching a final conclusion.

---

## 1. Anatomy of a Decision Tree

A decision tree consists of three core components:

```
               [ Root Node ]                 Level 0
               (Age <= 30?)
                /        \   <-- Branches (Edges)
            Yes/          \No
              v            v
     [ Internal Node ]   ( Income <= $50k? ) Level 1
      ( Has Job? )            /         \
        /      \          Yes/           \No
     Yes/      \No          v             v
       v        v       [ Leaf 2 ]    [ Leaf 3 ]
   [ Leaf 0 ] [ Leaf 1 ] (Approved)   (Rejected)
  (Approved)  (Rejected)
```

### 1. Root Node (Корінь)
The topmost node in the tree. It receives the entire dataset and performs the very first binary test on the most informative feature and threshold.

### 2. Internal / Decision Nodes (Вузли рішень)
Intermediate nodes that evaluate a specific feature against a threshold:
$$\text{Is } x_j \leq \theta\text{?}$$
- If **True (Yes)**, the sample moves along the left branch.
- If **False (No)**, the sample moves along the right branch.

### 3. Branches / Edges (Гілки)
The directional paths connecting a parent node to its child nodes, representing the outcome of the split condition.

### 4. Leaf / Terminal Nodes (Листя)
The end points of the tree where no further splits occur. A leaf node stores the final prediction:
- **Classification:** The majority class label $\hat{y} \in \{0, 1, \dots\}$ or class probabilities.
- **Regression:** The continuous numerical mean $\hat{y} \in \mathbb{R}$.

---

## 2. Geometric Interpretation: Axis-Aligned Partitions

Geometrically, every split condition ($x_j \leq \theta$) draws an **orthogonal, axis-aligned hyperplane** (horizontal or vertical lines in 2D space) dividing the feature space into two sub-regions:

```
       Feature 2 ^
                 |         Region 2 (Class 1)
                 |-----------------------
                 |  Region 1 |  Region 3
                 |  (Class 0)|  (Class 0)
                 +-------------------------> Feature 1
                            θ_1
```

Unlike linear models that separate classes with a single diagonal line ($\mathbf{w}^\top\mathbf{x} + b = 0$), decision trees build a grid of rectangular decision boxes.

---

## 3. Tree Traversal (Inference)

To classify a new sample $\mathbf{x}_{\text{query}}$:
1. Start at the **Root Node**.
2. Evaluate the test: $\mathbf{x}_{\text{query}}[j] \leq \theta$.
3. Follow the corresponding branch to the child node.
4. Repeat recursively until landing at a **Leaf Node**.
5. Output the leaf's stored value.

Inference time complexity is $O(\text{depth})$, which is at most $O(\log_2 n)$ for balanced trees—making decision tree predictions exceptionally fast.

---

## Summary

- Decision trees partition data through hierarchical binary rules ($x_j \leq \theta$).
- **Root Node:** initial split on the full dataset.
- **Internal Nodes:** intermediate tests directing samples down branches.
- **Leaf Nodes:** terminal endpoints holding the final prediction.
- Fast inference: prediction traverses top-to-bottom in $O(\text{depth})$ steps.

**Next:** [Gini Impurity](../gini-impurity/README.md)

