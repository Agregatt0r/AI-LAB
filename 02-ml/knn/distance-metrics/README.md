# Distance Metrics in K-Nearest Neighbors

KNN relies entirely on a distance function to quantify similarity between data points. Different distance metrics define different geometric concepts of "closeness" and affect which points are selected as the $k$ nearest neighbors.

---

## 1. Euclidean Distance ($L_2$ Norm)

The standard straight-line geometric distance between two points in Euclidean space:

$$d_2(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_{i=1}^d (u_i - v_i)^2}$$

### Characteristics:
- **Default metric** in almost all ML libraries (`scikit-learn`, PyTorch).
- Sensitive to large coordinate differences (squaring amplifies large gaps).
- Assumes isotropic (spherical) feature space.

---

## 2. Manhattan Distance ($L_1$ Norm / Taxicab)

The distance measured along axes at right angles, as if traversing a grid of city blocks:

$$d_1(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_1 = \sum_{i=1}^d |u_i - v_i|$$

### Characteristics:
- Does not square differences, making it **more robust to outliers** than Euclidean distance.
- Preferred when features represent distinct, non-continuous, or grid-like attributes.

---

## 3. Minkowski Distance ($L_p$ Metric)

The generalization of Euclidean and Manhattan distances parameterized by $p \geq 1$:

$$d_p(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_p = \left( \sum_{i=1}^d |u_i - v_i|^p \right)^{1/p}$$

| Value of $p$ | Metric | Formula | Geometry of Unit Sphere |
|---|---|---|---|
| $p = 1$ | **Manhattan** | $\sum \|u_i - v_i\|$ | Diamond |
| $p = 2$ | **Euclidean** | $\sqrt{\sum (u_i - v_i)^2}$ | Circle / Sphere |
| $p \to \infty$ | **Chebyshev** | $\max_i \|u_i - v_i\|$ | Square / Cube |

---

## 4. Cosine Distance

Measures the **angular difference** between two non-zero vectors, ignoring their magnitudes:

$$\text{Cosine Similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

$$d_{\text{cosine}}(\mathbf{u}, \mathbf{v}) = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

### When to use:
- **Text & NLP:** Word embeddings (Word2Vec, BERT) where a long document and short document about the same topic have similar direction but different word count lengths.
- **Recommendation Systems:** User preference vectors.

---

## 5. Metric Comparison

| Metric | Formula | Best For | Sensitive to Outliers |
|---|---|---|---|
| **Euclidean ($L_2$)** | $\sqrt{\sum (u_i - v_i)^2}$ | General continuous physical data | High |
| **Manhattan ($L_1$)** | $\sum \|u_i - v_i\|$ | High-dimensional data, noisy features | Low (robust) |
| **Cosine** | $1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$ | Text embeddings, directional data | Low (magnitude-invariant) |

---

## 6. Vectorized Pairwise Distance Matrix

Computing distances between $m$ test samples and $n$ training samples using nested Python loops is $O(m \cdot n)$ and too slow.

Using the algebraic expansion:
$$\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 - 2\mathbf{u}^\top\mathbf{v} + \|\mathbf{v}\|^2$$

We can compute the entire pairwise Euclidean distance matrix $\mathbf{D} \in \mathbb{R}^{m \times n}$ in a single vectorized NumPy operation:

$$\mathbf{D}^2 = \sum(\mathbf{X}_{\text{test}}^2, \text{axis}=1)[:, \text{None}] - 2 \mathbf{X}_{\text{test}} \mathbf{X}_{\text{train}}^\top + \sum(\mathbf{X}_{\text{train}}^2, \text{axis}=1)[\text{None}, :]$$

This leverages BLAS matrix multiplication and achieves **50x to 100x speedups**.

---

## Summary

- **Euclidean distance** is the default for continuous physical features.
- **Manhattan distance** is more robust to outlier coordinates and works well in higher dimensions.
- **Cosine distance** measures direction rather than length, making it ideal for embeddings.
- Vectorized distance matrix computation replaces slow nested loops with fast matrix multiplication.

**Previous:** [How KNN Works](../how-knn-works/README.md) · **Next:** [Choosing k](../choosing-k/README.md)

