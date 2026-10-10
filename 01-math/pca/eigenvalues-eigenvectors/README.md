# Eigenvalues and Eigenvectors

For a square matrix $A$, a non-zero vector $v$ is an **eigenvector** and $\lambda$ is its **eigenvalue** if:

$$A v = \lambda v$$

Geometrically, multiplying $v$ by $A$ only scales $v$ by $\lambda$ without rotating it off its span line.

## 1. Finding Eigenvalues

Rewrite as:
$$(A - \lambda I) v = 0$$

For a non-zero solution $v \neq 0$ to exist, the matrix $(A - \lambda I)$ must be singular:
$$\det(A - \lambda I) = 0$$

For a 2×2 matrix:
$$\det \begin{bmatrix} a - \lambda & b \\ c & d - \lambda \end{bmatrix} = \lambda^2 - \text{trace}(A)\lambda + \det(A) = 0$$

where:
- $\text{trace}(A) = a + d = \lambda_1 + \lambda_2$
- $\det(A) = ad - bc = \lambda_1 \cdot \lambda_2$

## 2. Finding Eigenvectors

Substitute each $\lambda$ back into $(A - \lambda I)v = 0$ to find the direction $v$.

## 3. Normalization

In PCA, eigenvectors are normalized to unit length ($\|v\| = 1$):
$$v_{\text{unit}} = \frac{v}{\sqrt{v_1^2 + v_2^2 + \dots}}$$
