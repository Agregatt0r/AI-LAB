# The Impact of Feature Scaling on KNN

Because K-Nearest Neighbors relies entirely on calculating distances between points, it is **one of the most sensitive algorithms to feature scales in all of machine learning**.

When features are not scaled to comparable numerical ranges, features with large values dominate the distance metric, rendering smaller-scale features invisible.

---

## 1. The Mathematical Problem

Consider two features:
- $x_1$: **Annual Income** in dollars: range $[\$20,000, \$150,000]$
- $x_2$: **Years of Experience**: range $[1, 15]$

Suppose we measure the Euclidean distance between two candidates:
- Candidate A: $(\$70,000, 2\text{ years})$
- Candidate B: $(\$72,000, 12\text{ years})$

$$d(A, B) = \sqrt{(70000 - 72000)^2 + (2 - 12)^2}$$
$$d(A, B) = \sqrt{(-2000)^2 + (-10)^2} = \sqrt{4,000,000 + 100} = \sqrt{4,000,100} \approx 2000.025$$

Notice what happened:
- Income contributed: $4,000,000$ ($99.9975\%$ of the squared distance)
- Experience contributed: $100$ ($0.0025\%$ of the squared distance)

A 10-year gap in professional experience is completely overshadowed by a trivial \$2,000 fluctuation in salary. The algorithm effectively throws away the experience feature!

---

## 2. Geometric Distortion

```
          Unscaled Space (Distorted Ellipse)                    Scaled Space (True Euclidean)
     Salary ^                                             Salary' ^
    $150k   |                                                +2   |         * (Neighbor)
            |        * (10 yrs difference                     0   |     *     ? (Query)
            |           treated as close!)                   -2   |         *
     $20k   |  ?                                                  +-------------------->
            +------------------------------->                     -2   0   +2  Experience'
            1              15  Experience
```

- **Unscaled:** The distance contours are extremely stretched ellipses. A point far away along the small-scale axis will still be considered a "nearest neighbor" if its large-scale feature is slightly closer.
- **Scaled:** Features are centered and share unit variance. Distance contours are spherical circles, giving each feature fair and equal voting weight.

---

## 3. Which Algorithms Require Feature Scaling?

| Requires Scaling | Why? | Examples |
|---|---|---|
| **Distance-Based** | Distances are computed directly across feature axes. | **KNN**, K-Means, SVM, PCA |
| **Gradient-Based** | Uneven scales produce narrow elliptical loss valleys, causing gradient descent to oscillate. | Linear Regression, Logistic Regression, Neural Networks |
| **Scale-Invariant (No Scaling Needed)** | Split rules only compare features to thresholds within their own 1D axis ($x_j \leq \theta$). | Decision Trees, Random Forests, XGBoost |

---

## Summary

- Distance metrics sum squared differences along all axes.
- Large numerical ranges dominate distances and drown out informative smaller-scale features.
- Scaling features is **mandatory** for KNN to produce geometrically valid neighborhoods.

**Previous:** [Choosing k](../choosing-k/README.md) · **Next:** [StandardScaler](../standard-scaler/README.md)

