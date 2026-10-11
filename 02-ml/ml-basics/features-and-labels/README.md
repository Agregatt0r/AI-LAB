# Features ($\mathbf{X}$) and Target / Labels ($\mathbf{y}$)

At the heart of every supervised learning problem is the distinction between what information is provided as input (**Features**) and what value the model must predict (**Target** or **Label**).

---

## 1. Mathematical Structure & Conventions

In Machine Learning tabular datasets:

$$\mathbf{X} \in \mathbb{R}^{n \times d}, \quad \mathbf{y} \in \mathbb{R}^n$$

- **$n$**: Number of samples (rows / examples / instances).
- **$d$**: Number of features (columns / dimensions / variables).

```
                 Features (X)                     Target (y)
          Feature 1   Feature 2   Feature d
Sample 1 [   x₁₁         x₁₂   ...   x₁d   ]  ───▶  [   y₁   ]
Sample 2 [   x₂₁         x₂₂   ...   x₂d   ]  ───▶  [   y₂   ]
...      [   ...         ...   ...   ...   ]  ───▶  [  ...   ]
Sample n [   xₙ₁         xₙ₂   ...   xₙd   ]  ───▶  [   yₙ   ]
```

### Capital $\mathbf{X}$ vs Lowercase $\mathbf{y}$
- **$\mathbf{X}$ is capitalized** because it is a **2D Matrix** (multiple rows and multiple columns).
- **$\mathbf{y}$ is lowercase** because it is a **1D Vector** (a single target value per sample).

---

## 2. Feature Types

Not all data is immediately ready as numerical input for matrix operations:

| Feature Type | Description | Examples | Handling in ML |
|---|---|---|---|
| **Numerical (Continuous)** | Measurements on an unbroken scale | Temperature ($21.4^\circ\text{C}$), Price (\$250,000), Weight | Scaling / Standardization |
| **Numerical (Discrete)** | Integer counts | Number of rooms ($3$), Number of website visits ($12$) | Directly usable or scaled |
| **Categorical (Nominal)** | Discrete categories without intrinsic order | City (`"Kyiv"`, `"Warsaw"`), Color (`"Red"`, `"Blue"`) | **One-Hot Encoding** |
| **Categorical (Ordinal)** | Categories with a clear inherent order | Education (`"BSc" < "MSc" < "PhD"`), Size (`"S" < "M" < "L"`) | Integer mapping ($1, 2, 3$) |

---

## 3. One-Hot Encoding

Computers cannot perform algebraic matrix multiplication on raw string strings like `"Kyiv"` or `"Paris"`.

**One-Hot Encoding** transforms a categorical column with $K$ unique categories into $K$ binary indicator columns ($0$ or $1$):

| City | Kyiv | Warsaw | Berlin |
|---|:---:|:---:|:---:|
| `"Kyiv"` | **1** | 0 | 0 |
| `"Warsaw"` | 0 | **1** | 0 |
| `"Berlin"` | 0 | 0 | **1** |

Now the categorical values are encoded into numerical vectors compatible with matrix linear algebra ($\mathbf{X}\mathbf{w} + b$).
