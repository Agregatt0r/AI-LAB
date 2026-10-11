# Learning Paradigms: Supervised vs Unsupervised Learning

Machine Learning algorithms are categorized into distinct paradigms based on what kind of data they learn from and what goal they pursue.

---

## 1. Supervised Learning (Навчання з учителем)

In **Supervised Learning**, the training dataset includes both input features ($\mathbf{X}$) and their corresponding true answers, known as **labels** or targets ($\mathbf{y}$).

$$\mathcal{D}_{\text{train}} = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \dots, (\mathbf{x}_n, y_n)\}$$

- **Analogy**: A student learning with a teacher who provides a textbook containing practice questions and answer keys.
- **Goal**: Learn an approximate mapping function $\hat{y} = f(\mathbf{x})$ that generalizes to new, unseen inputs $\mathbf{x}_{\text{new}}$.
- **Two Core Subtypes**:
  1. **Regression**: Target $y$ is continuous (e.g. predicting house price, temperature).
  2. **Classification**: Target $y$ is discrete/categorical (e.g. spam vs ham, malignant vs benign tumor).

---

## 2. Unsupervised Learning (Навчання без учителя)

In **Unsupervised Learning**, the training dataset contains only input features ($\mathbf{X}$). There are **no labels** or answers provided ($\mathbf{y}$ does not exist).

$$\mathcal{D}_{\text{train}} = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n\}$$

- **Analogy**: A toddler given a bucket of mixed colored blocks with no instructions; they naturally sort them by shape or color.
- **Goal**: Discover the underlying structure, clusters, probability distribution, or lower-dimensional representation of the data.
- **Two Core Subtypes**:
  1. **Clustering**: Grouping similar samples together (e.g. Customer Segmentation via K-Means).
  2. **Dimensionality Reduction**: Compressing many correlated features into fewer principal components (e.g. PCA from `01-math/pca`).

---

## 3. Reinforcement Learning (Навчання з підкріпленням)

In **Reinforcement Learning (RL)**, an autonomous **agent** learns by interacting with a dynamic **environment**.
- The agent takes **actions** ($a$), transitions to a new **state** ($s$), and receives scalar **rewards** or penalties ($r$).
- **Goal**: Learn an optimal **policy** $\pi(s)$ that maximizes cumulative expected rewards over time (e.g. Chess engines, autonomous driving, AlphaGo).

---

## 4. Comparison Summary

| Dimension | Supervised Learning | Unsupervised Learning | Reinforcement Learning |
|---|---|---|---|
| **Data Provided** | Features $\mathbf{X}$ + Labels $\mathbf{y}$ | Features $\mathbf{X}$ only | Environment states, Actions, Rewards |
| **Feedback Type** | Direct, explicit supervision | No feedback, self-organized | Delayed scalar reward / penalty |
| **Primary Goal** | Predict target for new input | Uncover latent structure or groups | Learn optimal sequence of decisions |
| **Typical Tasks** | Classification, Regression | Clustering, PCA, Anomaly Detection | Game playing, Robotics, Resource allocation |
| **Algorithms** | Linear/Logistic Reg., Decision Trees, SVM | K-Means, DBSCAN, PCA, Autoencoders | Q-Learning, PPO, Deep Q-Networks (DQN) |
