# Practice: Overfitting vs Pre-Pruning in Decision Trees
# Demonstrates:
#   - Unconstrained Tree (max_depth=None): 100% train accuracy, memorizes noise, poor test accuracy
#   - Regularized / Pre-pruned Tree (max_depth=4): balances bias and variance, higher test accuracy

import numpy as np


class SimpleTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifierPruning:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        return 1.0 - np.sum((counts / len(y)) ** 2)

    def _majority_vote(self, y):
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]

    def _find_best_split(self, X, y):
        best_gain = -1.0
        best_feat, best_thresh = None, None
        parent_gini = self._gini(y)
        n = len(y)

        for j in range(X.shape[1]):
            vals = np.sort(np.unique(X[:, j]))
            if len(vals) <= 1:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2.0

            for t in thresholds:
                left_mask = X[:, j] <= t
                right_mask = ~left_mask
                n_l, n_r = np.sum(left_mask), np.sum(right_mask)

                if n_l < self.min_samples_leaf or n_r < self.min_samples_leaf:
                    continue

                g_split = (n_l / n) * self._gini(y[left_mask]) + (n_r / n) * self._gini(y[right_mask])
                gain = parent_gini - g_split

                if gain > best_gain:
                    best_gain = gain
                    best_feat = j
                    best_thresh = t

        return best_feat, best_thresh

    def _grow_tree(self, X, y, depth=0):
        n_samples = len(y)
        n_classes = len(np.unique(y))

        # Stopping condition
        if (n_classes <= 1 or
            (self.max_depth is not None and depth >= self.max_depth) or
            n_samples < self.min_samples_split):
            return SimpleTreeNode(value=self._majority_vote(y))

        feat, thresh = self._find_best_split(X, y)
        if feat is None:
            return SimpleTreeNode(value=self._majority_vote(y))

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        left_node = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return SimpleTreeNode(feature=feat, threshold=thresh, left=left_node, right=right_node)

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def _traverse(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def count_leaves(self, node=None):
        if node is None:
            node = self.root
        if node.is_leaf:
            return 1
        return self.count_leaves(node.left) + self.count_leaves(node.right)


if __name__ == "__main__":
    # Generate Synthetic Dataset with 20% random label noise
    rng = np.random.default_rng(42)
    n = 300
    X = rng.uniform(-2.0, 2.0, size=(n, 2))
    # True underlying non-linear pattern: quadrant XOR
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)

    # Add 20% noise to simulate real-world noisy data
    noise_idx = rng.choice(n, size=int(0.20 * n), replace=False)
    y[noise_idx] = 1 - y[noise_idx]

    # Split 70% Train, 30% Test
    split = int(0.7 * n)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("--- 1. Training Unconstrained Tree (Overfitting) ---")
    tree_unc = DecisionTreeClassifierPruning(max_depth=None, min_samples_leaf=1)
    tree_unc.fit(X_train, y_train)

    train_acc_unc = tree_unc.score(X_train, y_train)
    test_acc_unc = tree_unc.score(X_test, y_test)
    leaves_unc = tree_unc.count_leaves()

    print(f"Total Leaves:    {leaves_unc}")
    print(f"Train Accuracy:  {train_acc_unc * 100:.2f}% (Memorized noisy training points)")
    print(f"Test Accuracy:   {test_acc_unc * 100:.2f}%")

    print("\n--- 2. Training Pre-Pruned Tree (max_depth=4, min_samples_leaf=4) ---")
    tree_pruned = DecisionTreeClassifierPruning(max_depth=4, min_samples_leaf=4)
    tree_pruned.fit(X_train, y_train)

    train_acc_pruned = tree_pruned.score(X_train, y_train)
    test_acc_pruned = tree_pruned.score(X_test, y_test)
    leaves_pruned = tree_pruned.count_leaves()

    print(f"Total Leaves:    {leaves_pruned}")
    print(f"Train Accuracy:  {train_acc_pruned * 100:.2f}%")
    print(f"Test Accuracy:   {test_acc_pruned * 100:.2f}%")

    print("\n--- 3. Comparison ---")
    print(f"Overfitting Gap (Train - Test) for Unconstrained: {(train_acc_unc - test_acc_unc) * 100:.2f}%")
    print(f"Overfitting Gap (Train - Test) for Pre-Pruned:    {(train_acc_pruned - test_acc_pruned) * 100:.2f}%")
    print(f"Test Accuracy Improvement from Pruning:         +{(test_acc_pruned - test_acc_unc) * 100:.2f}%")

    # Assertions
    assert train_acc_unc == 1.0, "Unconstrained tree must achieve 100% on training set"
    assert test_acc_pruned > test_acc_unc, "Pre-pruned tree must generalize better to test set"
    assert leaves_pruned < leaves_unc, "Pruned tree must have significantly fewer leaves"

    print("\nOverfitting and pre-pruning benefits successfully verified!")
