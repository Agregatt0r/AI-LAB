# Practice: Decision Tree Hyperparameters & Early Stopping
# Demonstrates how max_depth and min_samples_leaf control tree growth
# and compare with scikit-learn's DecisionTreeClassifier.

import numpy as np

try:
    from sklearn.tree import DecisionTreeClassifier as SklearnDTC
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# Minimal recursive Node and Tree to demonstrate early stopping
class SimpleNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self):
        return self.value is not None


class ConstrainedDecisionTree:
    """Decision tree implementing early stopping hyperparameters."""

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

                # Early stopping rule: min_samples_leaf constraint
                if n_l < self.min_samples_leaf or n_r < self.min_samples_leaf:
                    continue

                g_split = (n_l / n) * self._gini(y[left_mask]) + (n_r / n) * self._gini(y[right_mask])
                gain = parent_gini - g_split

                if gain > best_gain:
                    best_gain = gain
                    best_feat = j
                    best_thresh = t

        return best_feat, best_thresh

    def _build_tree(self, X, y, current_depth=0):
        n_samples = len(y)
        n_classes = len(np.unique(y))

        # Stopping criteria:
        # 1. Node is pure
        # 2. Reached max_depth
        # 3. Fewer samples than min_samples_split
        if (n_classes <= 1 or
            (self.max_depth is not None and current_depth >= self.max_depth) or
            n_samples < self.min_samples_split):
            return SimpleNode(value=self._majority_vote(y))

        feat, thresh = self._find_best_split(X, y)
        if feat is None:
            return SimpleNode(value=self._majority_vote(y))

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], current_depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], current_depth + 1)

        return SimpleNode(feature=feat, threshold=thresh, left=left_child, right=right_child)

    def fit(self, X, y):
        self.root = self._build_tree(X, y, current_depth=0)
        return self

    def _count_leaves(self, node):
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def _max_tree_depth(self, node):
        if node.is_leaf:
            return 0
        return 1 + max(self._max_tree_depth(node.left), self._max_tree_depth(node.right))


if __name__ == "__main__":
    # Generate Synthetic Dataset with 100 samples
    rng = np.random.default_rng(42)
    X = rng.uniform(-2.0, 2.0, size=(100, 2))
    # Concentric pattern
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2) < 1.0).astype(int)

    # 1. Test different max_depth settings
    depth_configs = [1, 2, 4, None]

    print("--- 1. Impact of max_depth on Tree Structure ---")
    print(f"{'max_depth':<12} | {'Actual Depth':<14} | {'Total Leaves':<12}")
    print("-" * 45)

    for md in depth_configs:
        tree = ConstrainedDecisionTree(max_depth=md, min_samples_split=2, min_samples_leaf=1)
        tree.fit(X, y)
        actual_depth = tree._max_tree_depth(tree.root)
        total_leaves = tree._count_leaves(tree.root)
        md_str = str(md) if md is not None else "None (Uncut)"
        print(f"{md_str:<12} | {actual_depth:<14} | {total_leaves:<12}")

        if md is not None:
            assert actual_depth <= md, f"Tree depth {actual_depth} exceeded max_depth {md}!"

    # 2. Test min_samples_leaf constraint
    print("\n--- 2. Impact of min_samples_leaf ---")
    tree_leaf1 = ConstrainedDecisionTree(max_depth=None, min_samples_leaf=1).fit(X, y)
    tree_leaf10 = ConstrainedDecisionTree(max_depth=None, min_samples_leaf=10).fit(X, y)

    leaves_1 = tree_leaf1._count_leaves(tree_leaf1.root)
    leaves_10 = tree_leaf10._count_leaves(tree_leaf10.root)

    print(f"min_samples_leaf = 1:  Total Leaves = {leaves_1}")
    print(f"min_samples_leaf = 10: Total Leaves = {leaves_10}")

    assert leaves_10 < leaves_1, "min_samples_leaf=10 should produce a smaller tree than leaf=1"

    if HAS_SKLEARN:
        sk_dtc = SklearnDTC(max_depth=2, random_state=42).fit(X, y)
        print(f"\nsklearn max_depth=2 leaf count: {sk_dtc.get_n_leaves()}")

    print("\nHyperparameter constraints and early stopping successfully verified!")

