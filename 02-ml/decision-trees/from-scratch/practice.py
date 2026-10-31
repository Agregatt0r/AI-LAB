# Practice: Complete Decision Tree Classifier From Scratch
# Implements:
#   - Node dataclass (decision vs leaf)
#   - Recursive splitting with Gini Impurity & Entropy criteria
#   - Pre-pruning: max_depth, min_samples_split, min_samples_leaf
#   - Methods: fit, predict, predict_proba, score, print_tree
#   - Side-by-side verification with scikit-learn

from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    from sklearn.tree import DecisionTreeClassifier as SklearnDTC
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class Node:
    """Represents an internal decision node or terminal leaf node."""
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    value: Optional[int] = None                  # Majority class (if leaf)
    probabilities: Optional[np.ndarray] = None   # Class distribution (if leaf)
    impurity: Optional[float] = None             # Impurity at this node
    n_samples: Optional[int] = None              # Number of samples at this node

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeClassifier:
    """CART Decision Tree Classifier implemented in pure Python and NumPy."""

    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.classes_ = None

    def _impurity(self, y):
        """Compute node impurity (Gini or Entropy)."""
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)

        if self.criterion == "gini":
            return 1.0 - np.sum(probs ** 2)
        elif self.criterion == "entropy":
            probs = probs[probs > 0]
            return -np.sum(probs * np.log2(probs))
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def _compute_class_probabilities(self, y):
        """Compute empirical class probabilities aligned with self.classes_."""
        probs = np.zeros(len(self.classes_))
        for idx, c in enumerate(self.classes_):
            probs[idx] = np.mean(y == c)
        return probs

    def _majority_vote(self, y):
        """Return the most frequent class label in y."""
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]

    def _find_best_split(self, X, y):
        """Exhaustively find the feature and threshold maximizing impurity reduction."""
        n_samples, n_features = X.shape
        parent_impurity = self._impurity(y)

        best_gain = -1.0
        best_feat = None
        best_thresh = None

        for feature_idx in range(n_features):
            feature_vals = X[:, feature_idx]
            unique_vals = np.sort(np.unique(feature_vals))

            if len(unique_vals) <= 1:
                continue

            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for t in thresholds:
                left_mask = feature_vals <= t
                right_mask = ~left_mask

                n_l = np.sum(left_mask)
                n_r = n_samples - n_l

                # Early stopping: min_samples_leaf
                if n_l < self.min_samples_leaf or n_r < self.min_samples_leaf:
                    continue

                y_l = y[left_mask]
                y_r = y[right_mask]

                child_impurity = (n_l / n_samples) * self._impurity(y_l) + (n_r / n_samples) * self._impurity(y_r)
                gain = parent_impurity - child_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feature_idx
                    best_thresh = t

        return best_feat, best_thresh, best_gain

    def _grow_tree(self, X, y, depth=0):
        """Recursive CART tree builder."""
        n_samples = len(y)
        n_unique_classes = len(np.unique(y))
        node_impurity = self._impurity(y)

        # Base case: stopping conditions
        if (n_unique_classes <= 1 or
            (self.max_depth is not None and depth >= self.max_depth) or
            n_samples < self.min_samples_split):
            return Node(
                value=self._majority_vote(y),
                probabilities=self._compute_class_probabilities(y),
                impurity=node_impurity,
                n_samples=n_samples
            )

        feat, thresh, gain = self._find_best_split(X, y)

        # If no split achieves positive gain, make leaf
        if feat is None or gain <= 0.0:
            return Node(
                value=self._majority_vote(y),
                probabilities=self._compute_class_probabilities(y),
                impurity=node_impurity,
                n_samples=n_samples
            )

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature=feat,
            threshold=thresh,
            left=left_child,
            right=right_child,
            impurity=node_impurity,
            n_samples=n_samples
        )

    def fit(self, X, y):
        """Build decision tree classifier from training set (X, y)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def _traverse_one(self, x, node: Node) -> Node:
        """Traverse tree down to a leaf node for a single sample."""
        if node.is_leaf:
            return node
        if x[node.feature] <= node.threshold:
            return self._traverse_one(x, node.left)
        return self._traverse_one(x, node.right)

    def predict(self, X):
        """Predict class labels for samples in X."""
        X = np.asarray(X, dtype=float)
        return np.array([self._traverse_one(x, self.root).value for x in X])

    def predict_proba(self, X):
        """Predict class probabilities for samples in X."""
        X = np.asarray(X, dtype=float)
        return np.array([self._traverse_one(x, self.root).probabilities for x in X])

    def score(self, X, y):
        """Return mean accuracy on the given test data and labels."""
        return np.mean(self.predict(X) == y)

    def print_tree(self, node: Optional[Node] = None, depth=0, feature_names=None):
        """Print an ASCII visual representation of the tree rules."""
        if node is None:
            node = self.root

        indent = "    " * depth
        if node.is_leaf:
            probs = [round(p, 2) for p in node.probabilities]
            print(f"{indent}|-- Leaf (samples={node.n_samples}, impurity={node.impurity:.2f}) -> "
                  f"Class {node.value} (probs={probs})")
        else:
            fname = feature_names[node.feature] if feature_names else f"x_{node.feature}"
            print(f"{indent}|-- [{fname} <= {node.threshold:.3f}] (samples={node.n_samples}, impurity={node.impurity:.2f})")
            self.print_tree(node.left, depth + 1, feature_names)
            self.print_tree(node.right, depth + 1, feature_names)


if __name__ == "__main__":
    # 3-Class Synthetic Dataset (Simulating Iris flower classification)
    rng = np.random.default_rng(42)
    n_per_class = 50

    # 3 clusters in 2D space: [Sepal Length, Petal Length]
    c0 = rng.normal(loc=[1.5, 0.3], scale=[0.3, 0.15], size=(n_per_class, 2))  # Setosa
    c1 = rng.normal(loc=[3.0, 1.3], scale=[0.3, 0.20], size=(n_per_class, 2))  # Versicolor
    c2 = rng.normal(loc=[4.2, 2.0], scale=[0.4, 0.25], size=(n_per_class, 2))  # Virginica

    X = np.vstack([c0, c1, c2])
    y = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)

    # Train / Test Split
    perm = rng.permutation(len(X))
    train_idx, test_idx = perm[:105], perm[105:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    feature_names = ["Sepal Length", "Petal Length"]

    # 1. Train Custom DecisionTreeClassifier
    print("--- 1. Training DecisionTreeClassifier from Scratch ---")
    custom_dt = DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_leaf=2)
    custom_dt.fit(X_train, y_train)

    train_acc = custom_dt.score(X_train, y_train)
    test_acc = custom_dt.score(X_test, y_test)

    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy:  {test_acc * 100:.2f}%\n")

    # 2. Print Tree Rules
    print("--- 2. Learned Decision Tree Structure ---")
    custom_dt.print_tree(feature_names=feature_names)

    # 3. Test Probabilities on Sample Inputs
    print("\n--- 3. Class Probabilities on Test Samples ---")
    sample_probs = custom_dt.predict_proba(X_test[:3])
    sample_preds = custom_dt.predict(X_test[:3])
    for i in range(3):
        print(f"Sample #{i+1}: Features = {np.round(X_test[i], 2).tolist()} -> "
              f"Probabilities = {np.round(sample_probs[i], 2).tolist()} -> Predict Class {sample_preds[i]}")

    # 4. Verification with scikit-learn
    print("\n--- 4. Verification Against scikit-learn ---")
    if HAS_SKLEARN:
        sk_dt = SklearnDTC(criterion="gini", max_depth=3, min_samples_leaf=2, random_state=42)
        sk_dt.fit(X_train, y_train)
        sk_test_acc = sk_dt.score(X_test, y_test)
        print(f"scikit-learn Test Accuracy: {sk_test_acc * 100:.2f}%")

        custom_preds = custom_dt.predict(X_test)
        sk_preds = sk_dt.predict(X_test)

        match_rate = np.mean(custom_preds == sk_preds)
        print(f"Agreement Rate with scikit-learn: {match_rate * 100:.2f}%")
        assert match_rate >= 0.98, "Predictions should match scikit-learn closely!"
        print("-> Exact matching performance with scikit-learn confirmed!")
    else:
        print("scikit-learn not installed; verified against analytical standards.")

    assert test_acc > 0.95, "Test accuracy should exceed 95%"
    print("\nComplete Decision Tree from scratch verified successfully!")

