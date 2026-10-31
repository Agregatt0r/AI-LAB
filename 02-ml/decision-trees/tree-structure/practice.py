# Practice: Decision Tree Structure & Traversal
# Demonstrates:
#   - Root, Internal Decision Nodes, and Leaf Nodes
#   - Binary condition evaluation (x[feature_idx] <= threshold)
#   - Tree traversal (forward inference path from root to leaf)
#   - ASCII tree visualization

from dataclasses import dataclass
from typing import Optional


@dataclass
class Node:
    """Represents a node in a decision tree."""
    # Split rule (only for internal nodes)
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    feature_name: Optional[str] = None

    # Children
    left: Optional["Node"] = None
    right: Optional["Node"] = None

    # Prediction value (only for leaf nodes)
    value: Optional[str] = None

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


def traverse_tree(x, node: Node, trace=False) -> str:
    """Traverse from root to leaf, evaluating binary conditions at each node."""
    current = node
    step = 1

    while not current.is_leaf:
        val = x[current.feature_idx]
        cond = val <= current.threshold
        branch = "Left (Yes)" if cond else "Right (No)"

        if trace:
            print(f"  Step {step}: Node [{current.feature_name} <= {current.threshold}] -> "
                  f"Sample value is {val} -> Follow {branch}")

        current = current.left if cond else current.right
        step += 1

    if trace:
        print(f"  Reached Leaf: Prediction = '{current.value}'\n")

    return current.value


def print_tree(node: Node, depth=0, prefix="Root: "):
    """Print an ASCII visual representation of the decision tree."""
    indent = "    " * depth
    if node.is_leaf:
        print(f"{indent}{prefix}Leaf -> Predict: '{node.value}'")
    else:
        print(f"{indent}{prefix}[{node.feature_name} <= {node.threshold}]")
        print_tree(node.left, depth + 1, prefix="-- Left (True)  -> ")
        print_tree(node.right, depth + 1, prefix="-- Right (False) -> ")


if __name__ == "__main__":
    # Define a toy Decision Tree for Loan Approval:
    # Features: [Age (years), Credit Score (0-800), Annual Income ($k)]
    # Feature 0: Age
    # Feature 1: Credit Score
    # Feature 2: Income
    feature_names = ["Age", "Credit Score", "Income"]

    # Build Tree Manually:
    # Root: Credit Score <= 600
    #   Left (Credit <= 600):
    #       Income <= 40?
    #           Left  -> "Denied"
    #           Right -> "Review"
    #   Right (Credit > 600):
    #       Income <= 30?
    #           Left  -> "Review"
    #           Right -> "Approved"
    tree = Node(
        feature_idx=1, threshold=600.0, feature_name="Credit Score",
        left=Node(
            feature_idx=2, threshold=40.0, feature_name="Income",
            left=Node(value="Denied"),
            right=Node(value="Review")
        ),
        right=Node(
            feature_idx=2, threshold=30.0, feature_name="Income",
            left=Node(value="Review"),
            right=Node(value="Approved")
        )
    )

    print("--- 1. Visual ASCII Tree Representation ---")
    print_tree(tree)
    print()

    # 2. Test Samples
    # Sample A: Age=25, Credit=550, Income=25 -> Denied
    # Sample B: Age=35, Credit=720, Income=80 -> Approved
    # Sample C: Age=45, Credit=580, Income=65 -> Review
    samples = [
        {"desc": "Applicant A (Low credit, low income)", "x": [25, 550, 25], "expected": "Denied"},
        {"desc": "Applicant B (High credit, high income)", "x": [35, 720, 80], "expected": "Approved"},
        {"desc": "Applicant C (Low credit, solid income)", "x": [45, 580, 65], "expected": "Review"},
    ]

    print("--- 2. Tracing Inference Traversal ---")
    for sample in samples:
        print(f"Testing: {sample['desc']}")
        prediction = traverse_tree(sample["x"], tree, trace=True)
        assert prediction == sample["expected"], f"Expected {sample['expected']}, got {prediction}"

    print("Tree structure and inference traversal successfully verified!")

