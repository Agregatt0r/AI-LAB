# Practice: Prediction Aggregation in Random Forests
# Demonstrates:
#   1. Hard Voting (Majority Vote) vs Soft Voting (Probability Averaging)
#   2. Edge case where Soft Voting resolves conflicting weak vs strong predictions
#   3. Regression Averaging & Tree-Disagreement Uncertainty Intervals

import numpy as np


def hard_voting(tree_predictions):
    """Aggregate discrete class predictions by majority vote.

    Args:
        tree_predictions: array of shape (n_trees, n_samples)
    """
    n_trees, n_samples = tree_predictions.shape
    final_preds = []
    for j in range(n_samples):
        classes, counts = np.unique(tree_predictions[:, j], return_counts=True)
        final_preds.append(classes[np.argmax(counts)])
    return np.array(final_preds)


def soft_voting(tree_probabilities):
    """Aggregate class probabilities by arithmetic average.

    Args:
        tree_probabilities: array of shape (n_trees, n_samples, n_classes)
    """
    # Mean probabilities across trees (axis 0): shape (n_samples, n_classes)
    mean_probs = np.mean(tree_probabilities, axis=0)
    final_preds = np.argmax(mean_probs, axis=1)
    return final_preds, mean_probs


def regression_aggregation(tree_outputs):
    """Compute ensemble point prediction and uncertainty (std across trees).

    Args:
        tree_outputs: array of shape (n_trees, n_samples)
    """
    point_predictions = np.mean(tree_outputs, axis=0)
    uncertainty_std = np.std(tree_outputs, axis=0)
    return point_predictions, uncertainty_std


if __name__ == "__main__":
    # -------------------------------------------------------------
    # 1. Classification: Hard Voting vs Soft Voting
    # -------------------------------------------------------------
    print("--- 1. Classification: Hard Voting vs Soft Voting ---")
    # 3 Trees predicting on an ambiguous sample:
    # Tree 1: [0.51, 0.49] -> Predicts 0 (Weak)
    # Tree 2: [0.52, 0.48] -> Predicts 0 (Weak)
    # Tree 3: [0.05, 0.95] -> Predicts 1 (Extremely Confident)
    tree_probs = np.array([
        [[0.51, 0.49]],
        [[0.52, 0.48]],
        [[0.05, 0.95]],
    ])  # shape (3 trees, 1 sample, 2 classes)

    # Derive hard labels from each tree
    tree_hard_labels = np.argmax(tree_probs, axis=2)  # shape (3, 1)

    hard_pred = hard_voting(tree_hard_labels)[0]
    soft_pred, avg_probs = soft_voting(tree_probs)

    print(f"Tree 1 Probabilities: {tree_probs[0, 0].tolist()} -> Hard Vote: {tree_hard_labels[0, 0]}")
    print(f"Tree 2 Probabilities: {tree_probs[1, 0].tolist()} -> Hard Vote: {tree_hard_labels[1, 0]}")
    print(f"Tree 3 Probabilities: {tree_probs[2, 0].tolist()} -> Hard Vote: {tree_hard_labels[2, 0]}")
    print(f"\nHard Voting Majority Result: Class {hard_pred} (Two weak votes defeated one confident vote)")
    print(f"Soft Voting Average Probs:   {np.round(avg_probs[0], 4).tolist()} -> Predicts Class {soft_pred[0]}")

    assert hard_pred == 0, "Hard vote should be class 0"
    assert soft_pred[0] == 1, "Soft vote should be class 1 due to high confidence of Tree 3"

    # -------------------------------------------------------------
    # 2. Regression: Point Prediction & Uncertainty Intervals
    # -------------------------------------------------------------
    print("\n--- 2. Regression Aggregation & Uncertainty Bounds ---")
    # 15 Trees predicting house prices on 2 test houses:
    # House A: Familiar suburban home (trees agree closely)
    # House B: Unusual luxury mansion (trees disagree widely)
    rng = np.random.default_rng(42)

    # House A predictions centered at $300k with low variance (scale=8,000)
    house_a_trees = rng.normal(loc=300000, scale=8000, size=15)
    # House B predictions centered at $850k with high variance (scale=75,000)
    house_b_trees = rng.normal(loc=850000, scale=75000, size=15)

    # Shape: (15 trees, 2 samples)
    reg_outputs = np.column_stack([house_a_trees, house_b_trees])

    preds, stds = regression_aggregation(reg_outputs)

    print(f"House A (Common type): Predicted Price = ${preds[0]:,.0f} ± ${stds[0]:,.0f} (High Confidence)")
    print(f"House B (Rare type):   Predicted Price = ${preds[1]:,.0f} ± ${stds[1]:,.0f} (High Uncertainty)")

    assert stds[0] < stds[1] * 0.25, "House A uncertainty should be much lower than House B"

    print("\nPrediction aggregation logic successfully verified!")
