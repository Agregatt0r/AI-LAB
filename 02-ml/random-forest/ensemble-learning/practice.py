# Practice: Condorcet's Jury Theorem & The Power of Diversity
# Demonstrates:
#   1. Theoretical Binomial calculation of ensemble majority vote accuracy
#   2. Simulation of independent base models vs majority voting ensemble
#   3. Impact of error correlation (diversity vs homogeneous models)

import math
import numpy as np


def condorcet_probability(B, p):
    """Compute theoretical majority vote accuracy for B independent models with accuracy p."""
    majority_threshold = math.ceil(B / 2)
    # Sum binomial probabilities from majority_threshold to B
    prob_majority = 0.0
    for k in range(majority_threshold, B + 1):
        comb = math.comb(B, k)
        prob_majority += comb * (p ** k) * ((1.0 - p) ** (B - k))
    return prob_majority


def simulate_ensemble(n_samples=2000, n_models=51, individual_acc=0.60, seed=42):
    """Simulate independent binary classifiers and evaluate their ensemble vote."""
    rng = np.random.default_rng(seed)

    # True labels (binary 0 or 1)
    y_true = rng.integers(0, 2, size=n_samples)

    # Matrix of model predictions: shape (n_models, n_samples)
    # Each model has probability `individual_acc` of predicting correctly
    correct_masks = rng.random((n_models, n_samples)) < individual_acc
    model_predictions = np.where(correct_masks, y_true, 1 - y_true)

    # Majority vote across models (axis 0)
    vote_sums = np.sum(model_predictions, axis=0)
    ensemble_predictions = (vote_sums >= (n_models / 2.0)).astype(int)

    # Accuracies
    individual_accuracies = [np.mean(model_predictions[i] == y_true) for i in range(n_models)]
    ensemble_accuracy = np.mean(ensemble_predictions == y_true)

    return np.mean(individual_accuracies), ensemble_accuracy


if __name__ == "__main__":
    # 1. Theoretical Accuracy Growth via Condorcet's Theorem
    print("--- 1. Theoretical Majority Vote Accuracy (Base Accuracy p = 0.60) ---")
    print(f"{'Ensemble Size (B)':<20} | {'Theoretical Majority Vote Accuracy'}")
    print("-" * 55)

    base_p = 0.60
    sizes = [1, 5, 11, 25, 51, 101]
    for b in sizes:
        p_ens = condorcet_probability(b, base_p)
        print(f"B = {b:<16} | {p_ens * 100:>6.2f}%")

    assert condorcet_probability(1, base_p) == base_p
    assert condorcet_probability(51, base_p) > 0.90
    assert condorcet_probability(101, base_p) > 0.97

    # 2. Empirical Simulation of 51 Independent Models
    print("\n--- 2. Empirical Simulation (2,000 samples, 51 independent models) ---")
    avg_indiv_acc, emp_ensemble_acc = simulate_ensemble(n_samples=2000, n_models=51, individual_acc=0.60)

    print(f"Average Individual Model Accuracy: {avg_indiv_acc * 100:.2f}%")
    print(f"Empirical Ensemble Accuracy:       {emp_ensemble_acc * 100:.2f}%")
    print(f"Accuracy Boost from Ensembling:    +{(emp_ensemble_acc - avg_indiv_acc) * 100:.2f}%")

    assert emp_ensemble_acc > avg_indiv_acc + 0.30, "Ensemble should give >30% accuracy boost over weak individual models"

    # 3. Demonstration of Correlated Models (Lack of Diversity)
    print("\n--- 3. Effect of Correlation (Identical / Cloned Models) ---")
    rng = np.random.default_rng(42)
    y_test = rng.integers(0, 2, size=1000)

    # 5 Identical models that make the exact same predictions
    single_model_preds = np.where(rng.random(1000) < 0.60, y_test, 1 - y_test)
    cloned_models = np.tile(single_model_preds, (5, 1))
    cloned_vote = (np.mean(cloned_models, axis=0) >= 0.5).astype(int)
    cloned_acc = np.mean(cloned_vote == y_test)

    print(f"5 Identical Cloned Models Ensemble Accuracy: {cloned_acc * 100:.2f}%")
    print("-> Notice: Without diversity, an ensemble gains 0% improvement over a single model!")

    assert np.isclose(cloned_acc, np.mean(single_model_preds == y_test))

    print("\nEnsemble learning and diversity principles successfully verified!")

