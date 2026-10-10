# Practice: Bayes' Theorem
# P(A|B) = [P(B|A) * P(A)] / P(B)

def bayes(prior, likelihood, false_positive):
    prior_not_a = 1 - prior
    evidence = (likelihood * prior) + (false_positive * prior_not_a)
    posterior = (likelihood * prior) / evidence
    return evidence, posterior


# Disease scenario
prior = 0.2           # P(Disease)
likelihood = 0.8      # P(Positive Test | Disease)
false_positive = 0.1  # P(Positive Test | Healthy)

p_b, p_a_given_b = bayes(prior, likelihood, false_positive)

print(f"Prior P(A): {prior:.2f} ({prior * 100:.1f}%)")
print(f"Likelihood P(B|A): {likelihood:.2f} ({likelihood * 100:.1f}%)")
print(f"Total Evidence Probability P(B): {p_b:.4f} ({p_b * 100:.2f}%)")
print(f"Posterior Probability P(A|B): {p_a_given_b:.4f} ({p_a_given_b * 100:.2f}%)")
