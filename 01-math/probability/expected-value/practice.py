# Practice: Discrete Probability Distributions and Expected Value
# Formula: E[X] = sum(x_i * P(x_i))

def expected_value(values, probabilities):
    if len(values) != len(probabilities):
        raise ValueError("Values and probabilities must have the same length.")
    return sum(v * p for v, p in zip(values, probabilities))


# Fair 6-sided die
values = [1, 2, 3, 4, 5, 6]
probabilities = [1 / 6] * 6

ev = expected_value(values, probabilities)

print("Values:", values)
print(f"Probability of each value: {probabilities[0]:.4f} (1/6 or {probabilities[0] * 100:.2f}%)")
print(f"Expected Value E[X]: {ev:.2f}")  # 3.50
