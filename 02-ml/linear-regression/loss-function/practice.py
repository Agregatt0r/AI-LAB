# Practice: Loss Function (Mean Squared Error & Parameter Search)
# Formula: MSE = (1/n) * Σ(y_true - y_pred)^2

def predict(x, w, b):
    """Compute predicted values for a scalar or list of inputs."""
    return w * x + b


def mse(y_true, y_pred):
    """Compute Mean Squared Error between true and predicted targets."""
    n = len(y_true)
    squared_errors = [(y_t - y_p) ** 2 for y_t, y_p in zip(y_true, y_pred)]
    return sum(squared_errors) / n


def mae(y_true, y_pred):
    """Compute Mean Absolute Error between true and predicted targets."""
    n = len(y_true)
    abs_errors = [abs(y_t - y_p) for y_t, y_p in zip(y_true, y_pred)]
    return sum(abs_errors) / n


if __name__ == "__main__":
    x = [1, 2, 3]
    y_true = [3, 5, 7]

    # Test candidate parameter combinations (w, b)
    parameters = [
        (1, 1),
        (2, 1),
        (3, 0),
        (0, 5),
    ]

    print("Evaluating candidate parameters on dataset:")
    print(f"Inputs x: {x}")
    print(f"Targets y: {y_true}\n")

    best_params = None
    lowest_error = float("inf")

    for w, b in parameters:
        y_pred = [predict(val, w, b) for val in x]
        error_mse = mse(y_true, y_pred)
        error_mae = mae(y_true, y_pred)

        print(f"w={w}, b={b:2d} -> ŷ={y_pred} | MSE={error_mse:.4f} | MAE={error_mae:.4f}")

        if error_mse < lowest_error:
            lowest_error = error_mse
            best_params = (w, b)

    print(f"\nBest parameter pair found: w={best_params[0]}, b={best_params[1]} with MSE={lowest_error}")
    assert best_params == (2, 1) and lowest_error == 0.0
