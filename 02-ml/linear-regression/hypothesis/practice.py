# Practice: Linear Hypothesis (Forward Prediction)
# Formula: y_hat = w * x + b

def predict(x, w, b):
    """Compute predicted value y_hat for a given input x."""
    return w * x + b


if __name__ == "__main__":
    # House sizes (square meters) and actual selling prices ($)
    x = [40, 60, 80, 100]
    y_true = [150000, 220000, 290000, 360000]

    # Model parameters
    w = 3500
    b = 10000

    # Generate predictions for all houses
    y_pred = [predict(val, w, b) for val in x]

    print("Inputs (x):      ", x)
    print("Actual (y):      ", y_true)
    print("Predictions (ŷ): ", y_pred)

    # Verification
    assert y_pred == [150000, 220000, 290000, 360000], "Predictions mismatch!"
    print("\nAll predictions match expected values!")
