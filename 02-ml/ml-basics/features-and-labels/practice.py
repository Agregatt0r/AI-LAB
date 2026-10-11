# Practice: Preparing Feature Matrix X and Target Vector y from Scratch
# Includes One-Hot Encoding for categorical features and matrix shape inspection

import numpy as np


def one_hot_encode(categories_list):
    """Convert a 1D list of string categories into a 2D one-hot binary array."""
    unique_categories = sorted(list(set(categories_list)))
    category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}

    n = len(categories_list)
    k = len(unique_categories)
    encoded = np.zeros((n, k), dtype=float)

    for i, cat in enumerate(categories_list):
        col_idx = category_to_idx[cat]
        encoded[i, col_idx] = 1.0

    return encoded, unique_categories


if __name__ == "__main__":
    # Raw tabular records for housing price dataset:
    # [Size (m²), Bedrooms, City, Price ($)]
    raw_dataset = [
        {"size": 50.0,  "bedrooms": 1, "city": "Kyiv",   "price": 60000.0},
        {"size": 85.0,  "bedrooms": 2, "city": "Lviv",   "price": 95000.0},
        {"size": 120.0, "bedrooms": 3, "city": "Kyiv",   "price": 150000.0},
        {"size": 65.0,  "bedrooms": 2, "city": "Odesa",  "price": 75000.0},
        {"size": 110.0, "bedrooms": 3, "city": "Lviv",   "price": 130000.0},
    ]

    # 1. Extract numerical features
    numerical_features = np.array([
        [item["size"], float(item["bedrooms"])] for item in raw_dataset
    ])

    # 2. Extract and encode categorical feature (City)
    raw_cities = [item["city"] for item in raw_dataset]
    city_encoded, unique_cities = one_hot_encode(raw_cities)

    # 3. Combine into final Feature Matrix X: [Size, Bedrooms, City_Kyiv, City_Lviv, City_Odesa]
    X = np.hstack([numerical_features, city_encoded])

    # 4. Extract Target Vector y (Price)
    y = np.array([item["price"] for item in raw_dataset])

    print("--- 1. Extracted Dataset Dimensions ---")
    print(f"Feature Matrix X Shape: {X.shape} (n={X.shape[0]} samples, d={X.shape[1]} features)")
    print(f"Target Vector y Shape:   {y.shape} (n={y.shape[0]} targets)\n")

    print("--- 2. One-Hot Encoded City Columns ---")
    print(f"Categories: {unique_cities}")
    print(city_encoded, "\n")

    print("--- 3. Final Feature Matrix X ---")
    header = ["Size(m²)", "Bedrooms"] + [f"City_{c}" for c in unique_cities]
    print(f"{'Sample':<8} | " + " | ".join(f"{col:<10}" for col in header))
    print("-" * 75)
    for i, row in enumerate(X):
        formatted_row = " | ".join(f"{val:<10.1f}" for val in row)
        print(f"#{i+1:<7} | {formatted_row} ──> y = ${y[i]:,.0f}")

    # Assertions
    assert X.shape == (5, 5)
    assert y.shape == (5,)
    assert np.all(city_encoded.sum(axis=1) == 1.0), "Each one-hot row must sum to 1"
    print("\nFeature Matrix X and Target Vector y successfully constructed!")
