# Practice: Mean, Median, Mode, and Outlier Impact

def compute_mean(data):
    return sum(data) / len(data)


def compute_median(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 1:
        return sorted_data[mid]
    return (sorted_data[mid - 1] + sorted_data[mid]) / 2.0


def compute_mode(data):
    counts = {}
    for item in data:
        counts[item] = counts.get(item, 0) + 1
    max_count = max(counts.values())
    modes = [val for val, count in counts.items() if count == max_count]
    return modes


# Dataset representing monthly salaries (in thousands)
normal_salaries = [30, 32, 35, 36, 38, 40, 42]
salaries_with_outlier = [30, 32, 35, 36, 38, 40, 1000]  # CEO joined

print("--- Clean Data ---")
print("Data:", normal_salaries)
print(f"Mean:   {compute_mean(normal_salaries):.2f}")
print(f"Median: {compute_median(normal_salaries):.2f}")
print(f"Mode:   {compute_mode(normal_salaries)}")

print("\n--- With Extreme Outlier (1000) ---")
print("Data:", salaries_with_outlier)
print(f"Mean:   {compute_mean(salaries_with_outlier):.2f} (skewed drastically!)")
print(f"Median: {compute_median(salaries_with_outlier):.2f} (stable and robust!)")
