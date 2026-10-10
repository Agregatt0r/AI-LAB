# Practice: Conditional Probability
# Formula: P(A|B) = P(A ∩ B) / P(B)

def conditional_probability(both, condition_total):
    return both / condition_total


total_students = 20
python_students = 12
csharp_students = 8
both_languages = 5

# Probability of knowing both: P(Python ∩ C#)
p_both = both_languages / total_students

# Probability of knowing Python: P(Python)
p_python = python_students / total_students

# Conditional Probability: P(C# | Python)
p_csharp_given_python = conditional_probability(both_languages, python_students)

print(f"Total students: {total_students}")
print(f"P(Python): {p_python:.2f} ({p_python * 100:.1f}%)")
print(f"P(Python ∩ C#): {p_both:.2f} ({p_both * 100:.1f}%)")
print(f"P(C# | Python): {p_csharp_given_python:.4f} ({p_csharp_given_python * 100:.2f}%)")
