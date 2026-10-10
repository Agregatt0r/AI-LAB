# Practice: Basic Probability and Complement Rule

def probability(favourable, total):
    return favourable / total


red = 3
blue = 7
total = red + blue

p_red = probability(red, total)
p_blue = probability(blue, total)
p_not_red = 1 - p_red

print(f"Total items: {total}")
print(f"Probability of red: {p_red:.2f} ({p_red * 100:.1f}%)")
print(f"Probability of blue: {p_blue:.2f} ({p_blue * 100:.1f}%)")
print(f"Probability of not red: {p_not_red:.2f} ({p_not_red * 100:.1f}%)")
