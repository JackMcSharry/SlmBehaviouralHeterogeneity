import argparse
import csv
from pathlib import Path
import statistics
from collections import Counter

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    description="Analyse a behavioural baseline results file."
)

parser.add_argument(
    "results_file",
    type=Path,
    help="Path to the baseline_results.csv file."
)

args = parser.parse_args()

with open(args.results_file, "r", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

valid_responses = [
    int(row["parsed_response"])
    for row in rows
    if row["valid"].strip().lower() == "true"
    and row["parsed_response"].strip()
]

if not valid_responses:
    raise ValueError("No valid responses found in the results file.")

# Calculate and print statistics
print(f"Valid responses: {len(valid_responses)}")
print(f"Mean: {statistics.mean(valid_responses):.2f}")
print(f"Median: {statistics.median(valid_responses):.2f}")
print(f"Variance: {statistics.pvariance(valid_responses):.2f}")
print(f"Standard deviation: {statistics.pstdev(valid_responses):.2f}")
print(f"Unique responses: {len(set(valid_responses))}")

# Generate and save a bar chart of the response distribution
response_values = list(range(11))
counts = Counter(valid_responses)
frequencies = [counts[value] for value in response_values]

plt.bar(response_values, frequencies)
plt.xticks(response_values)
plt.xlabel("Amount given (£)")
plt.ylabel("Number of participants")
plt.title("Baseline response distribution")
plt.tight_layout()

plot_path = args.results_file.parent / "response_distribution.png"
plt.savefig(plot_path, dpi=150)
plt.close()

print(f"Plot saved to: {plot_path}")