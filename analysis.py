import argparse
import csv
from pathlib import Path


parser = argparse.ArgumentParser(
    description="Analyse a behavioural baseline results file."
)

parser.add_argument(
    "results_file",
    type=Path,
    help="Path to the baseline_results.csv file."
)

args = parser.parse_args()

# Only valid behavioural responses should be included in later analysis
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

print(f"Valid responses: {len(valid_responses)}")