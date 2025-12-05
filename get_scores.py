import csv
import argparse

# Argument parser for command line arguments
parser = argparse.ArgumentParser(
    description="Script to obtain average StrongREJECT scores from a file.",
)
parser.add_argument(
    "path",
    type=str,
    help="the path to the CSV file containing StrongREJECT scores",
)

args = parser.parse_args()

scores = {
    "refusal": 0,
    "convincingness": 0,
    "specificity": 0,
    "score": 0,
}
data_points = 0

with open(args.path, "r") as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        try:
            scores["refusal"] += float(row["refusal"])
            scores["convincingness"] += float(row["convincingness"])
            scores["specificity"] += float(row["specificity"])
            scores["score"] += float(row["score"])

            data_points += 1
        except ValueError:
            continue

# Calculate averages
for key in scores:
    scores[key] /= data_points

# Print the average scores
print("Average StrongREJECT scores:")
for key, value in scores.items():
    print(f"{key}: {value:.4f}")