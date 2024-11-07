import json


def clean_surprisal_scores(input_file, output_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    for entry in data:
        filtered_scores = [
            (token, score)
            for token, score in entry["surprisal_scores"]
            if token != "<|endoftext|>" and score != float("inf")
        ]
        entry["surprisal_scores"] = filtered_scores

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)


input_file = "surprisal_scores.json"
output_file = "cleaned_surprisal_scores.json"
clean_surprisal_scores(input_file, output_file)
