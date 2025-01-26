import json
import pandas as pd

#Load surprisal scores
file_path = "../surprisals/cleaned_surprisal_scores.json"
with open(file_path, "r") as file:
    cleaned_surprisal_data = json.load(file)

#Load regression results
regression_results_path = "regression_results.json"
with open(regression_results_path, "r") as file:
    regression_results = json.load(file)

probabilities_dict = {entry["token"]: entry["predicted_probability"] for entry in regression_results}

#Switch surprisal_scores with probabilities
for entry in cleaned_surprisal_data:
    entry["probabilities"] = [
        [token, probabilities_dict.get(token, 0.0)]
        for token, surprisal in entry["surprisal_scores"]
    ]
    del entry["surprisal_scores"]

#Save file
output_file_path = "probabilities.json"
with open(output_file_path, "w") as file:
    json.dump(cleaned_surprisal_data, file, indent=4)

print(f"File saved at {output_file_path}")