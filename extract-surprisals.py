import json
from surprisal import AutoHuggingFaceModel


def extract_data(file_path):
    sentences = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if "model_output_text" in line:
                sentences.append(data["model_output_text"])
    return sentences


def save_surprisal_scores(sentences, output_file):
    m = AutoHuggingFaceModel.from_pretrained("gpt2")

    results = []
    for sentence, surprisal_result in zip(sentences, m.surprise(sentences)):
        words = sentence.split()
        surprisal_scores = list(surprisal_result)

        result = {
            "sentence": sentence,
            "words": words,
            "surprisal_scores": surprisal_scores,
        }
        results.append(result)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)


file_path = "datasets/val/mushroom.en-val.v2.jsonl"
output_file = "surprisal_scores.json"
sentences = extract_data(file_path)
save_surprisal_scores(sentences, output_file)
