import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

file_path = "../surprisals/cleaned_surprisal_scores.json"
with open(file_path, "r") as file:
    cleaned_surprisal_data = json.load(file)

#print(cleaned_surprisal_data)

data = []
for entry in cleaned_surprisal_data:
    for token, surprisal in entry["surprisal_scores"]:
        probability = np.exp(-surprisal)
        data.append({"token": token, "surprisal": surprisal, "probability": probability})

#print(data)

df = pd.DataFrame(data)

#Define features (X) and target (y)
X = df["surprisal"].values.reshape(-1, 1)
y = df["probability"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

#Evaluate the model on training and test data
y_train_pred = regression_model.predict(X_train)
y_test_pred = regression_model.predict(X_test)

#Calculate performance
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)


print(f"Training MSE: {train_mse}")
print(f"Test MSE: {test_mse}")
print(f"Training R-squared: {train_r2}")
print(f"Test R-squared: {test_r2}")

#Plot relationship using the test set
'''
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.5, label="Actual (Test Data)", s=10)
plt.plot(X_test, y_test_pred, color="red", label="Regression Line (Test)")
plt.xlabel("Surprisal Score")
plt.ylabel("Probability")
plt.title("Surprisal Score vs. Probability with Regression Line (Test Set)")
plt.legend()
plt.show()
'''

output_file_path = "regression_results.json"
df["predicted_probability"] = regression_model.predict(X)
df["predicted_probability"] = np.clip(df["predicted_probability"], 0, 1)

output_df = df[["token", "predicted_probability"]]
output_df.to_json(output_file_path, orient="records", index=False)

print(f"Probabilities saved to {output_file_path}")

