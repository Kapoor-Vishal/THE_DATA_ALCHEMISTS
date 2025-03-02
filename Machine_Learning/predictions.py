import pandas as pd
import joblib

# Load the trained best model
model = joblib.load("best_model.pkl")

# Load new data for prediction
new_data = pd.read_csv("../data/processed/cleaned_data.csv")  # Replace with your actual file

# Selecting same features as used in training
features = ["Revenue (USD)", "Operating Cost (USD)", "Load Factor (%)"]
X_new = new_data[features]

# Make predictions
new_data["Predicted Profit (USD)"] = model.predict(X_new)

# Save predictions to a CSV file
new_data.to_csv("../data/predicted_output.csv", index=False)

print("✅ Predictions saved successfully in 'predicted_output.csv'!")
