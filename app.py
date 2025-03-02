import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("Machine_Learning/best_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Airline Profit Prediction API!", "status": "Running"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Ensure data is formatted as a DataFrame
        df = pd.DataFrame([data])  # Convert single dict into DataFrame

        # Make prediction
        prediction = model.predict(df)

        # Return response
        return jsonify({"predicted_profit": prediction.tolist()}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Set dynamic port for cloud deployment (default: 5000)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
