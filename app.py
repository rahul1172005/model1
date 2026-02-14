from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load your trained model (update the path)
model = joblib.load("diabetes_model.pkl")

# If you used scaler, load it too (optional)
scaler = joblib.load("scaler.pkl")

# Define column order (must match training dataset)
FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]


# Home route
@app.route('/')
def home():
    return "<h1>Welcome to the Diabetes Prediction API!</h1>"


# Preprocess function
def preprocess_input(data):
    try:
        # Convert input JSON into DataFrame
        df = pd.DataFrame([data])

        # Ensure correct column order
        df = df[FEATURE_COLUMNS]

        # Scale input if scaler exists
        scaled_data = scaler.transform(df)

        return scaled_data

    except Exception as e:
        raise ValueError(f"Input preprocessing failed: {str(e)}")


# Prediction function
def predict_diabetes(features):
    try:
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        return int(prediction), float(probability)

    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")


# Predict API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "No JSON input provided"}), 400

        preprocessed_features = preprocess_input(data)
        predicted_class, predicted_proba = predict_diabetes(preprocessed_features)

        return jsonify({
            "prediction": predicted_class,
            "probability_of_diabetes": predicted_proba
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
