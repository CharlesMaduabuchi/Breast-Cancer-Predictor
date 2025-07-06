# app.py
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import logging

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Define the input features globally
feature_names = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean", "concave_point_mean"
]


# Load your trained XGBoost model
try:
    model = joblib.load("xgb_breast_cancer_model.pkl")
    app.logger.info("Model loaded successfully.")
except Exception as e:
    model = None
    app.logger.error(f"Model loading failed: {e}")

# Home route
@app.route("/")
def home():
    return render_template("index.html", feature_names=feature_names)

# API prediction route
@app.route("/predict", methods=["POST"])
def predict_api():
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    data = request.json
    features = data.get('features', None)

    if features is None or len(features) != 8:
        return jsonify({'error': 'Please provide 8 numeric features.'}), 400

    try:
        features = np.array(features, dtype=float).reshape(1, -1)
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]
        result = "Cancerous" if prediction == 1 else "Non-Cancerous"
        app.logger.info(f"Prediction: {result} | Confidence: {proba}")
        return jsonify({
            "prediction": result,
            "probability": round(float(proba), 4),
            "message": f"The patient is predicted to be {result.lower()} with {round(proba * 100, 2)}% confidence."
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# Form prediction route
@app.route("/predict-form", methods=["POST"])
def predict_form():
    try:
        input_features = [float(request.form.get(f)) for f in feature_names]
        features = np.array(input_features).reshape(1, -1)
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]
        result = "Cancerous" if prediction == 1 else "Non-Cancerous"
        return render_template(
            "index.html",
            feature_names=feature_names,
            prediction=result,
            confidence=f"{proba*100:.2f}%"
        )
    except Exception as e:
        return render_template(
            "index.html",
            feature_names=feature_names,
            error=f"Prediction failed: {str(e)}"
        )
    
# ✅ Contact page route
@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")

        app.logger.info(f"[CONTACT] From: {name} <{email}> | Message: {message}")

        return render_template("contact.html", message_sent=True)

    return render_template("contact.html", message_sent=False)

# Services
@app.route("/services")
def services():
    return render_template("services.html")


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

