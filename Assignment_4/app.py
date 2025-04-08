import numpy as np
import joblib
import re
import sklearn
from flask import Flask, request, jsonify

# Initialize Flask App
app = Flask(__name__)

# Load Model & Vectorizer
model = joblib.load('best_model.pkl')  
vectorizer = joblib.load('vectorizer.joblib')

def preprocess_message(message: str) -> str:
    """Preprocess a single message: Lowercase, remove non-alphabet characters, and tokenize."""
    cleaned_message = ' '.join(re.findall(r'\b[a-zA-Z]+\b', message.lower()))
    return cleaned_message

@app.route("/score", methods=["POST"])
def score():
    """Flask endpoint to predict the class of a given message."""
    data = request.get_json()
    
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400  # Return error if 'text' is missing

    text = data.get("text")
    threshold = data.get("threshold", 0.5)  # Default threshold = 0.5 if not provided

    # Preprocess and vectorize text
    text_vectorized = vectorizer.transform([preprocess_message(text)])

    # Get probability prediction
    propensity = model.predict_proba(text_vectorized)[0][1]

    # Determine prediction based on threshold
    prediction = propensity >= threshold

    # Return JSON response
    return jsonify({"prediction": bool(prediction), "propensity": float(propensity)})

if __name__ == "__main__":
    # app.run(port=5000, debug=True)
    app.run(host="0.0.0.0", port=5000, debug = True)
