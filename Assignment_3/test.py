import os
import time
import requests
import joblib
import subprocess
import json
from score import score
import app  
import score  

# Load the trained model
model = joblib.load("best_model.pkl")

# Unit Tests
def test_score_smoke_test():
    """Ensure the function executes without errors (sanity test)."""
    output = score("Sample input", model, 0.5)
    assert output is not None

def test_score_return_types():
    """Check if the function returns expected data types."""
    output = score("Sample input", model, 0.5)
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert isinstance(output[0], bool)  # Prediction must be True/False
    assert isinstance(output[1], float)  # Propensity must be a float

def test_score_prediction_valid():
    """Confirm the prediction is always binary (True/False)."""
    prediction, _ = score("Example message", model, 0.5)
    assert prediction in [True, False]

def test_score_propensity_range():
    """Ensure the propensity score lies between 0 and 1."""
    _, probability = score("Example message", model, 0.5)
    assert 0.0 <= probability <= 1.0

def test_lower_threshold():
    """When threshold is 0, prediction should always be True."""
    prediction, _ = score("Random input text", model, 0.0)
    assert prediction is True

def test_upper_threshold():
    """When threshold is 1, prediction should always be False."""
    prediction, _ = score("Another random text", model, 1.0)
    assert prediction is False

def test_clear_spam():
    """Check behavior on a well-known spam phrase."""
    spam_text = "Congratulations! You've won a FREE iPhone. Click the link to claim."
    prediction, probability = score(spam_text, model, 0.5)
    assert prediction is True
    assert probability > 0.5

def test_clear_non_spam():
    """Check behavior on a typical non-spam message."""
    non_spam_text = "Hey, are we still meeting for lunch tomorrow?"
    prediction, probability = score(non_spam_text, model, 0.5)
    assert prediction is False
    assert probability <= 0.5

def test_flask():
    """Integration test: Start Flask app, send request, and verify response."""

    # Start Flask app in the background
    flask_process = subprocess.Popen(["python", "app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(3)  # Wait for server to start

    # Define test payload (spam message)
    test_data = {
        "text": "do you want anytime any network mins text and a new video phone for only five pounds per week call now or reply for delivery tomorrow"
    }

    # Send POST request to Flask API
    response = requests.post("http://127.0.0.1:5000/score", json=test_data)

    # Parse JSON response
    json_response = response.json()

    # Print API response
    print("\n🔹 API Response:")
    print(json.dumps(json_response, indent=4))

    # Validate response
    assert response.status_code == 200
    assert "prediction" in json_response
    assert "propensity" in json_response
    assert isinstance(json_response["prediction"], bool)
    assert isinstance(json_response["propensity"], float)
    assert 0.0 <= json_response["propensity"] <= 1.0  # Ensure valid probability range

    # Stop Flask app gracefully
    flask_process.terminate()
    print("\n Flask app terminated.")

if __name__ == "__main__":
    # Run all tests
    test_score_smoke_test()
    test_score_return_types()
    test_score_prediction_valid()
    test_score_propensity_range()
    test_lower_threshold()
    test_upper_threshold()
    test_clear_spam()
    test_clear_non_spam()
    test_flask()
