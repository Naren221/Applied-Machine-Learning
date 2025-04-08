import numpy as np
from typing import Tuple
import joblib
import re
import sklearn


def preprocess_message(message: str) -> str:
    """Preprocess a single message: Lowercase, remove non-alphabet characters, and tokenize.
    
    Args:
        message (str): Input text message.
        
    Returns:
        str: Preprocessed text message.
    """
    # Lowercase the text and remove non-alphabetic characters
    cleaned_message = ' '.join(re.findall(r'\b[a-zA-Z]+\b', message.lower()))
    return cleaned_message

def score(text: str, model: sklearn.base.BaseEstimator, threshold: float) -> Tuple[bool, float]:
    
    vectorizer = joblib.load('vectorizer.joblib')
    text_vectorized = vectorizer.transform([preprocess_message(text)])
    
    # Get probability prediction
    propensity = model.predict_proba(text_vectorized)[0][1]
    
    # Determine prediction based on threshold
    prediction = propensity >= threshold
    
    return bool(prediction), float(propensity)

