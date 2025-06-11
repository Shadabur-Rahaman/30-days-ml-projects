from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load('sentiment_model.joblib')

# Create FastAPI instance
app = FastAPI(title='Sentiment Analysis API')

# Define request body model
class TextInput(BaseModel):
    text: str

# Health check endpoint
@app.get('/')
def health_check():
    return {'status': 'healthy'}

# Prediction endpoint
@app.post('/predict')
def predict_sentiment(input_data: TextInput):
    """
    Predict sentiment for input text
    
    Args:
        input_data: JSON with 'text' field
    
    Returns:
        JSON with sentiment prediction and confidence
    """
    # Make prediction
    prediction = model.predict([input_data.text])
    probabilities = model.predict_proba([input_data.text])[0]
    
    # Get confidence score
    confidence = float(np.max(probabilities))
    
    return {
        'text': input_data.text,
        'sentiment': prediction[0],
        'confidence': confidence
    }
