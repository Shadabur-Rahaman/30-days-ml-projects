import joblib
import numpy as np
from src.schema import InputData

def load_model(model_path: str = "model/model.pkl"):
    return joblib.load(model_path)

def make_prediction(model, data: InputData):
    features = np.array([[data.feature1, data.feature2, data.feature3]])
    pred = model.predict(features)
    return int(pred[0])
