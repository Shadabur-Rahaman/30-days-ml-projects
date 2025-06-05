import joblib
from tensorflow.keras.models import load_model

def load_tabular_model(path='artifacts/tabular_model.joblib'):
    """Load a saved scikit-learn model for tabular data."""
    return joblib.load(path)

def load_cnn_model(path='artifacts/cnn_model.h5'):
    """Load a saved CNN model for image data."""
    return load_model(path)
