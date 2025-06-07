from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model

def build_autoencoder(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(128, activation='relu'),
        Dense(np.prod(input_shape), activation='sigmoid'),
        Reshape(input_shape)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def save_model(model, path='artifacts/autoencoder_model.h5'):
    model.save(path)

def load_saved_model(path='artifacts/autoencoder_model.h5'):
    return load_model(path)
