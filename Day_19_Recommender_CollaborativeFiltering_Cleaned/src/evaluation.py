from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_model(true_ratings, predicted_ratings):
    rmse = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
    mae = mean_absolute_error(true_ratings, predicted_ratings)
    return rmse, mae
