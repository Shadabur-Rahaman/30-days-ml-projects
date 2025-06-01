import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def check_stationarity(timeseries):
    result = adfuller(timeseries)
    return {'ADF Statistic': result[0], 'p-value': result[1]}

def plot_acf_pacf(series, lags=40):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plot_acf(series, lags=lags, ax=plt.gca())
    plt.subplot(122)
    plot_pacf(series, lags=lags, ax=plt.gca())
    plt.tight_layout()
    plt.show()

def fit_arima_model(series, order=(1, 1, 1)):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit

def forecast(model_fit, steps=10):
    forecast = model_fit.forecast(steps=steps)
    return forecast
