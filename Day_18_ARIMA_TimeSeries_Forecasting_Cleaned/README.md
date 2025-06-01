# ARIMA Time Series Forecasting

This project demonstrates how to use **ARIMA** for univariate time series forecasting. The focus is on predicting future stock prices using historical data.

![Forecast Plot](images/original_vs_forecast_plot.png)

## 🔍 Overview

- Dataset: Historical stock prices
- Model: ARIMA (AutoRegressive Integrated Moving Average)
- Tools: Statsmodels, Pandas, Matplotlib
- Evaluation: AIC/BIC scores, residual diagnostics, visualization

## 📌 Highlights

- Visualized original time series data
- Checked for stationarity using ADF test
- Determined (p, d, q) using ACF and PACF
- Fit and forecast using ARIMA
- Compared predictions against actuals

## 🧰 File Structure
```bash
Day_18_ARIMA_TimeSeries_Forecasting_Cleaned/
├── notebooks/
│   └── Day18_ARIMA_TimeSeries_Forecasting_Cleaned.ipynb  # Main notebook
├── images/
│   ├── synthetic_data_table.png     
│   └── arima_forecast.png
    └── arima_summary.txt
    └── synthetic_stock_prices.png              
├── src/
│   └── arima_utils.py                    # Utility functions: differencing, ADF test, model fitting
├── requirements.txt                      # Dependencies: statsmodels, pandas, matplotlib, etc.
├── .gitignore                            # Ignore cache, checkpoints, etc.
└── README.md                             # Project summary, usage, insights
See folder structure above.

## 🚀 How to Run


pip install -r requirements.txt
cd notebooks/
jupyter notebook Day18_ARIMA_TimeSeries_Forecasting_Cleaned.ipynb
📚 Learning Outcomes
Mastered time series preprocessing

Understood ADF, ACF, PACF in forecasting

Applied ARIMA model for real-world prediction
