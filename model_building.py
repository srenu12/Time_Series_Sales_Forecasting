from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def train_arima_model(df, order=(5, 1, 0)):
    # Split data into train and test sets
    train_size = int(len(df) * 0.8)
    train, test = df[:train_size], df[train_size:]

    # Fit ARIMA model
    model = ARIMA(train, order=order)
    model_fit = model.fit()

    # Forecast
    forecast = model_fit.forecast(steps=len(test))
    return train, test, forecast

if __name__ == "__main__":
    from data_loading import load_data, preprocess_data
    file_path = 'your_dataset.csv'
    df = load_data(file_path)
    df_cleaned = preprocess_data(df)

    # Train ARIMA model
    train, test, forecast = train_arima_model(df_cleaned)
    
    # Show forecast vs actual sales
    print("Actual Sales:\n", test)
    print("Forecasted Sales:\n", forecast)
