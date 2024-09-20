import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_model(test, forecast):
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    
    return rmse, mae

def plot_forecast(test, forecast):
    plt.figure(figsize=(10, 6))
    plt.plot(test.index, test, label="Actual Sales")
    plt.plot(test.index, forecast, label="Forecasted Sales", color='red')
    plt.title("Sales Forecast vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    from model_building import train_arima_model
    from data_loading import load_data, preprocess_data
    file_path = 'your_dataset.csv'
    df = load_data(file_path)
    df_cleaned = preprocess_data(df)
    
    # Train model and get forecast
    train, test, forecast = train_arima_model(df_cleaned)
    
    # Evaluate the model
    evaluate_model(test, forecast)
    
    # Plot forecast vs actual sales
    plot_forecast(test, forecast)
