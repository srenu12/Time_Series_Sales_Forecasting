from prophet import Prophet
import pandas as pd

def train_prophet_model(df):
    # Prophet requires a specific format: two columns ['ds', 'y']
    df_prophet = df.reset_index().rename(columns={'date_column': 'ds', 'sales_column': 'y'})  # Update column names
    
    # Train Prophet model
    model = Prophet()
    model.fit(df_prophet)
    
    # Create a future dataframe
    future = model.make_future_dataframe(periods=12, freq='M')  # Forecast for next 12 months
    forecast = model.predict(future)
    
    return forecast

def plot_prophet_forecast(df, forecast):
    model = Prophet()
    model.fit(df.rename(columns={'date_column': 'ds', 'sales_column': 'y'}))  # Update column names
    
    model.plot(forecast)
    plt.show()

if __name__ == "__main__":
    from data_loading import load_data, preprocess_data
    file_path = 'your_dataset.csv'
    df = load_data(file_path)
    df_cleaned = preprocess_data(df)
    
    # Train Prophet model and make forecast
    forecast = train_prophet_model(df_cleaned)
    
    # Plot Prophet forecast
    plot_prophet_forecast(df_cleaned, forecast)
