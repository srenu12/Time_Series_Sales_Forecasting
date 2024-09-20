import matplotlib.pyplot as plt
import seaborn as sns

def plot_time_series(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df, label="Sales over Time")
    plt.title("Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.show()

def plot_decomposition(df):
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(df, model='additive')
    decomposition.plot()
    plt.show()

if __name__ == "__main__":
    from data_loading import load_data, preprocess_data
    file_path = 'your_dataset.csv'
    df = load_data(file_path)
    df_cleaned = preprocess_data(df)
    
    # Plot sales over time
    plot_time_series(df_cleaned)
    
    # Decompose time series into trend, seasonal, and residual components
    plot_decomposition(df_cleaned)
