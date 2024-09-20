import pandas as pd
import numpy as np

# Load the dataset (replace with your own dataset path)
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['date_column'])  # Update 'date_column' with the actual column name
    return df

def preprocess_data(df):
    # Handle missing values (drop or fill them, depending on your needs)
    df = df.dropna()  # Or df.fillna(method='ffill')
    
    # Set date column as index
    df.set_index('date_column', inplace=True)  # Replace 'date_column' with the actual name of your date column
    
    # Optional: Resample data if needed (e.g., daily to monthly)
    df = df.resample('M').sum()  # For monthly sales aggregation
    
    return df

if __name__ == "__main__":
    file_path = 'your_dataset.csv'  # Update with the actual file name
    df = load_data(file_path)
    df_cleaned = preprocess_data(df)
    print(df_cleaned.head())
