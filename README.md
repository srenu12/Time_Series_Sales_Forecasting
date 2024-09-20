# Sales Forecasting using Time Series Analysis

## Project Overview
This project uses historical sales data to forecast future sales using time series analysis techniques. The project demonstrates the use of ARIMA and Prophet models to predict future sales trends.

## Project Structure
- `data_loading.py`: Loads and preprocesses the data.
- `eda.py`: Performs exploratory data analysis and time series decomposition.
- `model_building.py`: Implements ARIMA for sales forecasting.
- `model_evaluation.py`: Evaluates the ARIMA model and visualizes the forecast vs actual sales.
- `prophet_forecasting.py`: Implements forecasting using Prophet.

## Requirements
- Python 3.x
- Pandas, Numpy, Matplotlib, Seaborn
- Statsmodels, Scikit-learn
- Prophet

## How to Run
1. Install the required libraries: pip install pandas numpy matplotlib seaborn statsmodels scikit-learn prophet
2. Clone the repository.
3. Place your dataset (`your_dataset.csv`) in the project folder.
4. Run the `model_evaluation.py` file to see the ARIMA model forecast.
5. You can also run `prophet_forecasting.py` for Prophet-based forecasting.
