import yfinance as yf
import numpy as np
import pandas as pd

def calculate_bollinger_recommendation(selected_stock, start_date, end_date):
    # Fetch stock data
    stock_data = yf.Ticker(selected_stock)
    stock_history = stock_data.history(period='1d', start=start_date, end=end_date)

    # Check if 'Adj Close' column is present, otherwise use 'Close'
    if 'Adj Close' in stock_history.columns:
        stock_data = stock_history[['Adj Close']].copy()
        stock_data.rename(columns={'Adj Close': 'Close'}, inplace=True)
    elif 'Close' in stock_history.columns:
        stock_data = stock_history[['Close']].copy()
    else:
        return "Error: Neither 'Adj Close' nor 'Close' columns are available in the data."

    # Parameters
    period = 20

    # Calculate Bollinger Bands
    stock_data['SMA'] = stock_data['Close'].rolling(window=period).mean()
    stock_data['STD'] = stock_data['Close'].rolling(window=period).std()
    stock_data['Upper'] = stock_data['SMA'] + (stock_data['STD'] * 2)
    stock_data['Lower'] = stock_data['SMA'] - (stock_data['STD'] * 2)

    # Prepare new DataFrame for signals
    new_stock_data = stock_data[period-1:].copy()

    # Function to get buy and sell signals
    def get_signal_bb(data):
        buy_signal = []
        sell_signal = []
        
        # Vectorized operations for performance improvement
        buy_signal = np.where(data['Close'] < data['Lower'], data['Close'], np.nan)
        sell_signal = np.where(data['Close'] > data['Upper'], data['Close'], np.nan)
        
        return buy_signal, sell_signal

    # Add buy and sell signals to DataFrame
    new_stock_data['Buy'], new_stock_data['Sell'] = get_signal_bb(new_stock_data)

    # Determine the most recent signal
    latest_data = new_stock_data.iloc[-1]
    if latest_data['Close'] > latest_data['Upper']:
        return "Sell"
    elif latest_data['Close'] < latest_data['Lower']:
        return "Buy"
    else:
        return "Hold"

# Example usage
print(calculate_bollinger_recommendation('AAPL', '2023-01-01', '2023-12-31'))