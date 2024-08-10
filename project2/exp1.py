import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Fonksiyon ve sınıf tanımları

def preprocess_data(df):
    df['Date'] = pd.to_datetime(df.index)
    df.set_index('Date', inplace=True)
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['Weekday'] = df.index.to_series().dt.weekday
    print("Preprocessed Data:")
    print(df.head())  # Eklenen satır
    return df

def perform_eda(df):
    print("Exploratory Data Analysis")
    print("Descriptive Statistics")
    print(df.describe())
    # Grafik oluşturma kodu (Plotly vb.)

def feature_engineering(df):
    df['Rolling_Mean'] = df['Close'].rolling(window=20).mean()
    df['Rolling_Std'] = df['Close'].rolling(window=20).std()
    return df

def create_pipeline():
    return Pipeline(steps=[
        ('data_prep', ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['Close'])
            ],
            remainder='passthrough'
        )),
        ('sequence_gen', SequenceGenerator()),
        ('lstm_model', LSTMModel())
    ])

class SequenceGenerator:
    def transform(self, data):
        X, y = [], []
        for i in range(len(data) - 60):
            X.append(data[i:i+60])
            y.append(data[i+60])
        return np.array(X), np.array(y)

class LSTMModel:
    def __init__(self):
        self.model = None
        
    def fit(self, X, y):
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            LSTM(50),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(X, y, epochs=10, batch_size=32)

    def predict(self, X):
        return self.model.predict(X)

    def inverse_transform(self, predictions):
        return predictions

def main():
    # Test with example stock data
    selected_stock = 'AAPL'
    start_date = '2024-01-01'
    end_date = '2024-08-01'
    
    # Download historical data
    df = yf.download(selected_stock, start=start_date, end=end_date)
    
    if df.empty:
        print(f"No data found for {selected_stock}.")
        return

    # Preprocess and prepare data
    df = preprocess_data(df)
    perform_eda(df)
    df = feature_engineering(df)

    # Initialize the pipeline
    pipeline = create_pipeline()
    
    # Prepare data
    data = df[['Close']]
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * .95))

    X_train, y_train = pipeline.named_steps['sequence_gen'].transform(
        pipeline.named_steps['data_prep'].transform(df)
    )

    # Train the pipeline
    pipeline.named_steps['lstm_model'].fit(X_train, y_train)

    # Create the testing data set
    test_data = pipeline.named_steps['data_prep'].transform(df)[training_data_len - 60:, :]
    X_test = pipeline.named_steps['sequence_gen'].transform(test_data)[0]
    
    # Get the model's predicted price values
    predictions = pipeline.named_steps['lstm_model'].predict(X_test)
    predictions = pipeline.named_steps['lstm_model'].inverse_transform(predictions)
    
    # Calculate the prediction dates
    prediction_dates = pd.date_range(end=end_date, periods=len(predictions) + 1, freq='B')[1:]
    
    # Plot the data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual Price'))
    fig.add_trace(go.Scatter(x=prediction_dates, y=predictions.flatten(), mode='lines', name='Predicted Price'))
    fig.update_layout(title=f'{selected_stock} Predicted Prices',
                      xaxis_title='Date',
                      yaxis_title='Price')
    fig.show()

if __name__ == "__main__":
    main()