from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import logging
import time
import plotly.graph_objects as go

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        selected_stock = data['selected_stock']
        start_date = data['start_date']
        end_date = data['end_date']
        
        start_time = time.time()  # Start timing
        
        logging.info(f"Starting LSTM prediction for {selected_stock} from {start_date} to {end_date}.")
        
        # Download historical data
        logging.info("Downloading historical data.")
        df = yf.download(selected_stock, start=start_date, end=end_date)
        
        if df.empty:
            logging.warning(f"No data found for {selected_stock}.")
            return jsonify({"status": "error", "message": "No data found"}), 404

        # Preprocess Data
        logging.info("Starting data preprocessing.")
        df = preprocess_data(df)
        
        logging.info("Performing Exploratory Data Analysis (EDA).")
        eda = perform_eda(df)
        
        logging.info("Starting feature engineering.")
        df = feature_engineering(df)

        # Initialize the pipeline
        logging.info("Creating ML pipeline.")
        pipeline = create_pipeline()
        
        # Prepare the data
        logging.info("Preparing training and testing data.")
        data = df[['Close']].values
        training_data_len = int(np.ceil(len(data) * .95))
        train_data = data[:training_data_len]
        test_data = data[training_data_len - 60:]

        # Fit and transform training data
        logging.info("Fitting and transforming training data.")
        X_train, y_train = pipeline.named_steps['sequence_gen'].transform(
            pipeline.named_steps['data_prep'].fit_transform(train_data)
        )

        # Transform test data
        logging.info("Transforming test data.")
        X_test, y_test = pipeline.named_steps['sequence_gen'].transform(
            pipeline.named_steps['data_prep'].transform(test_data)
        )
        
        # Train and evaluate the LSTM model
        logging.info("Training and evaluating the LSTM model.")
        mse_lstm, mae_lstm, r2_lstm, mape_lstm, time_lstm = pipeline.named_steps['lstm_model'].fit(X_train, y_train, X_test, y_test)
        
        # Log and display metrics
        logging.info(f"LSTM Model Performance - MSE: {mse_lstm}, MAE: {mae_lstm}, R2: {r2_lstm}, MAPE: {mape_lstm}, Time: {time_lstm}s")
        
        # Predict the prices
        logging.info("Making predictions with the LSTM model.")
        predictions = pipeline.named_steps['lstm_model'].model.predict(X_test)
        predictions = pipeline.named_steps['data_prep'].inverse_transform(predictions)
        
        # Calculate the prediction dates
        logging.info("Calculating prediction dates.")
        prediction_dates = pd.date_range(start=df.index[-len(predictions)], periods=len(predictions), freq='B')

        # Plot the data
        logging.info("Plotting the actual and predicted prices.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual Price'))
        fig.add_trace(go.Scatter(x=prediction_dates, y=predictions.flatten(), mode='lines', name='Predicted Price'))
        fig.update_layout(title=f'{selected_stock} Predicted Prices',
                          xaxis_title='Date',
                          yaxis_title='Price')
        
        # Save plot to a file or generate URL if needed
        plot_url = "URL_TO_THE_SAVED_PLOT"

        # End timing
        end_time = time.time()
        prediction_time = end_time - start_time
        logging.info(f"Prediction completed in {prediction_time:.2f} seconds.")
        
        result = {
            "dates": prediction_dates.tolist(),
            "predictions": predictions.flatten().tolist(),
            "prediction_time": prediction_time,  # Adding prediction time to the result
            "metrics": {
                "MSE": mse_lstm,
                "MAE": mae_lstm,
                "R2": r2_lstm,
                "MAPE": mape_lstm,
                "Training Time": time_lstm
            },
            "plot_url": plot_url  # Return URL or path to the plot
        }
        
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

def preprocess_data(df):
    logging.info("Starting data preprocessing.")
    df['Date'] = pd.to_datetime(df.index)
    df.set_index('Date', inplace=True)
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['Weekday'] = df.index.to_series().dt.weekday  # Haftanın günü
    
    logging.info("Data preprocessing completed.")
    return df

def perform_eda(df):
    logging.info("Starting Exploratory Data Analysis (EDA).")
    descriptive_stats = df.describe().to_dict()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title="Close Price Over Time",
                      xaxis_title='Date',
                      yaxis_title='Close Price')
    
    logging.info("Exploratory Data Analysis (EDA) completed.")
    return {
        "descriptive_stats": descriptive_stats,
        "plot_url": "URL_TO_THE_SAVED_EDA_PLOT"  # Return URL or path to the EDA plot
    }

def feature_engineering(df):
    logging.info("Starting feature engineering.")
    df['Rolling_Mean'] = df['Close'].rolling(window=20).mean()
    df['Rolling_Std'] = df['Close'].rolling(window=20).std()
    logging.info("Feature engineering completed.")
    
    return df

def create_pipeline():
    logging.info("Creating ML pipeline.")
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
        logging.info("Generating sequences for LSTM model.")
        X, y = [], []
        for i in range(len(data) - 60):
            X.append(data[i:i+60])
            y.append(data[i+60])
        logging.info("Sequence generation completed.")
        return np.array(X), np.array(y)

class LSTMModel:
    def __init__(self):
        logging.info("Initializing LSTM model.")
        self.model = None
        
    def fit(self, X_train, y_train, X_test, y_test):
        logging.info("Building and fitting LSTM model.")
        
        # Modelin oluşturulması
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(50),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Eğitim süresinin ölçülmesi
        start_time_lstm = time.time()
        self.model.fit(X_train, y_train, batch_size=1, epochs=1)
        end_time_lstm = time.time()
        time_lstm = end_time_lstm - start_time_lstm
        
        # Tahminlerin yapılması
        logging.info("Making predictions with the LSTM model.")
        predictions = self.model.predict(X_test)
        
        # Performans metriklerinin hesaplanması
        logging.info("Calculating performance metrics.")
        mse_lstm = mean_squared_error(y_test, predictions)
        mae_lstm = mean_absolute_error(y_test, predictions)
        r2_lstm = r2_score(y_test, predictions)
        mape_lstm = mean_absolute_percentage_error(y_test, predictions)
        
        logging.info("LSTM model training and evaluation completed.")
        
        return mse_lstm, mae_lstm, r2_lstm, mape_lstm, time_lstm

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8503, debug=True)