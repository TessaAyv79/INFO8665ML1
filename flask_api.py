from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import LSTM, Dense
import logging
from flask_cors import CORS

# Create Flask app instance
app = Flask(__name__)
CORS(app)  # Enable CORS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/test', methods=['GET', 'POST'])
def test():
    return jsonify({"status": "success", "message": "Test endpoint reached."})

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     data = request.json
#     # Tahmin mantığınızı buraya ekleyin
#     return jsonify({"message": "İstek alındı", "data": data})

@app.route('/predict', methods=['POST'])
def predict():
    logging.info("Predict endpoint was called.")
    try:
        if request.content_type != 'application/json':
            logging.error(f"Expected application/json but got {request.content_type}")
            return jsonify({"status": "error", "message": "Unsupported Media Type"}), 415

        # Get JSON data from request
        data = request.get_json()
        logging.debug(f"Received data: {data}")

        if not data:
            return jsonify({"status": "error", "message": "No JSON data received"}) 

        selected_stock = data.get('selected_stock')
        start_date = pd.to_datetime(data.get('start_date'))
        end_date = pd.to_datetime(data.get('end_date'))
        
        logging.debug(f"Selected stock: {selected_stock}, Start date: {start_date}, End date: {end_date}")

        if not selected_stock or not start_date or not end_date:
            return jsonify({"status": "error", "message": "Required fields are missing"})

        # Download data
        df = yf.download(selected_stock, start=start_date, end=end_date)
        logging.debug(f"Downloaded data: {df.head()}")

        if df.empty:
            logging.warning("No data found for the selected stock.")
            return jsonify({"status": "error", "message": "No data found"})

        # Data preprocessing and feature engineering
        df = preprocess_data(df)
        df = feature_engineering(df)
        
        # Logging intermediate data
        logging.debug(f"Preprocessed data: {df.head()}")

        # Create and fit the pipeline
        pipeline = create_pipeline()
        data = df[['Close']]
        dataset = data
        training_data_len = int(np.ceil(len(dataset) * .95))

        pipeline.named_steps['data_prep'].fit(dataset)

        X_train, y_train = pipeline.named_steps['sequence_gen'].transform(
            pipeline.named_steps['data_prep'].transform(dataset)
        )
        logging.debug(f"Training data X shape: {X_train.shape}, y shape: {y_train.shape}")

        pipeline.named_steps['lstm_model'].fit(X_train, y_train)
        logging.info("Model trained successfully.")
        return jsonify({"status": "success", "message": "Model trained successfully"})

    except ValueError as ve:
        logging.error(f"Value error: {ve}")
        return jsonify({"status": "error", "message": str(ve)})

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({"status": "error", "message": str(e)})

# Define your other endpoints here
@app.route('/train', methods=['GET', 'POST'])
def train():
    return jsonify({"status": "success", "message": "Model training endpoint not implemented."})

@app.route('/preprocess/preprocess', methods=['GET', 'POST'])
def preprocess():
    return jsonify({"status": "success", "message": "Data preprocessing endpoint not implemented."})

@app.route('/data', methods=['GET'])
def get_data():
    data = {"symbol": "AAPL", "price": 150.75, "volume": 10000}
    return jsonify({"status": "success", "data": data})

def preprocess_data(df):
    logging.info("Starting data preprocessing.")
    df['Date'] = pd.to_datetime(df.index)
    df.set_index('Date', inplace=True)
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['Weekday'] = df.index.to_series().dt.weekday
    
    logging.info("Data preprocessing completed.")
    return df

def feature_engineering(df):
    df['Rolling_Mean'] = df['Close'].rolling(window=20).mean()
    df['Rolling_Std'] = df['Close'].rolling(window=20).std()
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

if __name__ == "__main__":
    logging.info("Starting Flask application!")
    app.run(host="127.0.0.0", port=5000, debug=True)