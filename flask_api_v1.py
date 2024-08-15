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
# Configure logging
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log the entire request data
        logging.info("Request data: %s", request.get_data(as_text=True))
        
        # Ensure the request content type is application/json
        if request.content_type != 'application/json':
            return jsonify({"status": "error", "message": "Content-Type must be application/json"}), 415

        # Get JSON data from the request
        data = request.get_json()
        selected_stock = data['selected_stock']
        start_date = data['start_date']
        end_date = data['end_date']
        
        # Download data
        df = yf.download(selected_stock, start=start_date, end=end_date)
        
        if df.empty:
            return jsonify({"status": "error", "message": "No data found"}), 404

        # Data preprocessing
        df = preprocess_data(df)
        df = feature_engineering(df)
        
        # Prepare data
        data = df[['Close']]
        dataset = data
        training_data_len = int(np.ceil(len(dataset) * .95))

        # Create and fit the pipeline (do this only once after your model is ready)
        pipeline = create_pipeline()
        pipeline.named_steps['data_prep'].fit(dataset)
        
        # Create training and test datasets
        X_train, y_train = pipeline.named_steps['sequence_gen'].transform(
            pipeline.named_steps['data_prep'].transform(dataset)
        )
        
        # Fit the model (do this only once)
        pipeline.named_steps['lstm_model'].fit(X_train, y_train)
        
        # Create test dataset
        test_data = pipeline.named_steps['data_prep'].transform(dataset)[training_data_len - 60:, :]
        X_test = pipeline.named_steps['sequence_gen'].transform(test_data)[0]
        
        # Get model predictions
        predictions = pipeline.named_steps['lstm_model'].predict(X_test)
        
        # Calculate prediction dates
        prediction_dates = pd.date_range(end=end_date, periods=len(predictions) + 1, freq='B')[1:]
        
        result = {
            "dates": prediction_dates.tolist(),
            "predictions": predictions.flatten().tolist()
        }
        
        # Log the result
        logging.info("Prediction result: %s", result)
        
        return jsonify(result)
    
    except Exception as e:
        logging.error("Exception occurred: %s", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/train', methods=['GET'])
def train():
    return jsonify({"status": "success", "message": "Model training endpoint not implemented."})

@app.route('/preprocess', methods=['GET'])
def preprocess():
    return jsonify({"status": "success", "message": "Data preprocessing endpoint not implemented."})

@app.route('/data', methods=['GET'])
def get_data():
    # Example data
    data = {"symbol": "AAPL", "price": 150.75, "volume": 10000}
    return jsonify({"status": "success", "data": data})

def preprocess_data(df):
    logging.info("Veri ön işleme başlıyor.")
    df['Date'] = pd.to_datetime(df.index)
    df.set_index('Date', inplace=True)
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['Weekday'] = df.index.to_series().dt.weekday
    
    logging.info("Veri ön işleme tamamlandı.")
    return df

def feature_engineering(df):
    df['Rolling_Mean'] = df['Close'].rolling(window=20).mean()
    df['Rolling_Std'] = df['Close'].rolling(window=20).std()
    return df

def create_pipeline():
    logging.info("ML pipeline oluşturuluyor.")
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
    app.run(host="127.0.0.1", port=5000, debug=True)