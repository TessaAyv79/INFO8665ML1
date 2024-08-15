import logging

logging.basicConfig(level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict():
    logging.info("Starting API call to /predict")
    model_id = request.json['model_id']
    input_data = request.json['input_data']
    # Tahmin işlemi burada yapılır
    logging.info("API call to /predict completed")
    return jsonify({"status": "success", "predictions": [{"date": "2024-08-10", "predicted_price": 152.00}]})