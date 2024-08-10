import numpy as np  # NumPy'yi içe aktar

# Örnek veri seti oluşturma
X_train = np.random.rand(100, 60, 1)  # 100 örnek, 60 zaman adımı, 1 özellik
y_train = np.random.rand(100, 1)      # 100 etiket

# Modeli oluştur ve eğit
model = LSTMModel()
model.fit(X_train, y_train)

# Modelin eğitildiğinden emin olun
print("Model training completed.")from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

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
        self.model.fit(X, y, epochs=10, batch_size=32, verbose=1)

    def predict(self, X):
        return self.model.predict(X)

    def inverse_transform(self, predictions):
        return predictions  # Placeholder

# Örnek veri seti oluşturma
X_train = np.random.rand(100, 60, 1)  # 100 örnek, 60 zaman adımı, 1 özellik
y_train = np.random.rand(100, 1)      # 100 etiket

# Modeli oluştur ve eğit
model = LSTMModel()
model.fit(X_train, y_train)

# Modelin eğitildiğinden emin olun
print("Model training completed.")