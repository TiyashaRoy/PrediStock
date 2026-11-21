# PrediStock - LSTM Stock Price Prediction
# Module 2: Model Training

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load cleaned stock data

data = pd.read_csv("data/cleaned_stock_data.csv")

prices = data["Close"].values.reshape(-1, 1)

print("Stock data loaded successfully!")
print("Total records:", len(prices))

# 2. Normalize stock prices

split_index = int(len(prices) * 0.8)

scaler = MinMaxScaler(feature_range=(0, 1))

scaler.fit(prices[:split_index])

scaled_data = scaler.transform(prices)

# 3. Prepare training and testing data

window_size = 60

def create_sequences(dataset, start, end, window):
    X = []
    y = []

    for i in range(start, end):
        X.append(dataset[i-window:i])
        y.append(dataset[i])

    return np.array(X), np.array(y)

X_train, y_train = create_sequences(
    scaled_data,
    window_size,
    split_index,
    window_size
)

X_test, y_test = create_sequences(
    scaled_data,
    split_index,
    len(scaled_data),
    window_size
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# 4. Build LSTM neural network

model = Sequential([
    LSTM(50, return_sequences=True,
         input_shape=(window_size, 1)),

    Dropout(0.2),

    LSTM(50),

    Dropout(0.2),

    Dense(25),
    Dense(1)
])

# 5. Compile the model

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

print("LSTM model created successfully!")

# 6. Train the model

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# 7. Predict stock prices

predictions = model.predict(X_test)

predicted_prices = scaler.inverse_transform(predictions)

actual_prices = scaler.inverse_transform(y_test)

# 8. Evaluate the model

rmse = np.sqrt(
    mean_squared_error(actual_prices, predicted_prices)
)

mae = mean_absolute_error(
    actual_prices, predicted_prices
)

print("\nMODEL EVALUATION")
print("RMSE:", round(rmse, 2))
print("MAE:", round(mae, 2))

# 9. Save trained model and scaler

os.makedirs("models", exist_ok=True)

model.save("models/stock_lstm.keras")

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("\nModel training completed successfully!")
print("Model saved in models/stock_lstm.keras")
print("Scaler saved in models/scaler.pkl")
