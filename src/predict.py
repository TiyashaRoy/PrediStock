# PrediStock - Stock Price Prediction
# Generate next-step stock price prediction

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

# 1. Load the cleaned stock data

data = pd.read_csv("data/cleaned_stock_data.csv")

prices = data["Close"].values.reshape(-1, 1)

# 2. Load the trained LSTM model

model = load_model("models/stock_lstm.keras")

# 3. Load the saved scaler

scaler = joblib.load("models/scaler.pkl")

# 4. Select the last 60 trading days

if len(prices) < 60:
    raise ValueError("At least 60 prices are required.")

last_60_days = prices[-60:]

# 5. Normalize the data

scaled_data = scaler.transform(last_60_days)

# 6. Reshape data for LSTM

X = scaled_data.reshape(1, 60, 1)

# 7. Predict the next closing price

prediction = model.predict(X, verbose=0)

predicted_price = scaler.inverse_transform(
    prediction
)[0][0]

last_price = prices[-1][0]

# 8. Display results

print("\nPREDISTOCK PRICE PREDICTION")
print("---------------------------")

print("Last closing price: $", round(last_price, 2))

print("Predicted next-step price: $",
      round(predicted_price, 2))

# 9. Save prediction

os.makedirs("outputs", exist_ok=True)

result = pd.DataFrame({
    "Last_Closing_Price": [last_price],
    "Predicted_Next_Step_Price": [predicted_price]
})

result.to_csv(
    "outputs/next_prediction.csv",
    index=False
)

# 10. Visualize prediction

plt.figure(figsize=(10, 5))

plt.plot(
    range(60),
    last_60_days.flatten(),
    label="Historical Closing Prices"
)

plt.scatter(
    [60],
    [predicted_price],
    color="red",
    label="Predicted Next Price"
)

plt.title("PrediStock - Stock Price Prediction")
plt.xlabel("Trading Day")
plt.ylabel("Stock Price (USD)")
plt.legend()
plt.grid(True)

plt.savefig("outputs/prediction_chart.png")
plt.show()

print("\nPrediction completed successfully!")
print("Results saved in outputs folder.")
