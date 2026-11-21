import yfinance as yf
import os

# Select the stock
stock_symbol = "AAPL"

# Download historical stock data
stock = yf.Ticker(stock_symbol)

data = stock.history(period="5y")

# Display the data
print(data.head())

# Create data folder
os.makedirs("data", exist_ok=True)

# Save the data
data.to_csv("data/stock_data.csv")

print("Stock data collected successfully!")
