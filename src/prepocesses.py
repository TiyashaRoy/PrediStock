import pandas as pd

# Read stock data
data = pd.read_csv("data/stock_data.csv")

# Remove missing values
data = data.dropna()

# Select important columns
data = data[
    ["Date", "Open", "High", "Low", "Close", "Volume"]
]

# Save cleaned data
data.to_csv(
    "data/cleaned_stock_data.csv",
    index=False
)

print("Data preprocessing completed!")
print(data.head())
