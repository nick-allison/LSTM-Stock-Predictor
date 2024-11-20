#An explanation of the code is provided below 
#This Python script is used to fetch historical stock data, preprocess it, and set up a Long Short-Term Memory (LSTM) model for predicting future stock prices.
#
#The script begins by fetching historical data for a specific stock, in this case, Microsoft (MSFT), using the yf.download() function from the yfinance library. The data is fetched for a specific date range, from the start of 2020 to the current date.
#
#Next, the script isolates the 'Close' prices from the fetched data, as these are the values that will be used for prediction.
#
#The data is then normalized using the MinMaxScaler from the sklearn.preprocessing library. This scales the 'Close' prices to a range between 0 and 1, which can help improve the performance of the LSTM model later on.
#
#The script then creates a full data set, which consists of all the data.
#
#The full data is then split into x_full and y_full data sets. For each point in the full data, the previous 60 points are used as input features (x_full), and the current point is used as the output label (y_full). This means the model will be trained to predict the current stock price based on the previous 60 prices.
#
#The x_full data is then reshaped into a 3D array, which is the input shape expected by the LSTM model.
#
#Finally, the script starts building the LSTM model using the Sequential model from the keras library. The model consists of an LSTM layer with 50 units, followed by a Dropout layer for regularization (to prevent overfitting), and another LSTM layer with 50 units. The return_sequences parameter is set to True for the first LSTM layer because the next layer is also an LSTM layer. For the final LSTM layer, return_sequences is set to False because there are no more LSTM layers following it.
#
#The model is then compiled with the Adam optimizer and the mean squared error loss function, and it's trained on the full data set.
#
#After training, the model is used to predict the future stock prices for a specified number of days. The last 60 days of the scaled data are used as the initial input sequence for the predictions. For each future day, the model predicts the next price, which is then appended to the input sequence for the next prediction.
#
#The predicted prices are then inverse transformed to get the actual price predictions, and future dates are generated for plotting.
#
#Finally, the script plots the historical and predicted prices on a line chart, with the date on the x-axis and the price on the y-axis. The historical prices are shown in blue, and the predicted prices are shown in orange.
######################################################################################################################################

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
from datetime import datetime

# Fetching data
ticker_symbol = input("Enter the ticker symbol(ex:AAPL): ")
data = yf.download(ticker_symbol, start='2020-01-01', end=datetime.today())  # Adjust the end date as needed

# Use only the 'Close' price for prediction
close_prices = data[['Close']]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Create a full dataset for training
x_full, y_full = [], []

#
for i in range(60, len(scaled_data)):
    x_full.append(scaled_data[i-60:i, 0])
    y_full.append(scaled_data[i, 0])
    
x_full, y_full = np.array(x_full), np.array(y_full)

# Reshape the data into the shape accepted by the LSTM
x_full = np.reshape(x_full, (x_full.shape[0], x_full.shape[1], 1))

# Build the LSTM network model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(x_full.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_full, y_full, batch_size=1, epochs=1)

# Number of days to predict into the future
future_days = 30

# Prepare the last 60 days as the initial input sequence
input_seq = scaled_data[-60:].reshape(1, -1)
input_seq = np.reshape(input_seq, (1, 60, 1))

predicted_prices = []

# Predict future prices
for _ in range(future_days):
    predicted_price = model.predict(input_seq)
    predicted_prices.append(predicted_price[0, 0])
    predicted_price = np.reshape(predicted_price, (1, 1, 1))  # Reshape predicted_price
    input_seq = np.append(input_seq[:, 1:, :], predicted_price, axis=1)  # Append reshaped predicted_price

# Inverse transform to get actual price predictions
predicted_prices = np.array(predicted_prices).reshape(-1, 1)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Generate future dates for plotting
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date, periods=future_days)  # Adjusted to match the number of predicted prices

# Visualize the predictions
plt.figure(figsize=(10, 6))
plt.plot(data.index, close_prices, label='Historical Daily Close Price')
plt.plot(future_dates, predicted_prices, label='Predicted Daily Close Price')
plt.title(f'Future Price Prediction for {ticker_symbol}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.show()
