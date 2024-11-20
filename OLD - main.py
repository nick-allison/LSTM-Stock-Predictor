import numpy as np
import pandas as pd
import yfinance as yf
import glob
import json
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt

# --------------------------------------------
# Step 1: Read and Process Sentiment JSON Files
# --------------------------------------------

def load_sentiment_data(json_folder_path, ticker_symbol):
    sentiment_data = []
    urls_seen = set()
    for file in glob.glob(json_folder_path + "/*.json"):
        with open(file, 'r') as f:
            # Read the entire file content
            file_content = f.read()
            # Split the file content by closing braces to parse multiple JSON objects
            json_objects = file_content.split('}{')
            # Re-add the braces and parse each JSON object
            for i, obj in enumerate(json_objects):
                try:
                    if i == 0:
                        obj = obj + '}'
                    elif i == len(json_objects) - 1:
                        obj = '{' + obj
                    else:
                        obj = '{' + obj + '}'
                    data = json.loads(obj)
                    # Check if the stock_ticker matches
                    if data['stock_ticker'].upper() == ticker_symbol.upper():
                        # Check for duplicates based on URL
                        if data['url'] not in urls_seen:
                            urls_seen.add(data['url'])
                            sentiment_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON object in file {file}: {e}")
                    continue  # Skip malformed JSON objects
    return pd.DataFrame(sentiment_data)

# --------------------------------------------
# Main Script
# --------------------------------------------

def main():
    # Input parameters
    ticker_symbol = input("Enter the ticker symbol (e.g., AAPL): ").upper()
    json_folder_path = "json_data"  # Replace with your JSON files folder path

    # Load and process sentiment data
    sentiment_df = load_sentiment_data(json_folder_path, ticker_symbol)
    if sentiment_df.empty:
        print(f"No sentiment data found for ticker {ticker_symbol}.")
    else:
        # Convert 'datetime' from timestamp to datetime object
        sentiment_df['date'] = pd.to_datetime(sentiment_df['datetime'], unit='s').dt.date

        # Select relevant columns
        sentiment_cols = [
            'body_neg', 'body_neu', 'body_pos', 'body_compound',
            'title_neg', 'title_neu', 'title_pos', 'title_compound',
            'body_has_name', 'body_has_ticker', 'title_has_name', 'title_has_ticker'
        ]

        # Aggregate sentiment data by date
        sentiment_daily = sentiment_df.groupby('date')[sentiment_cols].mean().reset_index()

    # Fetch stock data
    start_date = '2020-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

    if stock_data.empty:
        print(f"No stock data found for ticker {ticker_symbol}.")
        return

    # Reset index to access 'Date' column
    stock_data.reset_index(inplace=True)
    stock_data['date'] = stock_data['Date'].dt.date

    # Merge stock data with sentiment data
    if not sentiment_df.empty:
        data_merged = pd.merge(stock_data, sentiment_daily, on='date', how='left')
    else:
        # If no sentiment data, fill sentiment columns with zeros
        data_merged = stock_data.copy()
        sentiment_cols = [
            'body_neg', 'body_neu', 'body_pos', 'body_compound',
            'title_neg', 'title_neu', 'title_pos', 'title_compound',
            'body_has_name', 'body_has_ticker', 'title_has_name', 'title_has_ticker'
        ]
        for col in sentiment_cols:
            data_merged[col] = 0

    # Fill missing sentiment values with zeros (neutral)
    data_merged[sentiment_cols] = data_merged[sentiment_cols].fillna(0)

    # Feature engineering and scaling
    features = [
        'Close', 'body_neg', 'body_neu', 'body_pos', 'body_compound',
        'title_neg', 'title_neu', 'title_pos', 'title_compound',
        'body_has_name', 'body_has_ticker', 'title_has_name', 'title_has_ticker'
    ]

    data_features = data_merged[features]

    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_features)

    # Prepare sequences for LSTM model
    sequence_length = 60  # Number of days to look back
    x_full, y_full = [], []

    for i in range(sequence_length, len(scaled_data)):
        x_full.append(scaled_data[i-sequence_length:i])
        y_full.append(scaled_data[i, 0])  # Predict 'Close' price

    x_full, y_full = np.array(x_full), np.array(y_full)

    # Split data into training, validation, and test sets
    train_size = int(len(x_full) * 0.7)
    val_size = int(len(x_full) * 0.15)
    test_size = len(x_full) - train_size - val_size

    x_train = x_full[:train_size]
    y_train = y_full[:train_size]

    x_val = x_full[train_size:train_size+val_size]
    y_val = y_full[train_size:train_size+val_size]

    x_test = x_full[train_size+val_size:]
    y_test = y_full[train_size+val_size:]

    # Build and train the LSTM model
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(x_full.shape[1], x_full.shape[2])),
        Dropout(0.2),
        LSTM(units=100),
        Dropout(0.2),
        Dense(units=50, activation='relu'),
        Dense(units=1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=50,
        batch_size=32
    )

    # Evaluate model on test data
    y_pred_test = model.predict(x_test)

    # Inverse transform predictions and actual values
    y_pred_test_full = np.concatenate([y_pred_test, np.zeros((y_pred_test.shape[0], scaled_data.shape[1]-1))], axis=1)
    y_pred_test_rescaled = scaler.inverse_transform(y_pred_test_full)[:, 0]

    y_test_full = np.concatenate([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_data.shape[1]-1))], axis=1)
    y_test_rescaled = scaler.inverse_transform(y_test_full)[:, 0]

    # Calculate accuracy metrics
    mse = mean_squared_error(y_test_rescaled, y_pred_test_rescaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_rescaled, y_pred_test_rescaled)

    # Direction accuracy
    y_test_direction = np.sign(np.diff(y_test_rescaled))
    y_pred_direction = np.sign(np.diff(y_pred_test_rescaled))
    direction_accuracy = np.mean(y_test_direction == y_pred_direction)

    print(f"\nModel Evaluation on Test Data:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Direction Accuracy: {direction_accuracy*100:.2f}%")

    # Predict the closing price for the day after today
    last_sequence = scaled_data[-sequence_length:]
    last_sequence = last_sequence.reshape((1, last_sequence.shape[0], last_sequence.shape[1]))

    predicted_price_scaled = model.predict(last_sequence)

    predicted_price_full = np.concatenate([predicted_price_scaled, np.zeros((1, scaled_data.shape[1]-1))], axis=1)
    predicted_price = scaler.inverse_transform(predicted_price_full)[:, 0]

    next_day = datetime.today() + timedelta(days=1)
    print(f"\nPredicted closing price for {ticker_symbol} on {next_day.strftime('%Y-%m-%d')}: {predicted_price[0]:.2f}")

    # Visualize the results (last 100 days)
    plt.figure(figsize=(12, 6))
    plt.plot(data_merged['Date'][-100:], data_merged['Close'][-100:], label='Historical Close Price')
    plt.plot([data_merged['Date'].iloc[-1] + timedelta(days=1)], [predicted_price[0]], 'ro', label='Predicted Close Price')
    plt.title(f'Predicted Close Price for {ticker_symbol} on {next_day.strftime("%Y-%m-%d")}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    main()
