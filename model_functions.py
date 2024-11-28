import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
import glob
import os
import random
import pickle

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

def load_sentiment_data(json_folder_path):
    import json

    sentiment_data = []
    urls_seen = set()
    for file in glob.glob(os.path.join(json_folder_path, "*.json")):
        with open(file, 'r', encoding='utf-8') as f:
            file_content = f.read().strip()
            try:
                # Attempt to load the entire content as JSON
                data_loaded = json.loads(file_content)

                if isinstance(data_loaded, list):
                    # The file contains a JSON array
                    for data in data_loaded:
                        if 'url' in data and data['url'] not in urls_seen:
                            urls_seen.add(data['url'])
                            sentiment_data.append(data)
                elif isinstance(data_loaded, dict):
                    # The file contains a single JSON object
                    data = data_loaded
                    if 'url' in data and data['url'] not in urls_seen:
                        urls_seen.add(data['url'])
                        sentiment_data.append(data)
                else:
                    print(f"Unsupported JSON format in file {file}.")
            except json.JSONDecodeError:
                # Handle concatenated JSON objects
                try:
                    json_objects = []
                    decoder = json.JSONDecoder()
                    idx = 0
                    while idx < len(file_content):
                        obj, idx = decoder.raw_decode(file_content, idx)
                        json_objects.append(obj)
                        # Skip any whitespace between JSON objects
                        while idx < len(file_content) and file_content[idx].isspace():
                            idx += 1
                    for data in json_objects:
                        if 'url' in data and data['url'] not in urls_seen:
                            urls_seen.add(data['url'])
                            sentiment_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file}: {e}")
                    continue  # Skip malformed JSON files
    return pd.DataFrame(sentiment_data)

def prepare_training_data(sentiment_df, visualize=True):
    try:
        sentiment_df['date'] = pd.to_datetime(sentiment_df['datetime'], unit='s').dt.date
    except Exception as e:
        print(f"Error converting datetime: {e}")
        return None, None

    # Get list of unique stock tickers
    sentiment_df['stock_ticker'] = sentiment_df['stock_ticker'].str.upper()
    stock_tickers = sentiment_df['stock_ticker'].unique()
    print(f"Found sentiment data for {len(stock_tickers)} stocks.")

    # Visualize the number of news articles per stock
    if visualize:
        article_counts = sentiment_df['stock_ticker'].value_counts()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=article_counts.index, y=article_counts.values, palette='viridis')
        plt.title('Number of News Articles per Stock')
        plt.xlabel('Stock Ticker')
        plt.ylabel('Number of Articles')
        plt.tight_layout()
        plt.savefig('static/article_counts.png')
        plt.close()

    # list to hold data from all stocks
    all_data = []

    # Loop through each stock ticker and prepare data
    for ticker in stock_tickers:
        print(f"Processing data for ticker: {ticker}")
        # Filter sentiment data for the ticker
        sentiment_data = sentiment_df[sentiment_df['stock_ticker'] == ticker]

        # Aggregate sentiment data by date
        sentiment_cols = [
            'body_neg', 'body_neu', 'body_pos', 'body_compound',
            'title_neg', 'title_neu', 'title_pos', 'title_compound',
            'body_has_name', 'body_has_ticker', 'title_has_name', 'title_has_ticker'
        ]
        sentiment_daily = sentiment_data.groupby('date')[sentiment_cols].mean().reset_index()

        # Fetch stock data starting from the earliest news article date
        start_date = sentiment_data['date'].min().strftime('%Y-%m-%d')
        end_date = datetime.today().strftime('%Y-%m-%d')
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        if stock_data.empty:
            print(f"No stock data found for ticker {ticker}. Skipping.")
            continue

        # Reset index to access 'Date' column
        stock_data.reset_index(inplace=True)
        stock_data['date'] = stock_data['Date'].dt.date

        # Merge stock data with sentiment data
        data_merged = pd.merge(stock_data, sentiment_daily, on='date', how='left')

        # Fill missing sentiment values with zeros (neutral)
        data_merged[sentiment_cols] = data_merged[sentiment_cols].fillna(0)

        # Normalize 'Close' price per stock using a rolling window
        data_merged['Close_Normalized'] = data_merged['Close'] / data_merged['Close'].rolling(window=60, min_periods=1).max()
        data_merged['Close_Normalized'].fillna(method='bfill', inplace=True)

        # Add a column for the ticker symbol
        data_merged['ticker'] = ticker

        # Append to the list
        all_data.append(data_merged)

    # Combine all data into a single DataFrame
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data, None
    else:
        print("No data available for training.")
        return None, None

def build_model(input_shape):
    model = Sequential([
        LSTM(units=128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(units=64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(units=32),
        BatchNormalization(),
        Dropout(0.2),
        Dense(units=64, activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=1)
    ])
    return model

def train_model(json_folder_path, model_save_path, sequence_length=60, batch_size=64, epochs=200):
    # Load sentiment data
    sentiment_df = load_sentiment_data(json_folder_path)
    if sentiment_df.empty:
        print("No sentiment data loaded.")
        return None

    # Prepare training data
    data, _ = prepare_training_data(sentiment_df, visualize=True)
    if data is None:
        print("No data available for training.")
        return None

    # Prepare features and target
    features = [
        'Open', 'High', 'Low', 'Volume', 'body_neg', 'body_neu', 'body_pos', 'body_compound',
        'title_neg', 'title_neu', 'title_pos', 'title_compound',
        'body_has_name', 'body_has_ticker', 'title_has_name', 'title_has_ticker'
    ]

    # Scaling features
    feature_scaler = StandardScaler()

    data_features = data.copy()
    data_features[features] = feature_scaler.fit_transform(data_features[features])

    # Target variable
    target = 'Close_Normalized'

    # Prepare sequences
    x_full = []
    y_full = []
    for i in range(sequence_length, len(data_features)):
        x_full.append(data_features[features].iloc[i-sequence_length:i].values)
        y_full.append(data_features[target].iloc[i])

    x_full = np.array(x_full)
    y_full = np.array(y_full)

    # Split into training and validation sets
    train_size = int(len(x_full) * 0.8)
    x_train = x_full[:train_size]
    y_train = y_full[:train_size]
    x_val = x_full[train_size:]
    y_val = y_full[train_size:]

    # Build and compile model
    model = build_model((sequence_length, len(features)))
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # Train model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, model_checkpoint, reduce_lr]
    )

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Model Loss by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/model_loss.png')
    plt.close()

    # Save scaler
    scaler_path = model_save_path.replace('.keras', '_scalers.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump({'feature_scaler': feature_scaler}, f)

    return model

def test_model(ticker_symbol, json_folder_path, model_save_path, image_path_normalized=None, image_path_actual=None, sequence_length=60):
    # Load sentiment data
    sentiment_df = load_sentiment_data(json_folder_path)
    if sentiment_df.empty:
        print("No sentiment data loaded.")
        return None

    # Prepare testing data
    data, _ = prepare_training_data(sentiment_df, visualize=False)
    if data is None:
        print("No data available for testing.")
        return None

    # Filter data for the specific ticker
    data = data[data['ticker'] == ticker_symbol]
    if data.empty:
        print(f"No data available for ticker {ticker_symbol}.")
        return None

    # Prepare features and target
    features = [
        'Open', 'High', 'Low', 'Volume', 'body_neg', 'body_neu', 'body_pos', 'body_compound',
        'title_neg', 'title_neu', 'title_pos', 'title_compound',
        'body_has_name', 'body_has_ticker', 'title_has_name', 'title_has_ticker'
    ]

    # Load scaler
    scaler_path = model_save_path.replace('.keras', '_scalers.pkl')
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    feature_scaler = scalers['feature_scaler']

    data_features = data.copy()
    data_features[features] = feature_scaler.transform(data_features[features])

    # Target variable
    target = 'Close_Normalized'

    # Prepare sequences
    x_full = []
    y_full = []
    for i in range(sequence_length, len(data_features)):
        x_full.append(data_features[features].iloc[i-sequence_length:i].values)
        y_full.append(data_features[target].iloc[i])

    x_full = np.array(x_full)
    y_full = np.array(y_full)

    if len(x_full) == 0:
        print("Not enough data to create sequences for testing.")
        return {'Error': 'Not enough data to create sequences for testing.'}

    # Load model
    model = build_model((sequence_length, len(features)))
    model.load_weights(model_save_path)

    # Predict
    y_pred = model.predict(x_full)

    #compare directly bc prices are normalized
    predicted_prices_normalized = y_pred.flatten()
    actual_prices_normalized = y_full.flatten()
    dates = data['Date'].iloc[sequence_length:].reset_index(drop=True)

    # Plot predictions vs actual normalized prices
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual_prices_normalized, label='Actual Normalized Close Price', color='blue')
    plt.plot(dates, predicted_prices_normalized, label='Predicted Normalized Close Price', color='orange')
    plt.title(f'{ticker_symbol} Predicted vs Actual Normalized Close Prices')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.tight_layout()
    if image_path_normalized:
        plt.savefig(image_path_normalized)
    else:
        plt.show()
    plt.close()

    # Compute evaluation metrics for normalized prices
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(actual_prices_normalized, predicted_prices_normalized)
    mae = mean_absolute_error(actual_prices_normalized, predicted_prices_normalized)
    r2 = r2_score(actual_prices_normalized, predicted_prices_normalized)

    # Prepare results
    results = {
        'MSE': mse,
        'MAE': mae,
        'R2_Score': r2
    }

    # Plot predictions vs actual actual prices
    actual_prices = data['Close'].iloc[sequence_length:].reset_index(drop=True).values
    # Reconstruct predicted actual prices
    # Denormalize predicted prices using rolling maximum
    rolling_max = data['Close'].rolling(window=60, min_periods=1).max().iloc[sequence_length:].reset_index(drop=True).values
    predicted_prices = predicted_prices_normalized * rolling_max

    # Check lengths
    min_len = min(len(dates), len(actual_prices), len(predicted_prices))
    dates = dates[:min_len]
    actual_prices = actual_prices[:min_len]
    predicted_prices = predicted_prices[:min_len]

    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual_prices, label='Actual Close Price', color='blue')
    plt.plot(dates, predicted_prices, label='Predicted Close Price', color='orange')
    plt.title(f'{ticker_symbol} Predicted vs Actual Close Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    if image_path_actual:
        plt.savefig(image_path_actual)
    else:
        plt.show()
    plt.close()

    # Compute directional accuracy
    actual_direction = np.sign(np.diff(actual_prices))
    predicted_direction = np.sign(np.diff(predicted_prices))
    directional_accuracy = np.mean(actual_direction == predicted_direction) * 100

    # Predicted closing price for tomorrow
    last_sequence = data_features[features].iloc[-sequence_length:].values.reshape(1, sequence_length, -1)
    predicted_normalized_price = model.predict(last_sequence)[0][0]
    last_rolling_max = data['Close'].rolling(window=60, min_periods=1).max().iloc[-1]
    predicted_price_tomorrow = predicted_normalized_price * last_rolling_max

    # Add new metrics to results
    results['Directional_Accuracy'] = directional_accuracy
    results['Predicted_Close_Price_Tomorrow'] = predicted_price_tomorrow

    return results