import numpy as np
import pandas as pd
import yfinance as yf
import glob
import json
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for better visuals
sns.set(style='whitegrid')

# --------------------------------------------
# Step 1: Read and Process Sentiment JSON Files
# --------------------------------------------

def load_sentiment_data(json_folder_path):
    sentiment_data = []
    urls_seen = set()
    for file in glob.glob(os.path.join(json_folder_path, "*.json")):
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
                    # Check for duplicates based on URL
                    if data['url'] not in urls_seen:
                        urls_seen.add(data['url'])
                        sentiment_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON object in file {file}: {e}")
                    continue  # Skip malformed JSON objects
    return pd.DataFrame(sentiment_data)

# --------------------------------------------
# Step 2: Prepare Data for Training
# --------------------------------------------

def prepare_training_data(sentiment_df):
    # Correct the datetime conversion
    try:
        # Try converting assuming 'datetime' is in milliseconds
        sentiment_df['date'] = pd.to_datetime(sentiment_df['datetime'], unit='s').dt.date
    except Exception as e:
        print(f"Error converting datetime: {e}")
        return None, None

    # Get list of unique stock tickers
    sentiment_df['stock_ticker'] = sentiment_df['stock_ticker'].str.upper()
    stock_tickers = sentiment_df['stock_ticker'].unique()
    print(f"Found sentiment data for {len(stock_tickers)} stocks.")

    # Visualize the number of news articles per stock
    article_counts = sentiment_df['stock_ticker'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=article_counts.index, y=article_counts.values, palette='viridis')
    plt.title('Number of News Articles per Stock')
    plt.xlabel('Stock Ticker')
    plt.ylabel('Number of Articles')
    plt.show()

    # Prepare a list to hold data from all stocks
    all_data = []
    max_close_prices = {}  # To store max 'Close' price for each stock

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

        # Normalize 'Close' price per stock
        max_close = data_merged['Close'].max()
        data_merged['Close'] = data_merged['Close'] / max_close

        # Store the max 'Close' price
        max_close_prices[ticker] = max_close

        # Add a column for the ticker symbol
        data_merged['ticker'] = ticker

        # Multiply normalized 'Close' by max_close to get actual prices for plotting
        data_merged['Actual_Close'] = data_merged['Close'] * max_close

        # Append to the list
        all_data.append(data_merged)

        # --------------------------------------------
        # Improved Visualization for Stock Price and News Articles
        # --------------------------------------------

        # Prepare data for the bar chart
        article_counts_daily = sentiment_data.groupby('date').size().reindex(data_merged['date'], fill_value=0)
        # Average sentiment per day
        sentiment_scores_daily = sentiment_data.groupby('date')['body_compound'].mean().reindex(data_merged['date'], fill_value=0)

        # Create a new DataFrame for plotting
        plot_data = pd.DataFrame({
            'Date': data_merged['Date'],
            'Close_Price': data_merged['Actual_Close'],  # Use actual prices
            'Article_Count': article_counts_daily.values,
            'Sentiment_Score': sentiment_scores_daily.values
        })

        # Normalize article counts for plotting
        max_articles = plot_data['Article_Count'].max()
        if max_articles == 0:
            max_articles = 1  # Prevent division by zero
        plot_data['Normalized_Article_Count'] = plot_data['Article_Count'] / max_articles * plot_data['Close_Price'].max() * 0.2  # Scale to 20% of max price

        # Determine colors based on sentiment
        def sentiment_to_color(sentiment):
            if sentiment > 0.05:
                return 'green'
            elif sentiment < -0.05:
                return 'red'
            else:
                return 'gray'

        colors = plot_data['Sentiment_Score'].apply(sentiment_to_color)

        # Plotting
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot the actual stock closing price
        ax1.plot(plot_data['Date'], plot_data['Close_Price'], label='Close Price', color='blue')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Plot the article counts as bars with sentiment colors
        ax2 = ax1.twinx()
        ax2.bar(plot_data['Date'], plot_data['Article_Count'], color=colors, alpha=0.3, label='News Articles')
        ax2.set_ylabel('Number of Articles', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

        # Add legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

        plt.title(f'{ticker} Stock Price with News Articles and Sentiment')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Combine all data into a single DataFrame
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data, max_close_prices
    else:
        print("No data available for training.")
        return None, None

# --------------------------------------------
# Step 3: Build the Advanced LSTM Model
# --------------------------------------------

def build_model(input_shape):
    model = Sequential([
        LSTM(units=128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(units=64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(units=32),
        BatchNormalization(),
        Dropout(0.3),
        Dense(units=64, activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=1)
    ])
    return model

# --------------------------------------------
# Step 4: Training Function
# --------------------------------------------

def train_model(combined_data, model_save_path, max_close_prices):
    # Feature engineering and scaling
    features = [
        # Exclude 'Close' from features to be scaled
        'body_neg', 'body_neu', 'body_pos', 'body_compound',
        'title_neg', 'title_neu', 'title_pos', 'title_compound',
        'body_has_name', 'body_has_ticker', 'title_has_name', 'title_has_ticker'
    ]

    # Keep 'Close' as the first column
    data_features = combined_data[['Close'] + features]

    # Scale features (excluding 'Close')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(data_features[features])

    # Combine scaled features with 'Close'
    scaled_data = np.hstack((data_features[['Close']].values, scaled_features))

    # Prepare sequences for LSTM model
    sequence_length = 60  # Number of days to look back
    x_full, y_full = [], []

    for i in range(sequence_length, len(scaled_data)):
        x_full.append(scaled_data[i-sequence_length:i])
        y_full.append(scaled_data[i, 0])  # Predict 'Close' price

    x_full, y_full = np.array(x_full), np.array(y_full)

    if len(x_full) == 0:
        print("Not enough data to create sequences for the LSTM model.")
        return

    # Split data into training and validation sets
    train_size = int(len(x_full) * 0.8)
    x_train = x_full[:train_size]
    y_train = y_full[:train_size]
    x_val = x_full[train_size:]
    y_val = y_full[train_size:]

    # Build and compile the model
    model = build_model((x_full.shape[1], x_full.shape[2]))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_save_path, save_best_only=True)

    # Train the model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=200,
        batch_size=64,
        callbacks=[ model_checkpoint]
    )

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Model Loss by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Save the scaler and max_close_prices
    scaler_path = os.path.join(os.path.dirname(model_save_path), 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        import pickle
        pickle.dump({'scaler': scaler, 'max_close_prices': max_close_prices}, f)

    print("Training complete. Model and scaler saved.")
    return model

# --------------------------------------------
# Step 5: Testing Function
# --------------------------------------------

def test_model(ticker_symbol, json_folder_path, model_save_path, optimize=False):
    # Load the trained model
    if not os.path.exists(model_save_path):
        print("Trained model not found. Please train the model first.")
        return

    model = load_model(model_save_path)

    # Load the scaler and max_close_prices
    scaler_path = os.path.join(os.path.dirname(model_save_path), 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        import pickle
        scaler_data = pickle.load(f)
        scaler = scaler_data['scaler']
        max_close_prices = scaler_data['max_close_prices']

    ticker_symbol = ticker_symbol.upper()
    if ticker_symbol not in max_close_prices:
        print(f"Max close price for ticker {ticker_symbol} not found.")
        return

    max_close = max_close_prices[ticker_symbol]

    # Load sentiment data for the specific stock
    sentiment_df = load_sentiment_data(json_folder_path)
    sentiment_df = sentiment_df[sentiment_df['stock_ticker'].str.upper() == ticker_symbol]
    if sentiment_df.empty:
        print(f"No sentiment data found for ticker {ticker_symbol}.")
        return

    # Correct the datetime conversion
    sentiment_df['date'] = pd.to_datetime(sentiment_df['datetime'], unit='s').dt.date

    # Select relevant columns
    sentiment_cols = [
        'body_neg', 'body_neu', 'body_pos', 'body_compound',
        'title_neg', 'title_neu', 'title_pos', 'title_compound',
        'body_has_name', 'body_has_ticker', 'title_has_name', 'title_has_ticker'
    ]

    # Aggregate sentiment data by date
    sentiment_daily = sentiment_df.groupby('date')[sentiment_cols].mean().reset_index()

    # Fetch stock data starting from the earliest news article date
    start_date = sentiment_df['date'].min().strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

    if stock_data.empty:
        print(f"No stock data found for ticker {ticker_symbol}.")
        return

    # Reset index to access 'Date' column
    stock_data.reset_index(inplace=True)
    stock_data['date'] = stock_data['Date'].dt.date

    # Merge stock data with sentiment data
    data_merged = pd.merge(stock_data, sentiment_daily, on='date', how='left')

    # Fill missing sentiment values with zeros (neutral)
    data_merged[sentiment_cols] = data_merged[sentiment_cols].fillna(0)

    # Normalize 'Close' price using the max_close for this stock
    data_merged['Close'] = data_merged['Close'] / max_close

    # Feature engineering and scaling
    features = [
        # Exclude 'Close' from features to be scaled
        'body_neg', 'body_neu', 'body_pos', 'body_compound',
        'title_neg', 'title_neu', 'title_pos', 'title_compound',
        'body_has_name', 'body_has_ticker', 'title_has_name', 'title_has_ticker'
    ]

    data_features = data_merged[['Close'] + features]

    # Scale features using the saved scaler
    scaled_features = scaler.transform(data_features[features])

    # Combine scaled features with 'Close'
    scaled_data = np.hstack((data_features[['Close']].values, scaled_features))

    # Prepare sequences for LSTM model
    sequence_length = 5  # Number of days to look back
    x_full, y_full = [], []

    for i in range(sequence_length, len(scaled_data)):
        x_full.append(scaled_data[i-sequence_length:i])
        y_full.append(scaled_data[i, 0])  # Predict 'Close' price

    x_full, y_full = np.array(x_full), np.array(y_full)

    if len(x_full) == 0:
        print("Not enough data to create sequences for the LSTM model.")
        return

    # Optionally optimize the model for the specific stock
    if optimize:
        print(f"Optimizing model for {ticker_symbol}...")
        # Split data into training and validation sets
        train_size = int(len(x_full) * 0.8)
        x_train = x_full[:train_size]
        y_train = y_full[:train_size]
        x_val = x_full[train_size:]
        y_val = y_full[train_size:]

        # Retrain the model
        early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(model_save_path, save_best_only=True)

        model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[model_checkpoint]
        )
        print("Model optimization complete.")

    # Evaluate model on test data
    y_pred = model.predict(x_full)

    # Inverse transform predictions and actual values
    # Multiply back by max_close to get actual prices
    y_pred_rescaled = y_pred.flatten() * max_close
    y_actual_rescaled = y_full * max_close

    # Calculate accuracy metrics
    mse = mean_squared_error(y_actual_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual_rescaled, y_pred_rescaled)

    # Direction accuracy
    y_actual_direction = np.sign(np.diff(y_actual_rescaled))
    y_pred_direction = np.sign(np.diff(y_pred_rescaled))
    direction_accuracy = np.mean(y_actual_direction == y_pred_direction)

    print(f"\nModel Evaluation on {ticker_symbol}:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Direction Accuracy: {direction_accuracy*100:.2f}%")

    # Predict the closing price for the day after the last date
    last_sequence = scaled_data[-sequence_length:]
    last_sequence = last_sequence.reshape((1, last_sequence.shape[0], last_sequence.shape[1]))

    predicted_price_scaled = model.predict(last_sequence)

    # Multiply back by max_close to get actual price
    predicted_price = predicted_price_scaled.flatten()[0] * max_close

    next_day = data_merged['Date'].iloc[-1] + timedelta(days=1)
    print(f"\nPredicted closing price for {ticker_symbol} on {next_day.strftime('%Y-%m-%d')}: {predicted_price:.2f}")

    # Prepare date array corresponding to y_actual_rescaled
    y_dates = data_merged['Date'][sequence_length:]

    # Ensure that y_dates and y_actual_rescaled have the same length
    assert len(y_dates) == len(y_actual_rescaled), "Date array and actual prices array must be of the same length."

    # Visualize the results (last 100 days)
    plt.figure(figsize=(12, 6))
    plt.plot(y_dates[-100:], y_actual_rescaled[-100:], label='Actual Close Price', color='blue')
    plt.plot(y_dates[-100:], y_pred_rescaled[-100:], label='Predicted Close Price', color='orange')
    plt.title(f'Predicted Close Prices for {ticker_symbol}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# --------------------------------------------
# Main Function
# --------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train or test the LSTM model.')
    parser.add_argument('--mode', choices=['train', 'test'], required=True, help='Mode: train or test')
    parser.add_argument('--ticker', type=str, help='Ticker symbol for testing')
    parser.add_argument('--json_path', type=str, required=True, help='Path to JSON files folder')
    parser.add_argument('--model_path', type=str, required=True, help='Path to save or load the model')
    parser.add_argument('--optimize', action='store_true', help='Optimize model for specific stock during testing')

    args = parser.parse_args()

    if args.mode == 'train':
        # Load sentiment data
        sentiment_df = load_sentiment_data(args.json_path)
        if sentiment_df.empty:
            print("No sentiment data found.")
            return

        # Prepare training data
        combined_data, max_close_prices = prepare_training_data(sentiment_df)
        if combined_data is not None:
            # Train the model
            train_model(combined_data, args.model_path, max_close_prices)
    elif args.mode == 'test':
        if not args.ticker:
            print("Please provide a ticker symbol for testing using --ticker.")
            return
        # Test the model on the specified ticker
        test_model(args.ticker.upper(), args.json_path, args.model_path, optimize=args.optimize)

if __name__ == "__main__":
    main()
