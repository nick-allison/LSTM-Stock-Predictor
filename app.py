import streamlit as st
import json
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from future import train_and_predict_lstm
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from Get_News_Data import news_api, cleaner
from Get_Stock_Data import get_stock_data
from datetime import datetime

# Title for the Application
st.title("LSTM Stock Predictor & Sentiment Analysis")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "News Analysis", "Stock Data Retrieval", "Stock Prediction"])

# Home Page
if page == "Home":
    st.write("""
    ## Welcome to the LSTM Stock Predictor
    Use this app to retrieve financial news, analyze stock prices, and predict future prices using LSTM.
    """)

# News Analysis Page
elif page == "News Analysis":
    st.header("News Sentiment Analysis")
    stock_name = st.text_input("Enter Stock Name (e.g., Tesla)")
    stock_ticker = st.text_input("Enter Stock Ticker (e.g., TSLA)")
    from_date = st.date_input("From Date", datetime(2024, 10, 1))
    to_date = st.date_input("To Date", datetime(2024, 11, 1))

    if st.button("Fetch News and Analyze"):
        if stock_name and stock_ticker:
            try:
                # Fetch and analyze news articles
                news_api.get_all_articles(stock_name, stock_ticker, str(from_date), str(to_date))
                # Read and display articles
                with open(f"{stock_name}.json", 'r') as file:
                    articles = json.load(file)
                    st.write("Fetched Articles:")
                    if articles:
                        for article in articles:
                            st.write(f"Title: {article['other']['title']}")
                            st.write(f"URL: {article['url']}")
                            st.write(f"Sentiment (Title Compound): {article['title_compound']}")
                            st.write("---")
                    else:
                        st.write("No articles were fetched.")
                st.success(f"News articles for {stock_name} fetched and analyzed successfully!")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter both stock name and ticker.")


# Stock Data Retrieval Page
elif page == "Stock Data Retrieval":
    st.header("Stock Data Retrieval")
    ticker_symbol = st.text_input("Enter the Stock Ticker (e.g., AAPL)")
    start_date = st.date_input("Start Date", datetime(2020, 1, 1))
    end_date = st.date_input("End Date", datetime.today())

    if st.button("Fetch Stock Data"):
        if ticker_symbol:
            data = yf.download(ticker_symbol, start=str(start_date), end=str(end_date))
            st.write(data)
            st.line_chart(data['Close'])
        else:
            st.warning("Please enter a valid stock ticker symbol.")

# Stock Prediction Page
elif page == "Stock Prediction":
    st.header("Predict Stock Prices")
    ticker_symbol = st.text_input("Enter the Stock Ticker for Prediction (e.g., MSFT)")
    future_days = st.slider("Select number of days to predict into the future", 1, 60)

    if st.button("Predict Future Prices"):
        if ticker_symbol:
            # Fetch data using yfinance
            data = yf.download(ticker_symbol, start='2020-01-01', end=datetime.today())
            close_prices = data[['Close']]

            # Normalize the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(close_prices)

            # Use the train_and_predict_lstm function from future.py
            try:
                predicted_prices = train_and_predict_lstm(scaled_data, future_days, scaler)

                # Generate future dates for plotting
                last_date = data.index[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

                # Plot predictions
                plt.figure(figsize=(10, 6))
                plt.plot(data.index, close_prices, label='Historical Daily Close Price')
                plt.plot(future_dates, predicted_prices, label='Predicted Daily Close Price', linestyle='--')
                plt.title(f'Future Price Prediction for {ticker_symbol}')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                st.pyplot(plt)

                st.success("Prediction completed successfully!")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.warning("Please enter a stock ticker symbol.")
