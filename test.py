import yfinance as yf

stock_data = yf.download('AAPL', start=start_date, end=end_date)
