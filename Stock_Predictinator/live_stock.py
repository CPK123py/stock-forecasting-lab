

import yfinance as yf

# Define the ticker symbol (e.g., AAPL for Apple)
ticker_symbol = 'ZOMATO.NS'

# Get the stock data (default interval is '1m' for minute-wise data)
stock = yf.Ticker(ticker_symbol)

# Get the live (delayed) data with 1-minute interval
live_data = stock.history(period='1d', interval='1m')

# Display the latest available data
print(live_data.tail(1))  # Shows the most recent data point
