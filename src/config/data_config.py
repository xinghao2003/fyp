"""
Configuration for data selection and fetching.
"""

# List of stock ticker symbols to fetch and process.
TICKERS = ["AAPL"]

# Date range for historical data.
START_DATE = "2015-01-01"
END_DATE = "2020-12-31"

# Data fetching interval and size.
INTERVAL = "1d"        # e.g., "1d", "60min"
OUTPUTSIZE = "full"    # or "compact"
