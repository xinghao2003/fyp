"""
Scripts/classes for fetching data from various APIs (Yahoo Finance, Alpha Vantage).
"""


import yfinance as yf
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from typing import Optional
from src.config import settings


class YahooFinanceCollector:
    """
    Collector for fetching historical market data from Yahoo Finance using yfinance.
    """

    def fetch_historical_data(self, ticker: str, start: str, end: str, interval: str = '1d') -> pd.DataFrame:
        """
        Fetch historical data for a given ticker symbol.
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL').
            start (str): Start date in 'YYYY-MM-DD' format.
            end (str): End date in 'YYYY-MM-DD' format.
            interval (str): Data interval ('1d', '1h', etc.). Default is '1d'.
        Returns:
            pd.DataFrame: Historical price data.
        """
        data = yf.download(ticker, start=start, end=end,
                           interval=interval, progress=False)
        if data is None:
            return pd.DataFrame()
        return data


class AlphaVantageCollector:
    """
    Collector for fetching historical market data from Alpha Vantage using alpha_vantage library.
    """

    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            api_key = getattr(settings, 'ALPHA_VANTAGE_API_KEY', None)
        if not isinstance(api_key, str) or not api_key:
            raise ValueError(
                "Alpha Vantage API key must be provided either via argument or settings.")
        self.ts = TimeSeries(key=api_key, output_format='pandas')

    def fetch_historical_data(self, ticker: str, interval: str = '1d', outputsize: str = 'compact') -> pd.DataFrame:
        """
        Fetch historical data for a given ticker symbol from Alpha Vantage.
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL').
            interval (str): Data interval ('1d', '60min', etc.). Default is '1d'.
            outputsize (str): 'compact' or 'full'. Default is 'compact'.
        Returns:
            pd.DataFrame: Historical price data.
        """
        if interval == '1d':
            data_tuple = self.ts.get_daily(
                symbol=ticker, outputsize=outputsize)
        elif interval.endswith('min'):
            data_tuple = self.ts.get_intraday(
                symbol=ticker, interval=interval, outputsize=outputsize)
        else:
            raise ValueError(f"Unsupported interval: {interval}")

        # Ensure data_tuple is a tuple and first element is a DataFrame
        if isinstance(data_tuple, tuple) and isinstance(data_tuple[0], pd.DataFrame):
            data = data_tuple[0]
        else:
            data = pd.DataFrame()
        return data
