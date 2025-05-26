import requests
import pandas as pd
import os
import time
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List, Optional, Dict
from dotenv import load_dotenv
import yfinance as yf

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)


class AlphaVantageDownloader:
    """
    A comprehensive class for downloading stock market data from Alpha Vantage API.

    This class provides methods to download various types of financial data including:
    - Daily, weekly, monthly stock prices
    - Intraday data with multiple frequencies
    - Multiple output formats (CSV, JSON, Parquet)
    - Batch downloading with rate limiting
    - Time range filtering
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://www.alphavantage.co/query",
        default_save_directory: str = "./data",
        default_rate_limit_delay: float = 12.0,
        default_output_size: str = "compact"
    ):
        """
        Initialize the Alpha Vantage downloader.

        Parameters:
        -----------
        api_key : Optional[str]
            Alpha Vantage API key. If None, will load from environment variable.
        base_url : str
            Base URL for Alpha Vantage API
        default_save_directory : str
            Default directory to save downloaded data
        default_rate_limit_delay : float
            Default delay between API calls in seconds
        default_output_size : str
            Default output size ('compact' or 'full')
        """
        self.api_key = api_key or self._load_api_key()
        self.base_url = base_url
        self.default_save_directory = default_save_directory
        self.default_rate_limit_delay = default_rate_limit_delay
        self.default_output_size = default_output_size

        # Frequency to function mapping
        self.frequency_mapping = {
            'daily': 'TIME_SERIES_DAILY',
            'weekly': 'TIME_SERIES_WEEKLY',
            'monthly': 'TIME_SERIES_MONTHLY',
            'intraday': 'TIME_SERIES_INTRADAY'
        }

        # Create default save directory
        os.makedirs(self.default_save_directory, exist_ok=True)
        logger.info(
            f"Initialized AlphaVantageDownloader with save directory: {self.default_save_directory}")

    def _load_api_key(self) -> str:
        """Load Alpha Vantage API key from environment variables."""
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            logger.error(
                "ALPHA_VANTAGE_API_KEY not found in environment variables")
            raise ValueError(
                "ALPHA_VANTAGE_API_KEY not found in environment variables. Please add it to your .env file.")
        logger.debug("Successfully loaded API key from environment variables")
        return api_key

    def download_data(
        self,
        symbols: List[str],
        function: str = "TIME_SERIES_DAILY",
        time_range: Optional[Dict[str, str]] = None,
        frequency: str = "daily",
        output_size: Optional[str] = None,
        adjusted: bool = True,
        save_format: str = "csv",
        save_directory: Optional[str] = None,
        rate_limit_delay: Optional[float] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Download dataset from Alpha Vantage API with customizable parameters.

        Parameters:
        -----------
        symbols : List[str]
            List of stock symbols to download (e.g., ['AAPL', 'GOOGL', 'MSFT'])
        function : str
            Alpha Vantage function name
        time_range : Optional[Dict[str, str]]
            Dictionary with 'start' and 'end' dates in 'YYYY-MM-DD' format
        frequency : str
            Data frequency ('daily', 'weekly', 'monthly', 'intraday')
        output_size : Optional[str]
            'compact' or 'full' (uses default if None)
        adjusted : bool
            Whether to use adjusted prices
        save_format : str
            Format to save data: 'csv', 'json', 'parquet', or 'none'
        save_directory : Optional[str]
            Directory to save data (uses default if None)
        rate_limit_delay : Optional[float]
            Delay between API calls (uses default if None)

        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary with symbol as key and DataFrame as value
        """
        # Use defaults if not provided
        output_size = output_size or self.default_output_size
        save_directory = save_directory or self.default_save_directory
        rate_limit_delay = rate_limit_delay or self.default_rate_limit_delay

        logger.info(f"Starting download for {len(symbols)} symbols: {symbols}")
        logger.debug(
            f"Download parameters - function: {function}, frequency: {frequency}, output_size: {output_size}")

        # Map frequency to function if using default
        if function == "TIME_SERIES_DAILY" and frequency in self.frequency_mapping:
            if frequency == 'daily':
                function = 'TIME_SERIES_DAILY_ADJUSTED' if adjusted else 'TIME_SERIES_DAILY'
            elif frequency == 'weekly':
                function = 'TIME_SERIES_WEEKLY_ADJUSTED' if adjusted else 'TIME_SERIES_WEEKLY'
            elif frequency == 'monthly':
                function = 'TIME_SERIES_MONTHLY_ADJUSTED' if adjusted else 'TIME_SERIES_MONTHLY'
            elif frequency == 'intraday':
                function = 'TIME_SERIES_INTRADAY'

        # Create save directory if needed
        if save_format != 'none':
            os.makedirs(save_directory, exist_ok=True)

        downloaded_data = {}

        for i, symbol in enumerate(symbols):
            logger.info(
                f"Downloading data for {symbol} ({i+1}/{len(symbols)})...")

            try:
                # Download data for single symbol
                df = self._download_single_symbol(
                    symbol=symbol,
                    function=function,
                    frequency=frequency,
                    output_size=output_size,
                    time_range=time_range
                )

                if df is None or df.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue

                # Add symbol column
                df['symbol'] = symbol

                # Store in dictionary
                downloaded_data[symbol] = df

                # Save individual file if requested
                if save_format != 'none':
                    self._save_data(df, symbol, save_format, save_directory)

                logger.info(
                    f"Successfully downloaded {len(df)} records for {symbol}")

            except Exception as e:
                logger.error(
                    f"Error downloading data for {symbol}: {str(e)}", exc_info=True)
                continue

            # Rate limiting
            if i < len(symbols) - 1:
                logger.debug(
                    f"Waiting {rate_limit_delay} seconds to respect rate limits...")
                time.sleep(rate_limit_delay)

        logger.info(
            f"Download complete! Retrieved data for {len(downloaded_data)} symbols.")
        return downloaded_data

    def _download_single_symbol(
        self,
        symbol: str,
        function: str,
        frequency: str,
        output_size: str,
        time_range: Optional[Dict[str, str]] = None
    ) -> Optional[pd.DataFrame]:
        """Download data for a single symbol."""
        logger.debug(
            f"Making API request for {symbol} with function {function}")

        # Build API parameters
        params = {
            'function': function,
            'symbol': symbol,
            'apikey': self.api_key,
            'datatype': 'json',
            'outputsize': output_size
        }

        # Add interval for intraday data
        if function == 'TIME_SERIES_INTRADAY':
            if frequency in ['1min', '5min', '15min', '30min', '60min']:
                params['interval'] = frequency
            else:
                params['interval'] = '5min'  # default

        # Add month parameter for specific time range (intraday only)
        if time_range and function == 'TIME_SERIES_INTRADAY':
            start_date = datetime.strptime(time_range['start'], '%Y-%m-%d')
            params['month'] = start_date.strftime('%Y-%m')

        # Make API request
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()

        data = response.json()

        # Check for API errors
        if 'Error Message' in data:
            logger.error(f"API Error for {symbol}: {data['Error Message']}")
            raise ValueError(f"API Error: {data['Error Message']}")

        if 'Note' in data:
            logger.warning(f"API Note for {symbol}: {data['Note']}")
            return None

        # Extract time series data
        df = self._extract_time_series_data(data, function)

        if df is None:
            return None

        # Filter by time range if specified (except for intraday which uses month parameter)
        if time_range and function != 'TIME_SERIES_INTRADAY':
            df = self._filter_by_time_range(df, time_range)

        return df

    def _extract_time_series_data(self, data: Dict, function: str) -> Optional[pd.DataFrame]:
        """Extract time series data from Alpha Vantage API response."""
        # Map function names to their corresponding keys in the response
        time_series_keys = {
            'TIME_SERIES_INTRADAY': lambda x: next((k for k in x.keys() if 'Time Series' in k), None),
            'TIME_SERIES_DAILY': 'Time Series (Daily)',
            'TIME_SERIES_DAILY_ADJUSTED': 'Time Series (Daily)',
            'TIME_SERIES_WEEKLY': 'Weekly Time Series',
            'TIME_SERIES_WEEKLY_ADJUSTED': 'Weekly Adjusted Time Series',
            'TIME_SERIES_MONTHLY': 'Monthly Time Series',
            'TIME_SERIES_MONTHLY_ADJUSTED': 'Monthly Adjusted Time Series'
        }

        # Get the appropriate key
        if function == 'TIME_SERIES_INTRADAY':
            time_series_key = time_series_keys[function](data)
        else:
            time_series_key = time_series_keys.get(function)

        if not time_series_key or time_series_key not in data:
            logger.error(
                f"Time series key '{time_series_key}' not found in response")
            logger.debug(f"Response data: {data}")
            return None

        time_series_data = data[time_series_key]

        if not time_series_data:
            return None

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series_data, orient='index')

        # Clean column names
        df.columns = [
            col.split('. ')[1] if '. ' in col else col for col in df.columns]

        # Convert to appropriate data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        if 'adjusted close' in df.columns:
            numeric_columns.append('adjusted close')
        if 'dividend amount' in df.columns:
            numeric_columns.append('dividend amount')

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        return df

    def _filter_by_time_range(self, df: pd.DataFrame, time_range: Dict[str, str]) -> pd.DataFrame:
        """Filter DataFrame by time range."""
        start_date = pd.to_datetime(time_range.get('start'))
        end_date = pd.to_datetime(time_range.get('end'))

        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        return df

    def _save_data(self, df: pd.DataFrame, symbol: str, save_format: str, save_directory: str):
        """Save DataFrame to file in specified format."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if save_format.lower() == 'csv':
            filepath = os.path.join(
                save_directory, f"{symbol}_{timestamp}.csv")
            df.to_csv(filepath)
        elif save_format.lower() == 'json':
            filepath = os.path.join(
                save_directory, f"{symbol}_{timestamp}.json")
            df.to_json(filepath, date_format='iso', orient='index')
        elif save_format.lower() == 'parquet':
            filepath = os.path.join(
                save_directory, f"{symbol}_{timestamp}.parquet")
            df.to_parquet(filepath)

        logger.info(f"Saved {symbol} data to {filepath}")

    def download_daily_data(
        self,
        symbols: List[str],
        adjusted: bool = True,
        time_range: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """Convenience method for downloading daily data."""
        function = 'TIME_SERIES_DAILY_ADJUSTED' if adjusted else 'TIME_SERIES_DAILY'
        return self.download_data(
            symbols=symbols,
            function=function,
            frequency='daily',
            time_range=time_range,
            adjusted=adjusted,
            **kwargs
        )

    def download_intraday_data(
        self,
        symbols: List[str],
        interval: str = "5min",
        time_range: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """Convenience method for downloading intraday data."""
        return self.download_data(
            symbols=symbols,
            function='TIME_SERIES_INTRADAY',
            frequency=interval,
            time_range=time_range,
            **kwargs
        )

    def download_weekly_data(
        self,
        symbols: List[str],
        adjusted: bool = True,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """Convenience method for downloading weekly data."""
        function = 'TIME_SERIES_WEEKLY_ADJUSTED' if adjusted else 'TIME_SERIES_WEEKLY'
        return self.download_data(
            symbols=symbols,
            function=function,
            frequency='weekly',
            adjusted=adjusted,
            **kwargs
        )

    def download_monthly_data(
        self,
        symbols: List[str],
        adjusted: bool = True,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """Convenience method for downloading monthly data."""
        function = 'TIME_SERIES_MONTHLY_ADJUSTED' if adjusted else 'TIME_SERIES_MONTHLY'
        return self.download_data(
            symbols=symbols,
            function=function,
            frequency='monthly',
            adjusted=adjusted,
            **kwargs
        )

    def example_usage():
        """Example of how to use the AlphaVantageDownloader class."""
        # Initialize the downloader
        downloader = AlphaVantageDownloader(
            default_save_directory='./stock_data',
            default_rate_limit_delay=12.0
        )

        # Example 1: Download daily data for multiple stocks
        stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        time_range = {'start': '2023-01-01', 'end': '2023-12-31'}

        daily_data = downloader.download_daily_data(
            symbols=stocks,
            time_range=time_range,
            output_size='full',
            save_format='csv'
        )

        # Example 2: Download intraday data using convenience method
        intraday_data = downloader.download_intraday_data(
            symbols=['AAPL'],
            interval='5min',
            output_size='compact',
            save_format='json'
        )

        # Example 3: Download weekly data
        weekly_data = downloader.download_weekly_data(
            symbols=['MSFT', 'GOOGL'],
            save_format='parquet'
        )

        return daily_data

    def download_stock_data():
        """
        Download intraday stock data for AAPL from AlphaVantage.
        At 60-minute intervals.
        For Years of data from 2007 to 2025 will be downloaded.
        This function uses the AlphaVantageDownloader class to fetch the data
        and save it in the specified format and directory.
        Rate limits per day are 25 requests for free users.
        """
        downloader = AlphaVantageDownloader(
            default_save_directory='./stock_data',
            default_rate_limit_delay=12.0
        )

        # Define the date range
        start_date = datetime.strptime('2009-02-01', '%Y-%m-%d')
        end_date = datetime.strptime('2025-04-01', '%Y-%m-%d')

        # Dictionary to store all downloaded data
        all_data = {}

        # Current date for iteration
        current_date = start_date

        logger.info(
            f"Starting multi-month download from {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")

        month_count = 0
        while current_date < end_date:
            month_str = current_date.strftime('%Y-%m')
            logger.info(f"Downloading data for month: {month_str}")

            try:
                # Download data for current month
                monthly_data = downloader.download_intraday_data(
                    symbols=['AAPL'],
                    interval='60min',
                    time_range={'start': current_date.strftime(
                        '%Y-%m-%d'), 'end': current_date.strftime('%Y-%m-%d')},
                    output_size='full',
                    save_format='csv'
                )

                # If data was successfully downloaded, add it to our collection
                if 'AAPL' in monthly_data and not monthly_data['AAPL'].empty:
                    if 'AAPL' not in all_data:
                        all_data['AAPL'] = monthly_data['AAPL']
                    else:
                        # Concatenate with existing data
                        all_data['AAPL'] = pd.concat(
                            [all_data['AAPL'], monthly_data['AAPL']], ignore_index=False)

                    logger.info(
                        f"Successfully downloaded {len(monthly_data['AAPL'])} records for {month_str}")
                else:
                    logger.warning(f"No data found for {month_str}")

                month_count += 1

            except Exception as e:
                logger.error(
                    f"Error downloading data for {month_str}: {str(e)}")

            # Move to next month
            current_date += relativedelta(months=1)

            # # Add a longer delay every 5 requests to be extra careful with rate limits
            # if month_count % 5 == 0:
            #     logger.info(
            #         f"Taking extended break to respect rate limits... (approx. 1 minute)")
            #     time.sleep(60)  # 1 minute break every 5 months

        # Sort the final combined data by date
        if 'AAPL' in all_data:
            all_data['AAPL'] = all_data['AAPL'].sort_index()

            # Save the combined dataset
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            combined_filepath = os.path.join(
                './stock_data', f"AAPL_combined_2007-2025_{timestamp}.csv")
            all_data['AAPL'].to_csv(combined_filepath)
            logger.info(
                f"Saved combined dataset with {len(all_data['AAPL'])} total records to {combined_filepath}")

        logger.info(
            f"Multi-month download complete! Downloaded data for {month_count} months.")
        return all_data


class YahooFinanceDownloader:
    """
    A comprehensive class for downloading stock market data from Yahoo Finance.

    This class provides methods to download various types of financial data including:
    - Daily, weekly, monthly stock prices
    - Intraday data with multiple frequencies
    - Multiple output formats (CSV, JSON, Parquet)
    - Batch downloading with rate limiting
    - Time range filtering
    """

    def __init__(
        self,
        default_save_directory: str = "./data",
        default_rate_limit_delay: float = 1.0,
        default_period: str = "1y"
    ):
        """
        Initialize the Yahoo Finance downloader.

        Parameters:
        -----------
        default_save_directory : str
            Default directory to save downloaded data
        default_rate_limit_delay : float
            Default delay between API calls in seconds
        default_period : str
            Default period for data download
        """
        if yf is None:
            raise ImportError(
                "yfinance library is required. Install with: pip install yfinance")

        self.default_save_directory = default_save_directory
        self.default_rate_limit_delay = default_rate_limit_delay
        self.default_period = default_period

        # Valid periods and intervals
        self.valid_periods = ['1d', '5d', '1mo', '3mo',
                              '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        self.valid_intervals = ['1m', '2m', '5m', '15m', '30m',
                                '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

        # Create default save directory
        os.makedirs(self.default_save_directory, exist_ok=True)
        logger.info(
            f"Initialized YahooFinanceDownloader with save directory: {self.default_save_directory}")

    def download_data(
        self,
        symbols: List[str],
        period: Optional[str] = None,
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        save_format: str = "csv",
        save_directory: Optional[str] = None,
        rate_limit_delay: Optional[float] = None,
        include_actions: bool = True,
        include_dividends: bool = True,
        auto_adjust: bool = True,
        prepost: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Download stock data from Yahoo Finance with customizable parameters.

        Parameters:
        -----------
        symbols : List[str]
            List of stock symbols to download
        period : Optional[str]
            Period to download (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval : str
            Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        start : Optional[str]
            Start date in 'YYYY-MM-DD' format
        end : Optional[str]
            End date in 'YYYY-MM-DD' format
        save_format : str
            Format to save data: 'csv', 'json', 'parquet', or 'none'
        save_directory : Optional[str]
            Directory to save data
        rate_limit_delay : Optional[float]
            Delay between downloads
        include_actions : bool
            Include stock actions (splits, dividends)
        include_dividends : bool
            Include dividend data
        auto_adjust : bool
            Automatically adjust prices for splits and dividends
        prepost : bool
            Include pre and post market data for intraday

        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary with symbol as key and DataFrame as value
        """
        # Use defaults if not provided
        period = period or self.default_period
        save_directory = save_directory or self.default_save_directory
        rate_limit_delay = rate_limit_delay or self.default_rate_limit_delay

        logger.info(
            f"Starting Yahoo Finance download for {len(symbols)} symbols: {symbols}")
        logger.debug(
            f"Parameters - period: {period}, interval: {interval}, start: {start}, end: {end}")

        # Validate parameters
        if interval not in self.valid_intervals:
            raise ValueError(
                f"Invalid interval. Must be one of: {self.valid_intervals}")

        if period and period not in self.valid_periods:
            raise ValueError(
                f"Invalid period. Must be one of: {self.valid_periods}")

        # Create save directory if needed
        if save_format != 'none':
            os.makedirs(save_directory, exist_ok=True)

        downloaded_data = {}

        for i, symbol in enumerate(symbols):
            logger.info(
                f"Downloading data for {symbol} ({i+1}/{len(symbols)})...")

            try:
                # Create ticker object
                ticker = yf.Ticker(symbol)

                # Download data
                if start and end:
                    df = ticker.history(
                        start=start,
                        end=end,
                        interval=interval,
                        actions=include_actions,
                        auto_adjust=auto_adjust,
                        prepost=prepost
                    )
                else:
                    df = ticker.history(
                        period=period,
                        interval=interval,
                        actions=include_actions,
                        auto_adjust=auto_adjust,
                        prepost=prepost
                    )

                if df is None or df.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue

                # Clean column names (convert to lowercase)
                df.columns = [col.lower() for col in df.columns]

                # Add symbol column
                df['symbol'] = symbol

                # Store in dictionary
                downloaded_data[symbol] = df

                # Save individual file if requested
                if save_format != 'none':
                    self._save_data(df, symbol, save_format, save_directory)

                logger.info(
                    f"Successfully downloaded {len(df)} records for {symbol}")

            except Exception as e:
                logger.error(
                    f"Error downloading data for {symbol}: {str(e)}", exc_info=True)
                continue

            # Rate limiting
            if i < len(symbols) - 1:
                logger.debug(f"Waiting {rate_limit_delay} seconds...")
                time.sleep(rate_limit_delay)

        logger.info(
            f"Download complete! Retrieved data for {len(downloaded_data)} symbols.")
        return downloaded_data

    def _save_data(self, df: pd.DataFrame, symbol: str, save_format: str, save_directory: str):
        """Save DataFrame to file in specified format."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if save_format.lower() == 'csv':
            filepath = os.path.join(
                save_directory, f"{symbol}_{timestamp}.csv")
            df.to_csv(filepath)
        elif save_format.lower() == 'json':
            filepath = os.path.join(
                save_directory, f"{symbol}_{timestamp}.json")
            df.to_json(filepath, date_format='iso', orient='index')
        elif save_format.lower() == 'parquet':
            filepath = os.path.join(
                save_directory, f"{symbol}_{timestamp}.parquet")
            df.to_parquet(filepath)

        logger.info(f"Saved {symbol} data to {filepath}")

    def download_daily_data(
        self,
        symbols: List[str],
        period: str = "1y",
        start: Optional[str] = None,
        end: Optional[str] = None,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """Convenience method for downloading daily data."""
        return self.download_data(
            symbols=symbols,
            period=period,
            interval='1d',
            start=start,
            end=end,
            **kwargs
        )

    def download_intraday_data(
        self,
        symbols: List[str],
        interval: str = "5m",
        period: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """Convenience method for downloading intraday data."""
        return self.download_data(
            symbols=symbols,
            period=period,
            interval=interval,
            start=start,
            end=end,
            **kwargs
        )

    def download_weekly_data(
        self,
        symbols: List[str],
        period: str = "2y",
        start: Optional[str] = None,
        end: Optional[str] = None,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """Convenience method for downloading weekly data."""
        return self.download_data(
            symbols=symbols,
            period=period,
            interval='1wk',
            start=start,
            end=end,
            **kwargs
        )

    def download_monthly_data(
        self,
        symbols: List[str],
        period: str = "max",
        start: Optional[str] = None,
        end: Optional[str] = None,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """Convenience method for downloading monthly data."""
        return self.download_data(
            symbols=symbols,
            period=period,
            interval='1mo',
            start=start,
            end=end,
            **kwargs
        )

    def get_stock_info(self, symbol: str) -> Dict:
        """Get detailed stock information."""
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            logger.error(f"Error getting info for {symbol}: {str(e)}")
            return {}

    def get_dividends(self, symbol: str, period: str = "max") -> pd.DataFrame:
        """Get dividend history for a stock."""
        try:
            ticker = yf.Ticker(symbol)
            return ticker.dividends
        except Exception as e:
            logger.error(f"Error getting dividends for {symbol}: {str(e)}")
            return pd.DataFrame()

    def get_splits(self, symbol: str) -> pd.DataFrame:
        """Get stock split history."""
        try:
            ticker = yf.Ticker(symbol)
            return ticker.splits
        except Exception as e:
            logger.error(f"Error getting splits for {symbol}: {str(e)}")
            return pd.DataFrame()

    def example_usage():
        """Example of how to use the YahooFinanceDownloader class."""
        # Initialize the downloader
        downloader = YahooFinanceDownloader(
            default_save_directory='./stock_data',
            default_rate_limit_delay=1.0
        )

        # Example 1: Download daily data for multiple stocks
        stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

        daily_data = downloader.download_daily_data(
            symbols=stocks,
            start='2023-01-01',
            end='2023-12-31',
            save_format='csv'
        )

        # Example 2: Download intraday data
        intraday_data = downloader.download_intraday_data(
            symbols=['AAPL'],
            interval='5m',
            period='1d',
            save_format='json'
        )

        # Example 3: Download weekly data with max period
        weekly_data = downloader.download_weekly_data(
            symbols=['MSFT', 'GOOGL'],
            period='max',
            save_format='parquet'
        )

        # Example 4: Get stock info
        aapl_info = downloader.get_stock_info('AAPL')
        print(f"AAPL Market Cap: {aapl_info.get('marketCap', 'N/A')}")

        return daily_data

    def download_stock_data():
        """
        Download daily stock data for AAPL from Yahoo Finance
        For Years of data from 2007 to 2024 will be downloaded.
        """
        downloader = YahooFinanceDownloader(
            default_save_directory='./stock_data',
            default_rate_limit_delay=1.0
        )

        # Define the date range
        start_date = '2007-01-01'
        end_date = '2024-12-31'

        # Download daily data for AAPL
        try:
            daily_data = downloader.download_daily_data(
                symbols=['AAPL'],
                start=start_date,
                end=end_date,
                save_format='csv'
            )
            logger.info(
                f"Successfully downloaded {len(daily_data.get('AAPL', []))} records for AAPL")
        except Exception as e:
            logger.error(f"Error downloading AAPL data: {str(e)}")

        return daily_data
