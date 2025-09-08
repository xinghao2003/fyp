#!/usr/bin/env python3
"""
Alpaca Data Downloader - Drop-in replacement for Yahoo Finance downloader

This module provides a compatible interface for downloading historical market data
from Alpaca API instead of Yahoo Finance. It maintains the same function signatures
and output format as the existing Yahoo Finance downloader for seamless integration.

Features:
- Same interface as existing Yahoo Finance downloader
- Automatic data format conversion for compatibility
- Enhanced data quality and reliability
- Support for real-time data streaming (future enhancement)
- Paper trading integration ready

Usage:
    # Direct replacement for Yahoo Finance downloader
    from alpaca_downloader import download_stock_data_alpaca
    
    results = download_stock_data_alpaca(
        symbols=["AAPL", "GOOGL", "MSFT"],
        period="1y",
        interval="1d",
        output_dir="./data"
    )

Requirements:
    - alpaca-trade-api
    - pandas
    - Alpaca account with API credentials (for live data)

Author: Generated for FYP Alpaca Integration
"""

import os
import pandas as pd
import json
from datetime import datetime, timedelta
import time
import argparse

# Try to import Alpaca API - graceful fallback if not available
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("⚠️ alpaca-trade-api not installed. Run: pip install alpaca-trade-api")


class AlpacaDataDownloader:
    """
    Alpaca data downloader with Yahoo Finance compatible interface.
    """
    
    def __init__(self, api_key=None, secret_key=None, paper=True):
        """
        Initialize Alpaca API client.
        
        Args:
            api_key: Alpaca API key (if None, looks for APCA_API_KEY_ID env var)
            secret_key: Alpaca secret key (if None, looks for APCA_API_SECRET_KEY env var)
            paper: Use paper trading endpoint (default: True)
        """
        self.api = None
        self.demo_mode = True
        
        if not ALPACA_AVAILABLE:
            print("⚠️ Alpaca API not available - running in demo mode")
            return
        
        # Try to get credentials from environment if not provided
        if api_key is None:
            api_key = os.getenv('APCA_API_KEY_ID')
        if secret_key is None:
            secret_key = os.getenv('APCA_API_SECRET_KEY')
        
        if api_key and secret_key:
            try:
                base_url = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
                self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
                self.demo_mode = False
                print("✅ Alpaca API client initialized successfully")
                
                # Test connection
                account = self.api.get_account()
                print(f"   Account status: {account.status}")
                
            except Exception as e:
                print(f"⚠️ Failed to initialize Alpaca API: {e}")
                print("   Running in demo mode")
                self.api = None
                self.demo_mode = True
        else:
            print("⚠️ No Alpaca API credentials found")
            print("   Set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables")
            print("   or pass api_key and secret_key parameters")
            print("   Running in demo mode")
    
    def download_stock_data(self, symbols=["AAPL"], period="1y", interval="1d", output_dir=None):
        """
        Download stock data with same interface as Yahoo Finance downloader.
        
        Args:
            symbols: List of stock symbols or single symbol string
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            interval: Data interval (1min, 5min, 15min, 30min, 1h, 1d)
            output_dir: Directory to save CSV files
            
        Returns:
            dict: Results summary with success/failure status for each symbol
        """
        
        # Convert single symbol to list
        if isinstance(symbols, str):
            symbols = [symbols]
        
        # Set output directory
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), f"alpaca-{interval}-{period}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        if self.demo_mode or not self.api:
            return self._demo_download(symbols, period, interval, output_dir)
        
        # Convert period to start/end dates
        end_date = datetime.now()
        period_map = {
            '1d': timedelta(days=1),
            '5d': timedelta(days=5),
            '1mo': timedelta(days=30),
            '3mo': timedelta(days=90),
            '6mo': timedelta(days=180),
            '1y': timedelta(days=365),
            '2y': timedelta(days=730),
            '5y': timedelta(days=1825),
            'max': timedelta(days=3650)  # 10 years max for Alpaca
        }
        start_date = end_date - period_map.get(period, timedelta(days=365))
        
        # Convert interval to Alpaca timeframe
        interval_map = {
            '1min': '1Min', '5min': '5Min', '15min': '15Min',
            '30min': '30Min', '1h': '1Hour', '1d': '1Day'
        }
        timeframe = interval_map.get(interval, '1Day')
        
        results = {}
        total_symbols = len(symbols)
        
        print(f"Downloading data for {total_symbols} symbol(s) from Alpaca API...")
        print(f"Period: {period}, Interval: {interval}")
        print(f"Output directory: {output_dir}")
        print("-" * 60)
        
        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{total_symbols}] Processing {symbol}...")
            
            try:
                # Get historical bars from Alpaca
                bars = self.api.get_bars(
                    symbol,
                    timeframe,
                    start=start_date.isoformat(),
                    end=end_date.isoformat(),
                    adjustment='raw'
                ).df
                
                if bars.empty:
                    print(f"❌ No data downloaded for {symbol}. Please check the symbol.")
                    results[symbol] = {"success": False, "error": "No data available", "records": 0}
                    continue
                
                # Convert to Yahoo Finance format for compatibility
                converted_data = self._convert_to_yahoo_format(bars, symbol)
                
                # Remove any rows with NaN values
                converted_data.dropna(inplace=True)
                
                # Generate filename matching Yahoo Finance format
                filename = f"{symbol}_USD-{interval}-{period}.csv"
                filepath = os.path.join(output_dir, filename)
                
                # Save to CSV
                converted_data.to_csv(filepath, index=False)
                
                print(f"✅ {symbol}: {len(converted_data)} records saved to {filename}")
                
                results[symbol] = {
                    "success": True,
                    "records": len(converted_data),
                    "file": filename,
                    "start_date": str(converted_data['date'].min()),
                    "end_date": str(converted_data['date'].max())
                }
                
                # Add small delay to respect rate limits
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Error downloading {symbol}: {str(e)}")
                results[symbol] = {"success": False, "error": str(e), "records": 0}
        
        return results
    
    def _convert_to_yahoo_format(self, alpaca_df, symbol):
        """
        Convert Alpaca DataFrame to Yahoo Finance format for compatibility.
        
        Args:
            alpaca_df: DataFrame from Alpaca API
            symbol: Stock symbol
            
        Returns:
            DataFrame: Converted data in Yahoo Finance format
        """
        converted_df = alpaca_df.copy()
        
        # Reset index to make timestamp a column
        converted_df = converted_df.reset_index()
        
        # Rename columns to match Yahoo format
        column_mapping = {
            'timestamp': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        
        # Only rename columns that exist
        existing_columns = {k: v for k, v in column_mapping.items() if k in converted_df.columns}
        converted_df = converted_df.rename(columns=existing_columns)
        
        # Add symbol column
        converted_df['symbol'] = symbol
        
        # Select only Yahoo Finance columns (in correct order)
        yahoo_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']
        available_columns = [col for col in yahoo_columns if col in converted_df.columns]
        converted_df = converted_df[available_columns]
        
        # Ensure proper date format (remove timezone info for compatibility)
        if 'date' in converted_df.columns:
            converted_df['date'] = pd.to_datetime(converted_df['date']).dt.tz_localize(None)
        
        return converted_df
    
    def _demo_download(self, symbols, period, interval, output_dir):
        """
        Demo mode that creates sample data files for testing integration.
        """
        print("🎭 Running in DEMO MODE - generating sample data")
        print("   To use real Alpaca data, set API credentials")
        
        results = {}
        
        for symbol in symbols:
            # Generate sample data that matches Yahoo Finance format
            sample_data = self._generate_sample_data(symbol, period, interval)
            
            # Save to CSV file
            filename = f"{symbol}_USD-{interval}-{period}.csv"
            filepath = os.path.join(output_dir, filename)
            sample_data.to_csv(filepath, index=False)
            
            print(f"📁 {symbol}: {len(sample_data)} sample records saved to {filename}")
            
            results[symbol] = {
                "success": True,
                "records": len(sample_data),
                "file": filename,
                "note": "Demo data - not real market data",
                "start_date": str(sample_data['date'].min()),
                "end_date": str(sample_data['date'].max())
            }
        
        return results
    
    def _generate_sample_data(self, symbol, period, interval):
        """
        Generate sample OHLCV data for demo purposes.
        """
        import numpy as np
        
        # Calculate number of periods
        period_days = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90,
            '6mo': 180, '1y': 365, '2y': 730, '5y': 1825, 'max': 1825
        }
        
        interval_freq = {
            '1min': 'T', '5min': '5T', '15min': '15T',
            '30min': '30T', '1h': 'H', '1d': 'D'
        }
        
        days = period_days.get(period, 365)
        freq = interval_freq.get(interval, 'D')
        
        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        if interval == '1d':
            dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days only
        else:
            dates = pd.date_range(start=start_date, end=end_date, freq=freq)
            # Filter to market hours for intraday data
            dates = dates[(dates.hour >= 9) & (dates.hour < 16)]
        
        n_records = len(dates)
        
        # Generate realistic price data using random walk
        np.random.seed(hash(symbol) % 2**32)  # Consistent data for same symbol
        
        base_price = 100 + (hash(symbol) % 500)  # Different base price per symbol
        returns = np.random.normal(0.0002, 0.02, n_records)  # Small daily drift with volatility
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate OHLCV data
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            daily_vol = abs(np.random.normal(0, 0.01))  # Daily volatility
            
            # Generate realistic OHLC within reasonable bounds
            high = price * (1 + daily_vol)
            low = price * (1 - daily_vol)
            open_price = price + np.random.normal(0, price * 0.005)
            close_price = price + np.random.normal(0, price * 0.005)
            
            # Ensure OHLC relationships are maintained
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            # Generate volume (more volume on price moves)
            volume_base = 1000000 + (hash(symbol) % 5000000)
            volume = int(volume_base * (1 + abs(returns[i]) * 10))
            
            data.append({
                'date': date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_price, 2),
                'volume': volume,
                'symbol': symbol
            })
        
        return pd.DataFrame(data)


def download_stock_data_alpaca(symbols=["AAPL"], period="1y", interval="1d", output_dir=None,
                              api_key=None, secret_key=None, paper=True):
    """
    Drop-in replacement for Yahoo Finance download function using Alpaca API.
    
    This function provides the same interface as the existing Yahoo Finance
    downloader but uses Alpaca API for enhanced data quality and features.
    
    Args:
        symbols: List of stock symbols or single symbol string
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
        interval: Data interval (1min, 5min, 15min, 30min, 1h, 1d)
        output_dir: Directory to save CSV files
        api_key: Alpaca API key (optional)
        secret_key: Alpaca secret key (optional)
        paper: Use paper trading endpoint (default: True)
        
    Returns:
        dict: Results summary with success/failure status for each symbol
    """
    
    downloader = AlpacaDataDownloader(api_key=api_key, secret_key=secret_key, paper=paper)
    return downloader.download_stock_data(symbols, period, interval, output_dir)


def compare_with_yahoo_finance(symbols=["AAPL"], period="1mo", interval="1d"):
    """
    Compare Alpaca data with Yahoo Finance data for the same symbols.
    This function demonstrates data quality differences.
    """
    
    print("=" * 60)
    print("ALPACA vs YAHOO FINANCE DATA COMPARISON")
    print("=" * 60)
    
    try:
        # Try to import Yahoo Finance for comparison
        import yfinance as yf
        
        for symbol in symbols:
            print(f"\nComparing {symbol} data:")
            
            # Get Yahoo Finance data
            try:
                ticker = yf.Ticker(symbol)
                yahoo_data = ticker.history(period=period, interval=interval)
                yahoo_records = len(yahoo_data)
                print(f"  Yahoo Finance: {yahoo_records} records")
            except Exception as e:
                print(f"  Yahoo Finance: Error - {e}")
                yahoo_records = 0
            
            # Get Alpaca data (demo mode)
            downloader = AlpacaDataDownloader()
            alpaca_results = downloader.download_stock_data([symbol], period, interval, "/tmp/alpaca_test")
            alpaca_records = alpaca_results[symbol]["records"] if alpaca_results[symbol]["success"] else 0
            print(f"  Alpaca API: {alpaca_records} records")
            
            # Compare
            if yahoo_records > 0 and alpaca_records > 0:
                difference = abs(yahoo_records - alpaca_records)
                print(f"  Difference: {difference} records")
                
                if difference <= 5:
                    print("  ✅ Similar data coverage")
                else:
                    print(f"  ⚠️ Significant difference in record count")
            
    except ImportError:
        print("⚠️ yfinance not available for comparison")


def main():
    """
    Command-line interface for Alpaca data downloader.
    """
    
    parser = argparse.ArgumentParser(description='Download stock data using Alpaca API')
    parser.add_argument('symbols', nargs='*', default=['AAPL'], 
                       help='Stock symbols to download (default: AAPL)')
    parser.add_argument('--period', default='1y',
                       choices=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'],
                       help='Data period (default: 1y)')
    parser.add_argument('--interval', default='1d',
                       choices=['1min', '5min', '15min', '30min', '1h', '1d'],
                       help='Data interval (default: 1d)')
    parser.add_argument('--output-dir', help='Output directory for CSV files')
    parser.add_argument('--compare', action='store_true',
                       help='Compare with Yahoo Finance data')
    parser.add_argument('--demo', action='store_true',
                       help='Run analysis demo')
    
    args = parser.parse_args()
    
    if args.demo:
        # Run comprehensive demo
        print("Running Alpaca integration demo...")
        
        # Test basic download
        results = download_stock_data_alpaca(
            symbols=['AAPL', 'GOOGL', 'MSFT'],
            period='1mo',
            interval='1d',
            output_dir='/tmp/alpaca_demo'
        )
        
        print("\nDownload results:")
        for symbol, result in results.items():
            status = "✅" if result["success"] else "❌"
            print(f"  {status} {symbol}: {result.get('records', 0)} records")
        
        # Compare data quality
        if args.compare:
            compare_with_yahoo_finance(['AAPL'], '1mo', '1d')
        
    else:
        # Regular download
        results = download_stock_data_alpaca(
            symbols=args.symbols,
            period=args.period,
            interval=args.interval,
            output_dir=args.output_dir
        )
        
        # Print results summary
        print(f"\nDownload Summary:")
        print("-" * 40)
        successful = sum(1 for r in results.values() if r["success"])
        total = len(results)
        print(f"Successful: {successful}/{total}")
        
        for symbol, result in results.items():
            if result["success"]:
                print(f"✅ {symbol}: {result['records']} records")
            else:
                print(f"❌ {symbol}: {result['error']}")


if __name__ == "__main__":
    main()