import yfinance as yf
import pandas as pd
import datetime
import os
import json


def download_stock_data(symbols=["AAPL"], period="1y", interval="1h", output_dir=None):
    """
    Download stock data from Yahoo Finance for multiple symbols

    Args:
        symbols: List of stock symbols or single symbol string (e.g., ["AAPL", "MSFT", "GOOGL"] or "AAPL")
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        output_dir: Directory to save CSV files (defaults to script directory)

    Returns:
        dict: Results summary with success/failure status for each symbol
    """
    # Convert single symbol to list
    if isinstance(symbols, str):
        symbols = [symbols]

    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(__file__)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    total_symbols = len(symbols)

    print(
        f"Downloading data for {total_symbols} symbol(s) with {interval} interval for {period} period...")
    print(f"Output directory: {output_dir}")
    print("-" * 60)

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{total_symbols}] Processing {symbol}...")

        try:
            # Create ticker object
            ticker = yf.Ticker(symbol)

            # Download historical data
            data = ticker.history(period=period, interval=interval)

            if data.empty:
                print(
                    f"❌ No data downloaded for {symbol}. Please check the symbol.")
                results[symbol] = {"success": False,
                                   "error": "No data available", "records": 0}
                continue

            # Reset index to make datetime a column
            data.reset_index(inplace=True)

            # Handle the datetime column name - it could be 'Date' or 'Datetime'
            datetime_col = None
            if 'Date' in data.columns:
                datetime_col = 'Date'
            elif 'Datetime' in data.columns:
                datetime_col = 'Datetime'

            if datetime_col is None:
                print(f"❌ No datetime column found for {symbol}")
                results[symbol] = {
                    "success": False, "error": "No datetime column found", "records": 0}
                continue

            # Rename columns to match trading env format
            rename_dict = {
                datetime_col: 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            data.rename(columns=rename_dict, inplace=True)

            # Select only needed columns (check if they exist first)
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            available_cols = [
                col for col in required_cols if col in data.columns]
            data = data[available_cols]

            # Remove any rows with NaN values
            data.dropna(inplace=True)

            # Generate filename with symbol and parameters
            filename = f"{symbol}_USD-{interval}-{period}.csv"
            filepath = os.path.join(output_dir, filename)

            # Save to CSV
            data.to_csv(filepath, index=False)

            print(
                f"✅ Successfully downloaded {len(data)} records for {symbol}")
            print(f"   📄 Saved to: {filename}")

            # Only show date range if date column exists and has data
            if 'date' in data.columns and len(data) > 0:
                print(
                    f"   📅 Date range: {data['date'].min()} to {data['date'].max()}")
                date_range_str = f"{data['date'].min()} to {data['date'].max()}"
            else:
                date_range_str = "Date range unavailable"

            results[symbol] = {
                "success": True,
                "records": len(data),
                "filepath": filepath,
                "date_range": date_range_str
            }

        except Exception as e:
            print(f"❌ Error downloading {symbol}: {e}")
            results[symbol] = {"success": False, "error": str(e), "records": 0}

        print()  # Add spacing between symbols

    return results


def print_download_summary(results):
    """Print a summary of download results"""
    successful = sum(1 for r in results.values() if r["success"])
    total = len(results)
    total_records = sum(r.get("records", 0) for r in results.values())

    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"✅ Successful downloads: {successful}/{total}")
    print(f"📊 Total records downloaded: {total_records}")
    print()

    for symbol, result in results.items():
        if result["success"]:
            print(
                f"✅ {symbol}: {result['records']} records - {result['date_range']}")
        else:
            print(f"❌ {symbol}: Failed - {result['error']}")


# Legacy function for backward compatibility
def download_aapl_data(period="1y", interval="1h", filename="AAPL_USD-Hourly.csv"):
    """Legacy function - use download_stock_data instead"""
    print("⚠️  This function is deprecated. Use download_stock_data instead.")
    result = download_stock_data(["AAPL"], period, interval)
    return result["AAPL"]["success"]


if __name__ == "__main__":
    # Load symbols from tickers.json

    try:
        with open('tickers.json', 'r') as f:
            symbols_by_category = json.load(f)
    except FileNotFoundError:
        print("Error: tickers.json file not found!")
        exit(1)
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in tickers.json!")
        exit(1)

    # Flatten all symbols into a single list
    all_symbols = []
    for category, symbols in symbols_by_category.items():
        all_symbols.extend(symbols)

    print(
        f"Downloading data for {len(all_symbols)} symbols across all categories")
    print("Configuration: 2 years of data with 1-hour intervals")
    print("-" * 80)

    # Download all symbols
    results = download_stock_data(
        symbols=all_symbols,
        period="5y",
        interval="1d"
    )

    # Print detailed summary by category
    print("=" * 80)
    print("DETAILED DOWNLOAD SUMMARY BY CATEGORY")
    print("=" * 80)

    for category, symbols in symbols_by_category.items():
        successful = sum(1 for symbol in symbols if results[symbol]["success"])
        total = len(symbols)
        print(f"\n{category}: {successful}/{total} successful")

        for symbol in symbols:
            result = results[symbol]
            if result["success"]:
                print(f"  ✅ {symbol}: {result['records']} records")
            else:
                print(f"  ❌ {symbol}: {result['error']}")

    print_download_summary(results)
