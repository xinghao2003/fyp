import yfinance as yf
import json
import pandas as pd
import argparse
from datetime import datetime, timedelta


def check_yahoo_symbols(json_file='tickers.json'):
    """Check if symbols exist on Yahoo Finance"""

    # Load symbols from specified JSON file

    try:
        with open(json_file, 'r') as f:
            symbols_by_category = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_file} file not found!")
        return pd.DataFrame()
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file}!")
        return pd.DataFrame()

    results = []

    print("Checking Yahoo Finance symbols...\n")

    for category, symbols in symbols_by_category.items():
        print(f"Checking {category}:")

        for symbol in symbols:
            try:
                # Try to get ticker info
                ticker = yf.Ticker(symbol)
                info = ticker.info

                # Check if we got valid data
                if info and len(info) > 1 and 'symbol' in info:
                    status = "✓ Valid"
                    name = info.get('longName', info.get('shortName', 'N/A'))
                else:
                    # Try getting recent data as fallback
                    hist = ticker.history(period="5d")
                    if not hist.empty:
                        status = "✓ Valid (data available)"
                        name = "N/A"
                    else:
                        status = "✗ Invalid"
                        name = "N/A"

            except Exception as e:
                status = f"✗ Error: {str(e)[:50]}"
                name = "N/A"

            results.append({
                'Category': category,
                'Symbol': symbol,
                'Status': status,
                'Name': name
            })

            print(f"  {symbol:12} - {status}")

        print()

    # Summary
    df = pd.DataFrame(results)
    valid_count = len(df[df['Status'].str.contains('✓')])
    total_count = len(df)

    print(f"\nSUMMARY:")
    print(f"Valid symbols: {valid_count}/{total_count}")
    print(f"Invalid symbols: {total_count - valid_count}/{total_count}")

    # Show invalid symbols
    invalid_symbols = df[df['Status'].str.contains('✗')]
    if not invalid_symbols.empty:
        print(f"\nInvalid symbols:")
        for _, row in invalid_symbols.iterrows():
            print(f"  {row['Symbol']} ({row['Category']})")

    return df


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Check Yahoo Finance symbols from JSON file')
    parser.add_argument('json_file', nargs='?', default='tickers.json',
                        help='JSON file containing symbols by category (default: tickers.json)')

    args = parser.parse_args()

    # Run the check
    results_df = check_yahoo_symbols(args.json_file)
