import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def check_yahoo_symbols():
    """Check if symbols exist on Yahoo Finance"""

    # Define symbols by category
    symbols_by_category = {
        "U.S. Equity Benchmarks": ["SPY", "QQQ", "IWM", "DIA"],
        "U.S. Sector / Thematic ETFs": ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLY", "XLV", "XLU", "XLRE", "XLC"],
        "Large-Cap Single Stocks": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA", "HD", "NKE", "WMT",
                                    "KO", "PG", "JPM", "BAC", "GS", "JNJ", "LLY", "PFE", "UNH", "BA", "CAT",
                                    "XOM", "CVX", "COP", "NEE", "AMT"],
        "Fixed-Income & Credit": ["TLT", "IEF", "SHY", "LQD", "HYG"],
        "Commodities": ["GLD", "SLV", "USO", "UNG", "DBC"],
        "Volatility & Tail-Risk": ["^VIX", "VXX"],
        "Currencies (FX)": ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X"],
        "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
        "Non-U.S. Large Caps / ADRs": ["TSM", "BABA", "TM", "RIO", "HSBC"]
    }

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
    # Install required packages if not available
    try:
        import yfinance
    except ImportError:
        print("Installing yfinance...")
        import subprocess
        subprocess.check_call(["pip", "install", "yfinance"])
        import yfinance as yf

    # Run the check
    results_df = check_yahoo_symbols()
