# Stock Data Download Tools

This directory contains tools for downloading and validating stock market data from Yahoo Finance.

## Files

### Scripts

**1-check_yahoo_symbols.py**

- Validates stock symbols from JSON files against Yahoo Finance API
- Checks symbol availability and retrieves basic information
- Provides summary of valid/invalid symbols
- Usage: `python 1-check_yahoo_symbols.py [json_file]`

**2-download_yahoo_stock_data.py**

- Downloads historical stock data from Yahoo Finance
- Supports multiple symbols, periods, and intervals
- Saves data as CSV files with standardized column names
- Usage: `python 2-download_yahoo_stock_data.py <json_file> [--period PERIOD] [--interval INTERVAL]`

### Configuration Files

**tickers.json**

- Main symbol configuration file containing 60+ symbols across 8 asset categories:
  - U.S. Equity Benchmarks (SPY, QQQ, IWM, DIA)
  - U.S. Sector/Thematic ETFs (XLB, XLE, XLF, etc.)
  - Large-Cap Single Stocks (AAPL, MSFT, GOOGL, etc.)
  - Fixed-Income & Credit (TLT, IEF, LQD, etc.)
  - Commodities (GLD, SLV, USO, etc.)
  - Volatility & Tail-Risk (^VIX, VXX)
  - Currencies/FX (EURUSD=X, USDJPY=X, etc.)
  - Crypto (BTC-USD, ETH-USD, SOL-USD)
  - Non-U.S. Large Caps/ADRs (TSM, BABA, etc.)

**tickers-backtest.json**

- Smaller subset with 16 symbols for backtesting purposes
- Contains representative symbols from each major category

## Workflow

1. **Validate Symbols**: Run `1-check_yahoo_symbols.py` to verify symbol availability
2. **Download Data**: Use `2-download_yahoo_stock_data.py` to fetch historical data
3. **Output**: CSV files saved with format `{SYMBOL}_USD-{interval}-{period}.csv`

## Example Usage

```bash
# Check all symbols in main file
python 1-check_yahoo_symbols.py tickers.json

# Download daily data for all symbols
python 2-download_yahoo_stock_data.py tickers.json --period max --interval 1d
```

## Output Format

Downloaded CSV files contain standardized columns:

- `date`: Timestamp
- `open`, `high`, `low`, `close`: Price data
- `volume`: Trading volume
- `symbol`: Stock symbol identifier
