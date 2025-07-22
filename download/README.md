# Stock Data Download Tools

Tools for downloading and validating stock market data from Yahoo Finance.

## Overview

This directory provides a complete toolkit for acquiring historical stock market data from Yahoo Finance. The tools validate symbol availability, download historical data, and format it consistently for downstream analysis and trading applications.

## Features

- **Symbol validation** - Verify symbol availability against Yahoo Finance API
- **Historical data download** - Fetch OHLCV data with configurable periods and intervals
- **Multiple asset classes** - Support for stocks, ETFs, commodities, crypto, and FX
- **Standardized output** - Consistent CSV format with normalized column names
- **Batch processing** - Process multiple symbols efficiently
- **Flexible configuration** - JSON-based symbol management

## Components

### Scripts

**1-check_yahoo_symbols.py**
- Validates stock symbols from JSON files against Yahoo Finance API
- Checks symbol availability and retrieves basic information
- Provides summary of valid/invalid symbols

**2-download_yahoo_stock_data.py**
- Downloads historical stock data from Yahoo Finance
- Supports multiple symbols, periods, and intervals
- Saves data as CSV files with standardized column names

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

## Usage

### Basic Workflow

1. **Validate Symbols**: Run `1-check_yahoo_symbols.py` to verify symbol availability
2. **Download Data**: Use `2-download_yahoo_stock_data.py` to fetch historical data
3. **Output**: CSV files saved with format `{SYMBOL}_USD-{interval}-{period}.csv`

### Command Syntax

**Symbol Validation:**
```bash
python 1-check_yahoo_symbols.py [json_file]
```

**Data Download:**
```bash
python 2-download_yahoo_stock_data.py <json_file> [--period PERIOD] [--interval INTERVAL]
```

## Examples

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
