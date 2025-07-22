# Financial Data Preprocessing Pipeline

6-step pipeline for preparing financial time series data for reinforcement learning trading environments.

## Overview

This preprocessing pipeline transforms raw stock market CSV data into ML-ready format through technical indicator calculation, normalization, filtering, cleaning, temporal splitting, and final formatting. The pipeline ensures data quality, prevents lookahead bias, and optimizes for reinforcement learning environments.

## Features

- **Technical indicators** - 15+ indicators including MACD, RSI, Bollinger Bands
- **Advanced normalization** - Feature-specific strategies preventing lookahead bias
- **Data quality assurance** - NaN handling and temporal consistency
- **Temporal splitting** - Proper train/validation/test splits for time series
- **ML optimization** - Pickle format compatible with gym environments
- **Batch processing** - Directory-wide processing with progress tracking

## Pipeline Components

Execute scripts in numerical order (1-6) for complete preprocessing:

### 1. Technical Indicators (`1-add-indicators.py`)

Adds technical indicators using the stockstats library.

**Features added:**

- Uses `stockstats.wrap()` and `init_all()` to calculate all available indicators
- Specifically accesses: `close_10_ema`, `close_10_sma`
- Includes: MACD, RSI, ADX, Bollinger Bands, KDJ, ATR, and other technical indicators

**Usage:**

```bash
python 1-add-indicators.py <path_to_csv_directory>
```

### 2. Feature Normalization (`2-normalization.py`)

Comprehensive normalization with different strategies for different feature types.

**Normalization strategies:**

- **Price features** (open, high, low): Normalized as percentage deviation from previous close
- **Close price**: Converted to returns (`pct_change()`)
- **Moving averages/bands**: Normalized relative to previous close
- **Volume**: Log-transformed then rolling z-score (252-day window)
- **Bounded oscillators** (RSI, KDJ): Scaled to [-1, 1] range
- **Unbounded indicators** (MACD, ADX, ATR): Rolling z-score normalization

**Key parameters:**

- `WINDOW = 252` (1 year of trading days)
- `MIN_PERIODS = 30` (minimum for reliable statistics)
- Prevents lookahead bias by using previous values for normalization

**Usage:**

```bash
python 2-normalization.py <path_to_csv_directory>
```

### 3. Date Range Filtering (`3-date-cap.py`)

Filters data to a consistent date range across all files.

**Default range:** 2015-01-01 to 2025-06-30 (configurable)

- Handles mixed timezone data by converting to UTC
- Removes records outside the specified range

**Usage:**

```bash
python 3-date-cap.py <path_to_csv_directory> [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]
```

### 4. Data Cleaning (`4-cleaning.py`)

Removes initial rows containing NaN values from technical indicator calculations.

**Process:**

- Checks first 30 rows for NaN values in technical indicators
- Removes consecutive rows with NaN from the beginning
- Preserves data integrity by only removing initial incomplete records

**Monitored columns:**

- Base: `open, close, high, low, volume, macd, rsi, close_10_sma, close_10_ema, adx, boll_ub, boll_lb, boll, kdjk, kdjd, kdjj, atr`
- Normalized: All corresponding `norm_*` columns

**Usage:**

```bash
python 4-cleaning.py <path_to_csv_directory>
```

### 5. Train/Validation/Test Split (`5-split-fix.py`)

Splits data temporally into training, validation, and test sets.

**Split strategy:**

- **Training**: Until 2020-12-31
- **Validation**: 2021-01-01 to 2024-12-31 (4-year period)
- **Test**: 2025-01-01 onwards

**Features:**

- Creates `train/`, `val/`, `test/` subdirectories
- Deletes original files after successful splitting
- Validates all splits have data before proceeding
- 4-year validation period ensures stable Sharpe ratio calculation

**Usage:**

```bash
python 5-split-fix.py <path_to_csv_directory>
```

### 6. Gym Environment Formatting (`6-prepare-gym-compatible-data.py`)

Converts CSV files to pickle format compatible with `MultiDatasetTradingEnv`.

**Process:**

- Converts date column to UTC-naive DatetimeIndex
- Validates presence of all required columns
- Reports NaN values (but doesn't drop them, assuming data is clean)
- Converts to pickle format and removes original CSV

**Required columns:**

```python
['open', 'close', 'high', 'low', 'volume', 'macd', 'rsi', 'close_10_sma', 
 'close_10_ema', 'adx', 'boll_ub', 'boll_lb', 'boll', 'kdjk', 'kdjd', 
 'kdjj', 'atr'] + corresponding norm_* columns
```

**Usage:**

```bash
python 6-prepare-gym-compatible-data.py <path_to_csv_directory>
```

## Usage

### Complete Pipeline Execution

```bash
# Step 1: Add technical indicators
python 1-add-indicators.py /path/to/raw/csvs

# Step 2: Apply normalization
python 2-normalization.py /path/to/raw/csvs

# Step 3: Filter by date range
python 3-date-cap.py /path/to/raw/csvs

# Step 4: Clean NaN values
python 4-cleaning.py /path/to/raw/csvs

# Step 5: Split into train/val/test
python 5-split-fix.py /path/to/raw/csvs

# Step 6: Convert to pickle format
python 6-prepare-gym-compatible-data.py /path/to/raw/csvs
```

### Individual Script Usage

```bash
# Date filtering with custom range
python 3-date-cap.py /path/to/csvs --start-date 2020-01-01 --end-date 2024-12-31

# Processing specific directories
python 1-add-indicators.py /specific/directory
```

## Configuration

### Default Parameters
- **Date range**: 2015-01-01 to 2025-06-30
- **Normalization window**: 252 days (1 trading year)
- **Minimum periods**: 30 days for reliable statistics
- **Data splits**: Train (until 2020), Val (2021-2024), Test (2025+)

### Key Settings
- `WINDOW = 252` (rolling window for normalization)
- `MIN_PERIODS = 30` (minimum data points for calculations)
- Prevents lookahead bias in all normalization steps

## Output

### Data Flow
```
Raw CSV → Technical Indicators → Normalization → Date Filter → Clean NaNs → Split → Pickle
```

### File Formats

**Input (Raw CSV):**
```csv
date,open,high,low,close,volume
2023-06-26 09:30:00-04:00,186.83,187.12,185.95,186.45,9851788
```

**Output (Final Pickle):**
- DatetimeIndex (UTC-naive, sorted ascending)
- All original OHLCV data
- Technical indicators (MACD, RSI, etc.)
- Normalized features (norm_* columns)
- Ready for `MultiDatasetTradingEnv`

## Examples

```bash
# Process single directory
python 1-add-indicators.py ./stock_data

# Custom date range
python 3-date-cap.py ./data --start-date 2020-01-01 --end-date 2024-12-31

# Full pipeline for multiple datasets
for dir in data/*/; do
    python 1-add-indicators.py "$dir"
    python 2-normalization.py "$dir"
    python 3-date-cap.py "$dir"
    python 4-cleaning.py "$dir"
    python 5-split-fix.py "$dir"
    python 6-prepare-gym-compatible-data.py "$dir"
done
```

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computations
- stockstats: Technical indicator calculations
- pathlib: File system operations
