"""
Scripts for parsing and standardizing raw data formats.
"""


import pandas as pd
import os


def parse_csv_generic(filepath, date_col=None, index_col=None, **kwargs):
    """
    Generic CSV parser for financial data.
    Args:
        filepath (str): Path to the CSV file.
        date_col (str, optional): Name of the column to parse as dates.
        index_col (str, optional): Name of the column to set as index.
        **kwargs: Additional arguments for pd.read_csv.
    Returns:
        pd.DataFrame: Parsed DataFrame.
    """
    if date_col:
        df = pd.read_csv(filepath, parse_dates=[
                         date_col], index_col=index_col, **kwargs)
    else:
        df = pd.read_csv(filepath, index_col=index_col, **kwargs)
    return df


def parse_alpha_vantage_csv(filepath):
    """
    Parse Alpha Vantage CSV file and standardize columns.
    Args:
        filepath (str): Path to the Alpha Vantage CSV file.
    Returns:
        pd.DataFrame: Standardized DataFrame.
    """
    df = pd.read_csv(filepath)
    # Try to detect and standardize column names (Alpha Vantage uses e.g. 'timestamp', 'open', 'high', ...)
    col_map = {
        'timestamp': 'date',
        'time': 'date',
        'date': 'date',
        '1. open': 'open',
        '2. high': 'high',
        '3. low': 'low',
        '4. close': 'close',
        '5. volume': 'volume',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
    }
    # Lowercase and strip columns
    df.columns = [c.lower().strip() for c in df.columns]
    # Rename columns if possible
    df = df.rename(columns={c: col_map.get(c, c) for c in df.columns})
    # Parse date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    return df


def prepare_for_gym_anytrading(df):
    """
    Prepare DataFrame for use with gym_anytrading.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV data

    Returns:
        pd.DataFrame: DataFrame formatted for gym_anytrading
    """
    required_cols = ['open', 'high', 'low', 'close']

    # Validate required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for gym_anytrading: {missing_cols}")

    # Create a copy and ensure proper data types
    result = df.copy()

    # Ensure numeric columns are float
    for col in ['open', 'high', 'low', 'close']:
        result[col] = pd.to_numeric(result[col], errors='coerce')

    if 'volume' in result.columns:
        result['volume'] = pd.to_numeric(result['volume'], errors='coerce')

    # Remove any rows with NaN values in OHLC columns
    result = result.dropna(subset=required_cols)

    # Ensure data is sorted by date if date column exists
    if 'date' in result.columns:
        result = result.sort_values('date').reset_index(drop=True)

    # gym_anytrading expects data to be indexed by integer, not date
    # Remove or move date column to avoid confusion
    if 'date' in result.columns:
        # Keep date info but don't use as index
        result = result.reset_index(drop=True)

    # Rename columns to match gym_anytrading expectations (title case)
    column_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'date': 'Date'
    }
    result = result.rename(columns=column_mapping)

    return result
