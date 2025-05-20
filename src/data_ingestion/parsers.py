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
