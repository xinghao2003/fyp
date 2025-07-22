import os
import pandas as pd
import numpy as np
from sympy import root

COLUMNS = ['open', 'close', 'high', 'low', 'volume', 'macd', 'rsi', 'close_10_sma',
           'close_10_ema', 'adx', 'boll_ub', 'boll_lb', 'boll', 'kdjk', 'kdjd', 'kdjj', 'atr']

NORM_COLUMNS = [f'norm_{col}' for col in COLUMNS]


def clean_first_record(csv_path):
    df = pd.read_csv(csv_path)
    if df.empty:
        return False

    # Check first 30, due to normalization
    max_check = min(30, len(df))
    rows_to_delete = 0

    # Find consecutive rows with null values from the beginning
    for i in range(max_check):
        row = df.iloc[i]
        if row[COLUMNS + NORM_COLUMNS].isnull().any():  # If any column has null value
            rows_to_delete = i + 1
        else:
            break  # Stop at first row without null values

    if rows_to_delete > 0:
        df = df.iloc[rows_to_delete:]
        df.to_csv(csv_path, index=False)
        return True
    return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean first records with null values from CSV files')
    parser.add_argument('path', help='Path to directory containing CSV files')
    args = parser.parse_args()
    
    changed_files = []
    for dirpath, _, filenames in os.walk(args.path):
        for fname in filenames:
            if fname.lower().endswith('.csv'):
                fpath = os.path.join(dirpath, fname)
                try:
                    if clean_first_record(fpath):
                        print(f"Changed: {fpath}")
                        changed_files.append(fpath)
                except Exception as e:
                    print(f"Error processing {fpath}: {e}")
    if not changed_files:
        print("No files changed.")


if __name__ == "__main__":
    main()
