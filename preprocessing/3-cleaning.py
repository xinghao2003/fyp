import os
import pandas as pd
import numpy as np
from sympy import root

COLUMNS = [
    "rate", "boll_ub", "boll_lb", "dx", "adx", "adxr",
    "wt1", "wt2", "cci", "rsi", "stochrsi", "cr-ma1", "cr-ma2", "cr-ma3"
]


def clean_first_record(csv_path):
    df = pd.read_csv(csv_path)
    if df.empty:
        return False

    # Check first 5 records or all records if less than 5
    max_check = min(5, len(df))
    rows_to_delete = 0

    # Find consecutive rows with null values from the beginning
    for i in range(max_check):
        row = df.iloc[i]
        if row[COLUMNS].isnull().any():  # If any column has null value
            rows_to_delete = i + 1
        else:
            break  # Stop at first row without null values

    if rows_to_delete > 0:
        df = df.iloc[rows_to_delete:]
        df.to_csv(csv_path, index=False)
        return True
    return False


def main():
    root_folder = r"1d-2005"
    changed_files = []
    for dirpath, _, filenames in os.walk(root_folder):
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
