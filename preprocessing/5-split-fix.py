import os
import pandas as pd
from pathlib import Path
import glob
from datetime import datetime, timedelta


"""
Reason:
1.  Provides a solid four-year period for evaluation, which is generally long enough to calculate a stable and meaningful Sharpe ratio. It captures a variety of market conditions.
2.  Overlapping time range of all validation datasets.

Hence, the Sharpe ratio calculation in evaluation is a fair and accurate representation of agent's performance across the diverse set of assets in multi-dataset environment.
"""


def split_csv_files(folder_path):
    """
    Split all CSV files in a folder by date ranges into train/val/test sets.
    Train: until end of 2020
    Val: 2021-01-01 to 2024-12-31
    Test: 2025 onwards

    Args:
        folder_path (str): Path to the folder containing CSV files
    """
    folder_path = Path(folder_path)

    # Create subfolders if they don't exist
    train_dir = folder_path / "train"
    val_dir = folder_path / "val"
    test_dir = folder_path / "test"

    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    # Find all CSV files in the folder (excluding subfolders)
    csv_files = glob.glob(str(folder_path / "*.csv"))

    # Statistics tracking
    successful_splits = 0
    failed_splits = 0
    empty_splits = 0

    for csv_file in csv_files:
        print(f"Processing {csv_file}...")

        try:
            # Read CSV file
            df = pd.read_csv(csv_file)

            # Convert date column to datetime (assuming first column is date)
            date_column = df.columns[0]
            df[date_column] = pd.to_datetime(df[date_column])

            # Sort by date to ensure proper order
            df = df.sort_values(date_column)

            # Fixed date ranges
            train_mask = df[date_column] <= '2020-12-31'
            val_mask = (df[date_column] >=
                        '2021-01-01') & (df[date_column] <= '2024-12-31')
            test_mask = df[date_column] >= '2025-01-01'

            # Split data
            train_data = df[train_mask]
            val_data = df[val_mask]
            test_data = df[test_mask]

            # Get filename without path
            filename = Path(csv_file).name

            # Check if we have data in all splits
            has_train = not train_data.empty
            has_val = not val_data.empty
            has_test = not test_data.empty

            if not (has_train and has_val and has_test):
                print(
                    f"  Warning: Missing data in some splits - Train: {has_train}, Val: {has_val}, Test: {has_test}")
                if not has_train:
                    print(f"  This is problematic - no training data available!")
                    empty_splits += 1

            # Save split data (only if not empty)
            if has_train:
                train_data.to_csv(train_dir / filename, index=False)
                print(f"  Train: {len(train_data)} rows saved")

            if has_val:
                val_data.to_csv(val_dir / filename, index=False)
                print(f"  Val: {len(val_data)} rows saved")

            if has_test:
                test_data.to_csv(test_dir / filename, index=False)
                print(f"  Test: {len(test_data)} rows saved")

            # Delete original file
            os.remove(csv_file)
            print(f"  Original file deleted")
            successful_splits += 1

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            failed_splits += 1

    # Print summary
    print(f"\nSplitting Summary:")
    print(f"  Successful splits: {successful_splits}")
    print(f"  Failed splits: {failed_splits}")
    print(f"  Splits with missing training data: {empty_splits}")


if __name__ == "__main__":
    # Change this path to your data folder
    data_folder = r"1d-2005"

    # Split using fixed date ranges
    split_csv_files(data_folder)
