import os
import pandas as pd
from pathlib import Path
import glob


def split_csv_files(folder_path):
    """
    Split all CSV files in a folder by date ranges into train/val/test sets.

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

    for csv_file in csv_files:
        print(f"Processing {csv_file}...")

        try:
            # Read CSV file
            df = pd.read_csv(csv_file)

            # Convert date column to datetime (assuming first column is date)
            date_column = df.columns[0]
            df[date_column] = pd.to_datetime(df[date_column])

            # Define date ranges
            train_mask = (df[date_column] >=
                          '2005-01-01') & (df[date_column] <= '2019-12-31')
            val_mask = (df[date_column] >=
                        '2020-01-01') & (df[date_column] <= '2022-12-31')
            test_mask = (df[date_column] >=
                         '2023-01-01') & (df[date_column] <= '2025-06-30')

            # Split data
            train_data = df[train_mask]
            val_data = df[val_mask]
            test_data = df[test_mask]

            # Get filename without path
            filename = Path(csv_file).name

            # Save split data (only if not empty)
            if not train_data.empty:
                train_data.to_csv(train_dir / filename, index=False)
                print(f"  Train: {len(train_data)} rows saved")

            if not val_data.empty:
                val_data.to_csv(val_dir / filename, index=False)
                print(f"  Val: {len(val_data)} rows saved")

            if not test_data.empty:
                test_data.to_csv(test_dir / filename, index=False)
                print(f"  Test: {len(test_data)} rows saved")

            # Delete original file
            os.remove(csv_file)
            print(f"  Original file deleted")

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")


if __name__ == "__main__":
    # Change this path to your data folder
    data_folder = r"1d-2005"
    split_csv_files(data_folder)
