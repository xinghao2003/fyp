import os
import pandas as pd
from pathlib import Path
import glob
from datetime import datetime, timedelta


def get_dynamic_date_ranges(df, date_column, min_train_years=3, min_val_years=1, min_test_years=1):
    """
    Calculate dynamic date ranges based on available data while ensuring regime coverage.

    Args:
        df (pd.DataFrame): DataFrame with date column
        date_column (str): Name of the date column
        min_train_years (int): Minimum years for training set
        min_val_years (int): Minimum years for validation set  
        min_test_years (int): Minimum years for test set

    Returns:
        tuple: (train_start, train_end, val_start, val_end, test_start, test_end)
    """
    start_date = df[date_column].min()
    end_date = df[date_column].max()

    # Calculate total years available
    total_years = (end_date - start_date).days / 365.25

    # Check if we have enough data
    min_total_years = min_train_years + min_val_years + min_test_years
    if total_years < min_total_years:
        raise ValueError(
            f"Dataset only spans {total_years:.1f} years, but need at least {min_total_years} years")

    # Strategy 1: Use fixed ratios (70/15/15) if we have enough data
    if total_years >= 7:  # If we have at least 7 years
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15
    else:
        # Strategy 2: Use minimum requirements and distribute remaining time
        remaining_years = total_years - min_total_years
        # Give 60% of extra to training
        train_years = min_train_years + (remaining_years * 0.6)
        # Give 20% of extra to validation
        val_years = min_val_years + (remaining_years * 0.2)
        # Give 20% of extra to test
        test_years = min_test_years + (remaining_years * 0.2)

        train_ratio = train_years / total_years
        val_ratio = val_years / total_years
        test_ratio = test_years / total_years

    # Calculate split points
    train_duration = timedelta(days=int(total_years * train_ratio * 365.25))
    val_duration = timedelta(days=int(total_years * val_ratio * 365.25))

    train_start = start_date
    train_end = train_start + train_duration
    val_start = train_end + timedelta(days=1)
    val_end = val_start + val_duration
    test_start = val_end + timedelta(days=1)
    test_end = end_date

    return train_start, train_end, val_start, val_end, test_start, test_end


def split_csv_files(folder_path, use_dynamic_split=True, fallback_to_fixed=True):
    """
    Split all CSV files in a folder by date ranges into train/val/test sets.

    Args:
        folder_path (str): Path to the folder containing CSV files
        use_dynamic_split (bool): Whether to use dynamic date ranges
        fallback_to_fixed (bool): Whether to fallback to fixed dates if dynamic fails
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

            if use_dynamic_split:
                try:
                    # Try dynamic splitting first
                    train_start, train_end, val_start, val_end, test_start, test_end = get_dynamic_date_ranges(
                        df, date_column
                    )

                    print(
                        f"  Dynamic split - Train: {train_start.date()} to {train_end.date()}")
                    print(
                        f"  Dynamic split - Val: {val_start.date()} to {val_end.date()}")
                    print(
                        f"  Dynamic split - Test: {test_start.date()} to {test_end.date()}")

                    # Create masks for dynamic ranges
                    train_mask = (df[date_column] >= train_start) & (
                        df[date_column] <= train_end)
                    val_mask = (df[date_column] >= val_start) & (
                        df[date_column] <= val_end)
                    test_mask = (df[date_column] >= test_start) & (
                        df[date_column] <= test_end)

                except ValueError as e:
                    if fallback_to_fixed:
                        print(f"  Dynamic split failed: {e}")
                        print(f"  Falling back to fixed date ranges...")
                        # Use original fixed date ranges
                        train_mask = (
                            df[date_column] >= '2005-01-01') & (df[date_column] <= '2019-12-31')
                        val_mask = (
                            df[date_column] >= '2020-01-01') & (df[date_column] <= '2022-12-31')
                        test_mask = (
                            df[date_column] >= '2023-01-01') & (df[date_column] <= '2025-06-30')
                    else:
                        print(f"  Skipping file due to insufficient data: {e}")
                        failed_splits += 1
                        continue
            else:
                # Use fixed date ranges
                train_mask = (
                    df[date_column] >= '2005-01-01') & (df[date_column] <= '2019-12-31')
                val_mask = (df[date_column] >=
                            '2020-01-01') & (df[date_column] <= '2022-12-31')
                test_mask = (
                    df[date_column] >= '2023-01-01') & (df[date_column] <= '2025-06-30')

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

    # Use dynamic splitting with fallback to fixed dates (recommended)
    split_csv_files(data_folder, use_dynamic_split=True,
                    fallback_to_fixed=True)

    # Alternative usage options:
    # Pure dynamic splitting (fails if insufficient data):
    # split_csv_files(data_folder, use_dynamic_split=True, fallback_to_fixed=False)

    # Original fixed splitting:
    # split_csv_files(data_folder, use_dynamic_split=False)
