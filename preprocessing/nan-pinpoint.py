import pandas as pd
import argparse
import sys


def find_nan_lines(csv_file, column_name):
    """
    Find line numbers where NaN values occur in a specified column

    Args:
        csv_file (str): Path to the CSV file
        column_name (str): Name of the column to check for NaN values

    Returns:
        list: Line numbers (1-indexed) containing NaN values
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Check if column exists
        if column_name not in df.columns:
            print(f"Error: Column '{column_name}' not found in the CSV file.")
            print(f"Available columns: {', '.join(df.columns)}")
            return []

        # Find rows with NaN values in the specified column
        nan_mask = df[column_name].isna()
        nan_rows = df[nan_mask]

        # Get line numbers (adding 2 because pandas index is 0-based and we need to account for header)
        nan_lines = (nan_rows.index + 2).tolist()

        return nan_lines

    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        return []
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description='Find NaN records in a CSV column')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument(
        'column_name', help='Name of the column to check for NaN values')

    args = parser.parse_args()

    # Find NaN lines
    nan_lines = find_nan_lines(args.csv_file, args.column_name)

    if nan_lines:
        print(
            f"NaN values found in column '{args.column_name}' at the following lines:")
        for line_num in nan_lines:
            print(f"Line {line_num}")
        print(f"\nTotal NaN records: {len(nan_lines)}")
    else:
        print(f"No NaN values found in column '{args.column_name}'.")


if __name__ == "__main__":
    main()
