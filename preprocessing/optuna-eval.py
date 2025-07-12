import os
import glob
import pandas as pd
from datetime import datetime


def process_csv_files(folder_path):
    """
    Process all CSV files in the specified folder and print their date ranges.

    Args:
        folder_path (str): Path to the folder containing CSV files
    """
    # Get all CSV files in the folder
    csv_pattern = os.path.join(folder_path, "*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return

    print(f"Found {len(csv_files)} CSV files in {folder_path}")
    print("-" * 80)

    for csv_file in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)

            # Check if 'date' column exists
            if 'date' not in df.columns:
                print(f"{os.path.basename(csv_file)}: No 'date' column found")
                continue

            # Get first and last dates
            if len(df) == 0:
                print(f"{os.path.basename(csv_file)}: Empty file")
                continue

            # Convert date column to datetime if it's not already
            df['date'] = pd.to_datetime(df['date'])

            start_date = df['date'].iloc[0]
            end_date = df['date'].iloc[-1]

            print(f"{os.path.basename(csv_file)}:")
            print(f"  Start Date: {start_date}")
            print(f"  End Date:   {end_date}")
            print(f"  Total Records: {len(df)}")
            print()

        except Exception as e:
            print(f"Error processing {os.path.basename(csv_file)}: {str(e)}")
            print()


if __name__ == "__main__":
    # Specify the folder path containing CSV files
    folder_path = r"1d-2005\val"

    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
    elif not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a directory.")
    else:
        process_csv_files(folder_path)
