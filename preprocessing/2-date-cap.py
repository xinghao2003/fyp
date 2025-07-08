import pandas as pd
import os
from datetime import datetime


def process_csv_files(folder_path):
    """
    Process all CSV files in the specified folder and remove records before 2005
    """
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist!")
        return

    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the folder!")
        return

    print(f"Found {len(csv_files)} CSV files to process...")

    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Check if 'date' column exists
            if 'date' not in df.columns:
                print(
                    f"Warning: No 'date' column found in {csv_file}, skipping...")
                continue

            original_count = len(df)

            # Convert date column to datetime with UTC to handle mixed timezones
            df['date'] = pd.to_datetime(df['date'], utc=True)

            # Create datetime object for comparison
            cutoff_date = pd.to_datetime('2005-01-01', utc=True)

            # Filter records from 2005 onwards
            df_filtered = df[df['date'] >= cutoff_date]

            filtered_count = len(df_filtered)
            removed_count = original_count - filtered_count

            # Save the filtered data back to the same file
            df_filtered.to_csv(file_path, index=False)

            print(
                f"Processed {csv_file}: Removed {removed_count} records, kept {filtered_count} records")

        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")


if __name__ == "__main__":
    # Set the folder path - change this to your actual folder path
    folder_path = r"1d-2005"

    # If no path provided, use current directory
    if not folder_path:
        folder_path = "."

    process_csv_files(folder_path)
    print("Processing complete!")
