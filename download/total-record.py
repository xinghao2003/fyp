import pandas as pd
import os
from datetime import datetime


def calculate_total_records(folder_path):
    """
    Calculate the total number of records across all CSV files in the specified folder
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

    total_records = 0

    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            record_count = len(df)
            total_records += record_count

            print(f"Processed {csv_file}: {record_count} records")

        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")

    print(f"\nTotal records across all CSV files: {total_records}")
    return total_records


if __name__ == "__main__":
    # Set the folder path - change this to your actual folder path
    folder_path = r"C:\Users\xingh\Desktop\latest_gym\download\1d-2005"

    # If no path provided, use current directory
    if not folder_path:
        folder_path = "."

    calculate_total_records(folder_path)
    print("Processing complete!")
