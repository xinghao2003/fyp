import pandas as pd
from stockstats import wrap
import os
import glob
import argparse

def main():
    parser = argparse.ArgumentParser(description='Add technical indicators to CSV files')
    parser.add_argument('path', help='Path to directory containing CSV files')
    args = parser.parse_args()

    # Get all CSV files in the specified directory
    csv_files = glob.glob(os.path.join(args.path, '*.csv'))

    if not csv_files:
        print(f"No CSV files found in {args.path}")
        return

    # Process each CSV file
    for csv_file in csv_files:
        print(f"Processing {os.path.basename(csv_file)}...")

        # Read the CSV file
        data = pd.read_csv(csv_file)

        # Wrap with stockstats and calculate MACD
        df = wrap(data)
        df.init_all()

        close_10_ema = df["close_10_ema"]
        close_10_sma = df["close_10_sma"]

        # Save the modified dataframe back to the same file
        df.to_csv(csv_file, index=True)

        print(f"Completed {os.path.basename(csv_file)}")

    print("All files processed successfully!")

if __name__ == "__main__":
    main()
