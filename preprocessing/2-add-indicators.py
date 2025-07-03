import pandas as pd
from stockstats import wrap
import os
import glob

# Get all CSV files in the 1d-2005 directory
csv_files = glob.glob(
    r'C:\Users\xingh\Desktop\latest_gym\preprocessing\1d-2005\*.csv')

# Process each CSV file
for csv_file in csv_files:
    print(f"Processing {os.path.basename(csv_file)}...")

    # Read the CSV file
    data = pd.read_csv(csv_file)

    # Wrap with stockstats and calculate MACD
    df = wrap(data)
    macd = df['macd']  # This adds the MACD column to the dataframe

    # Save the modified dataframe back to the same file
    df.to_csv(csv_file, index=True)

    print(f"Completed {os.path.basename(csv_file)}")

print("All files processed successfully!")
