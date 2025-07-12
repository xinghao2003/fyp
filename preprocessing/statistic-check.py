import pandas as pd
import numpy as np
import sys


def analyze_csv_statistics(file_path):
    """
    Read a CSV file and output statistical properties for each column
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        print(f"Dataset shape: {df.shape}")
        print("=" * 80)

        # General info about the dataset
        print("DATASET OVERVIEW:")
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        print(f"Column names: {list(df.columns)}")
        print("=" * 80)

        # Data types and missing values
        print("DATA TYPES AND MISSING VALUES:")
        info_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        print(info_df.to_string(index=False))
        print("=" * 80)

        # Statistical summary for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print("NUMERICAL COLUMNS STATISTICS:")
            print(df[numerical_cols].describe())
            print("=" * 80)

        # Statistical summary for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print("CATEGORICAL COLUMNS STATISTICS:")
            for col in categorical_cols:
                print(f"\nColumn: {col}")
                print(f"Unique values: {df[col].nunique()}")
                print(
                    f"Most frequent value: {df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'}")
                print(f"Value counts (top 10):")
                print(df[col].value_counts().head(10))
                print("-" * 40)

        print("=" * 80)
        print("ANALYSIS COMPLETE")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")


if __name__ == "__main__":
    # Get file path from command line argument or prompt user
    if len(sys.argv) > 1:
        csv_file_path = sys.argv[1]
    else:
        csv_file_path = r"C:\Users\xingh\Desktop\fyp-code\preprocessing\1d-2005-bak\AAPL_USD-1d-max_normalized.csv"

    analyze_csv_statistics(csv_file_path)
