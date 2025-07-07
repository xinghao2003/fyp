import os
import pandas as pd
import glob


def validate_pkl_columns(folder_path):
    """
    Check all PKL files in folder and subfolders for required columns.

    Args:
        folder_path (str): Path to the folder to scan

    Returns:
        dict: Dictionary with file paths as keys and missing columns as values
    """

    # Required columns (based on what gym-compatible.py generates)
    required_columns = [
        'open', 'high', 'low', 'close', 'volume', 'rate', 'middle', 'tp', 'boll',
        'boll_ub', 'boll_lb', 'macd', 'macds', 'macdh', 'pvo', 'pvos', 'pvoh', 'ppo',
        'ppos', 'ppoh', 'qqe', 'qqel', 'qqes', 'cr', 'cr-ma1', 'cr-ma2', 'cr-ma3',
        'tr', 'dx', 'adx', 'adxr', 'log-ret', 'wt1', 'wt2', 'supertrend_ub',
        'supertrend_lb', 'supertrend', 'bop', 'cti', 'eribull', 'eribear', 'rvgi',
        'rvgis', 'kst', 'num', 'ao', 'aroon', 'atr', 'cci', 'change', 'chop', 'cmo',
        'coppock', 'dma', 'ichimoku', 'inertia', 'ftr', 'kama', 'kdjk', 'kdjd', 'kdjj',
        'ker', 'mfi', 'ndi', 'pdi', 'pgo', 'psl', 'rsi', 'rsv', 'stochrsi', 'tema',
        'trix', 'wr', 'vr', 'vwma', 'close_10_ema', 'close_10_sma'
    ]

    required_columns_set = set(required_columns)
    invalid_files = {}
    valid_files = []
    empty_files = []
    error_files = {}

    # Find all PKL files recursively
    pkl_pattern = os.path.join(folder_path, '**', '*.pkl')
    pkl_files = glob.glob(pkl_pattern, recursive=True)

    print(f"Found {len(pkl_files)} PKL files to validate...")

    for pkl_file in pkl_files:
        try:
            # Read the pickle file
            df = pd.read_pickle(pkl_file)

            # Check if file has no records
            if len(df) == 0:
                empty_files.append(pkl_file)
                continue

            file_columns_set = set(df.columns)

            # Check for missing columns
            missing_columns = required_columns_set - file_columns_set

            if missing_columns:
                invalid_files[pkl_file] = sorted(list(missing_columns))
            else:
                valid_files.append(pkl_file)
                # Print additional info for valid files
                print(
                    f"✓ {os.path.basename(pkl_file)}: {len(df)} rows, columns: {list(df.columns)}")

        except Exception as e:
            error_files[pkl_file] = str(e)

    # Print results
    print(f"\nValidation Results:")
    print(f"- Valid files: {len(valid_files)}")
    print(f"- Invalid files: {len(invalid_files)}")
    print(f"- Empty files (no records): {len(empty_files)}")
    print(f"- Error files: {len(error_files)}")

    if empty_files:
        print(f"\nFiles with NO RECORDS:")
        for file_path in empty_files:
            print(f"  {file_path}")

    if invalid_files:
        print(f"\nFiles missing required columns:")
        for file_path, missing_cols in invalid_files.items():
            print(f"\n{file_path}")
            print(
                f"  Missing columns ({len(missing_cols)}): {', '.join(missing_cols)}")

    if error_files:
        print(f"\nFiles with read errors:")
        for file_path, error in error_files.items():
            print(f"\n{file_path}")
            print(f"  Error: {error}")

    return {
        'valid_files': valid_files,
        'invalid_files': invalid_files,
        'empty_files': empty_files,
        'error_files': error_files
    }


def main():
    """Main function to run the validation."""
    # Set the folder path to scan
    folder_path = r"1d-2005"

    if not folder_path:
        folder_path = os.getcwd()

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    print(f"Scanning folder: {folder_path}")
    results = validate_pkl_columns(folder_path)

    # Save results to a text file
    output_file = os.path.join(os.path.dirname(
        __file__), 'pkl_validation_results.txt')
    with open(output_file, 'w') as f:
        f.write(f"PKL Validation Results\n")
        f.write(f"Scanned folder: {folder_path}\n")
        f.write(f"Valid files: {len(results['valid_files'])}\n")
        f.write(f"Invalid files: {len(results['invalid_files'])}\n")
        f.write(f"Empty files (no records): {len(results['empty_files'])}\n")
        f.write(f"Error files: {len(results['error_files'])}\n\n")

        if results['empty_files']:
            f.write("Files with NO RECORDS:\n")
            for file_path in results['empty_files']:
                f.write(f"{file_path}\n")
            f.write("\n")

        if results['invalid_files']:
            f.write("Files missing required columns:\n")
            for file_path, missing_cols in results['invalid_files'].items():
                f.write(f"\n{file_path}\n")
                f.write(
                    f"Missing columns ({len(missing_cols)}): {', '.join(missing_cols)}\n")

        if results['error_files']:
            f.write("\nFiles with read errors:\n")
            for file_path, error in results['error_files'].items():
                f.write(f"\n{file_path}\n")
                f.write(f"Error: {error}\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
