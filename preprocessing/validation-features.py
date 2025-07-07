import os
import pandas as pd
import glob


def validate_csv_records(folder_path):
    """
    Check all CSV files in folder and subfolders for records (non-empty files).

    Args:
        folder_path (str): Path to the folder to scan

    Returns:
        dict: Dictionary with file paths as keys and record counts as values
    """

    empty_files = {}
    valid_files = {}
    error_files = {}

    # Find all CSV files recursively
    csv_pattern = os.path.join(folder_path, '**', '*.csv')
    csv_files = glob.glob(csv_pattern, recursive=True)

    print(f"Found {len(csv_files)} CSV files to validate...")

    for csv_file in csv_files:
        try:
            # Read the CSV file and check number of records
            df = pd.read_csv(csv_file)
            record_count = len(df)

            if record_count == 0:
                empty_files[csv_file] = record_count
            else:
                valid_files[csv_file] = record_count

        except Exception as e:
            error_files[csv_file] = str(e)

    # Print results
    print(f"\nValidation Results:")
    print(f"- Files with records: {len(valid_files)}")
    print(f"- Empty files: {len(empty_files)}")
    print(f"- Error files: {len(error_files)}")

    if empty_files:
        print(f"\nFiles with no records:")
        for file_path in empty_files.keys():
            print(f"  {file_path}")

    if error_files:
        print(f"\nFiles with read errors:")
        for file_path, error in error_files.items():
            print(f"\n{file_path}")
            print(f"  Error: {error}")

    return {
        'valid_files': valid_files,
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
    results = validate_csv_records(folder_path)

    # Save results to a text file
    output_file = os.path.join(os.path.dirname(
        __file__), 'record_validation_results.txt')
    with open(output_file, 'w') as f:
        f.write(f"CSV Record Validation Results\n")
        f.write(f"Scanned folder: {folder_path}\n")
        f.write(f"Files with records: {len(results['valid_files'])}\n")
        f.write(f"Empty files: {len(results['empty_files'])}\n")
        f.write(f"Error files: {len(results['error_files'])}\n\n")

        if results['valid_files']:
            f.write("Files with records:\n")
            for file_path, record_count in results['valid_files'].items():
                f.write(f"{file_path} - {record_count} records\n")

        if results['empty_files']:
            f.write("\nFiles with no records:\n")
            for file_path in results['empty_files'].keys():
                f.write(f"{file_path}\n")

        if results['error_files']:
            f.write("\nFiles with read errors:\n")
            for file_path, error in results['error_files'].items():
                f.write(f"\n{file_path}\n")
                f.write(f"Error: {error}\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
