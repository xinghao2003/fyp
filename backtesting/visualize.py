import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from pathlib import Path
import numpy as np


def extract_symbol_from_filename(filename):
    """Extract symbol name from file column value"""
    # Remove file extension and extract symbol part
    # Examples: AGG_USD-1d-max.csv -> AGG, AVAX-USD_USD-1d-max.csv -> AVAX
    base_name = filename.replace('.csv', '')

    # Handle different patterns
    if '_USD-1d-max' in base_name:
        symbol = base_name.split('_USD-1d-max')[0]
    elif '-USD_USD-1d-max' in base_name:
        symbol = base_name.split('-USD_USD-1d-max')[0]
    elif '=X_USD-1d-max' in base_name:
        symbol = base_name.split('=X_USD-1d-max')[0]
    else:
        # Fallback: take everything before first underscore or dash
        symbol = re.split('[_-]', base_name)[0]

    return symbol


def extract_strategy_from_csv_name(csv_filename):
    """Extract strategy name from CSV filename"""
    # Examples:
    # rl_backtest_summary_best_model_20250716_155114.csv -> RL_best_model
    # backtest_summary_RsiReversion_20250716_154919.csv -> RsiReversion

    base_name = csv_filename.replace('.csv', '')

    if base_name.startswith('rl_backtest_summary_'):
        # RL strategy
        parts = base_name.split('_')
        if 'best_model' in parts:
            return 'RL_best_model'
        else:
            return 'RL_' + '_'.join(parts[3:-2])  # Remove timestamp
    elif base_name.startswith('backtest_summary_'):
        # Traditional strategy
        parts = base_name.split('_')
        return parts[2]  # Strategy name is after 'backtest_summary_'
    else:
        # Fallback
        return base_name.split('_')[0]


def read_backtest_csvs(folder_path):
    """Read all CSV files from folder and combine into a single DataFrame"""
    all_data = []

    folder = Path(folder_path)
    csv_files = list(folder.glob('*.csv'))

    if not csv_files:
        raise ValueError(f"No CSV files found in {folder_path}")

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            # Add strategy name
            strategy = extract_strategy_from_csv_name(csv_file.name)
            df['Strategy'] = strategy
            df['CSV_File'] = csv_file.name

            # Extract symbol from File column
            df['Symbol'] = df['File'].apply(extract_symbol_from_filename)

            all_data.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    if not all_data:
        raise ValueError("No valid CSV files could be read")

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


def create_comparison_plot(df, metric, title=None, figsize=(12, 8)):
    """Create a grouped bar plot comparing metric across symbols and strategies"""

    if metric not in df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in data. Available columns: {list(df.columns)}")

    # Pivot data for plotting
    pivot_df = df.pivot(index='Symbol', columns='Strategy', values=metric)

    # Create the plot
    plt.figure(figsize=figsize)
    ax = pivot_df.plot(kind='bar', ax=plt.gca(), width=0.8)

    # Customize the plot
    plt.title(
        title or f'{metric} Comparison Across Symbols and Strategies', fontsize=14, pad=20)
    plt.xlabel('Symbol', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def create_multiple_plots(df, metrics, output_folder=None):
    """Create multiple comparison plots for different metrics"""

    available_metrics = [col for col in df.columns if col not in [
        'File', 'Strategy', 'CSV_File', 'Symbol']]

    # Filter metrics to only include available ones
    valid_metrics = [m for m in metrics if m in available_metrics]
    invalid_metrics = [m for m in metrics if m not in available_metrics]

    if invalid_metrics:
        print(f"Warning: These metrics are not available: {invalid_metrics}")
        print(f"Available metrics: {available_metrics}")

    if not valid_metrics:
        raise ValueError("No valid metrics provided")

    figures = []

    for metric in valid_metrics:
        try:
            fig = create_comparison_plot(df, metric)
            figures.append((metric, fig))

            # Save plot if output folder specified
            if output_folder:
                output_path = Path(output_folder)
                output_path.mkdir(exist_ok=True)
                fig.savefig(output_path / f'{metric}_comparison.png',
                            dpi=300, bbox_inches='tight')
                print(
                    f"Saved plot: {output_path / f'{metric}_comparison.png'}")

        except Exception as e:
            print(f"Error creating plot for {metric}: {e}")

    return figures


def main():
    """Main function to demonstrate usage"""

    # Configuration
    folder_path = r"C:\Users\xingh\Desktop\fyp-code\backtesting\result\plot"
    output_folder = r"c:\Users\xingh\Desktop\fyp-code\backtesting\plots"

    # Key metrics to plot
    key_metrics = [
        'Win Rate [%]',
        'Return [%]',
        'Sharpe Ratio',
        'Max. Drawdown [%]',
        'CAGR [%]',
        'Profit Factor',
        '# Trades'
    ]

    try:
        # Read and process data
        print("Reading CSV files...")
        df = read_backtest_csvs(folder_path)
        print(
            f"Loaded data for {len(df)} entries across {df['Strategy'].nunique()} strategies")
        print(f"Symbols: {sorted(df['Symbol'].unique())}")
        print(f"Strategies: {sorted(df['Strategy'].unique())}")

        # Create plots
        print("\nCreating comparison plots...")
        figures = create_multiple_plots(df, key_metrics, output_folder)

        print(f"\nCreated {len(figures)} plots")

        # Show plots
        plt.show()

        # Return data for further analysis if needed
        return df, figures

    except Exception as e:
        print(f"Error: {e}")
        return None, None


if __name__ == "__main__":
    df, figures = main()
