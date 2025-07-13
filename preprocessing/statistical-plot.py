import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path


def analyze_csv_files(folder_path, column):
    """
    Analyze all CSV files in the folder for the specified column statistics
    """
    results = {
        'filename': [],
        'min': [],
        'max': [],
        'mean': [],
        'median': []
    }

    folder = Path(folder_path)
    csv_files = list(folder.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return None

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            if column not in df.columns:
                print(
                    f"Warning: '{column}' column not found in {csv_file.name}")
                continue

            col_data = df[column].dropna()

            if len(col_data) == 0:
                print(
                    f"Warning: No valid data in '{column}' column for {csv_file.name}")
                continue

            results['filename'].append(csv_file.name)
            results['min'].append(col_data.min())
            results['max'].append(col_data.max())
            results['mean'].append(col_data.mean())
            results['median'].append(col_data.median())

        except Exception as e:
            print(f"Error processing {csv_file.name}: {str(e)}")

    return pd.DataFrame(results)


def plot_statistical_distributions(stats_df, column, output_dir, save_plots=True):
    """
    Create distribution plots for each statistical property
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    if save_plots:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create subplots for all statistics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Distribution of Statistical Properties for "{column}" Across CSV Files',
                 fontsize=16, fontweight='bold')

    statistics = ['min', 'max', 'mean', 'median']
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum']

    for i, (stat, color) in enumerate(zip(statistics, colors)):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        # Histogram with KDE
        ax.hist(stats_df[stat], bins=20, alpha=0.7,
                color=color, edgecolor='black', density=True)

        # Add KDE curve
        try:
            sns.kdeplot(data=stats_df[stat], ax=ax,
                        color='darkred', linewidth=2)
        except:
            pass

        ax.set_title(
            f'Distribution of {stat.capitalize()} Values', fontweight='bold')
        ax.set_xlabel(f'{stat.capitalize()} Value')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)

        # Add statistics text
        mean_val = stats_df[stat].mean()
        std_val = stats_df[stat].std()
        ax.axvline(mean_val, color='red', linestyle='--',
                   alpha=0.8, label=f'Mean: {mean_val:.4f}')
        ax.text(0.05, 0.95, f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Increase space between subplots
    plt.tight_layout(pad=4.0, rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.07,
                        wspace=0.25, hspace=0.35)

    if save_plots:
        plt.savefig(output_dir / 'statistical_distributions_overview.png',
                    dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Create individual detailed plots
    for stat, color in zip(statistics, colors):
        plt.figure(figsize=(10, 6))

        # Box plot and histogram combination
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Histogram
        ax1.hist(stats_df[stat], bins=20, alpha=0.7,
                 color=color, edgecolor='black')
        ax1.set_title(f'Histogram of {stat.capitalize()} Values')
        ax1.set_xlabel(f'{stat.capitalize()} Value')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)

        # Box plot
        box_plot = ax2.boxplot(stats_df[stat], patch_artist=True)
        box_plot['boxes'][0].set_facecolor(color)
        ax2.set_title(f'Box Plot of {stat.capitalize()} Values')
        ax2.set_ylabel(f'{stat.capitalize()} Value')
        ax2.grid(True, alpha=0.3)

        plt.suptitle(
            f'Detailed Analysis of {stat.capitalize()} Distribution for "{column}"', fontweight='bold')
        plt.tight_layout()

        if save_plots:
            plt.savefig(output_dir / f'{stat}_distribution_detailed.png',
                        dpi=300, bbox_inches='tight')
        plt.close(fig)


def main():
    # Specify the folders and columns to process
    csv_folders = [r"1d-2005\train", "1d-2005\test", r"1d-2005\val"]
    columns = ['open', 'close', 'high', 'low', 'volume', 'macd', 'rsi', 'close_10_sma',
               'close_10_ema', 'adx', 'boll_ub', 'boll_lb', 'boll', 'kdjk', 'kdjd', 'kdjj', 'atr']
    norm_columns = [f'norm_{col}' for col in columns]
    columns += norm_columns

    for csv_folder in csv_folders:
        if not os.path.exists(csv_folder):
            print(f"Error: Folder '{csv_folder}' does not exist.")
            continue

        for column in columns:
            print(
                f"\nAnalyzing CSV files in: {csv_folder} for column: {column}")

            # Analyze all CSV files
            stats_df = analyze_csv_files(csv_folder, column)

            if stats_df is None or len(stats_df) == 0:
                print(
                    f"No valid data found to analyze for column '{column}' in folder '{csv_folder}'.")
                continue

            print(
                f"\nProcessed {len(stats_df)} CSV files successfully for column '{column}'.")
            print("\nSummary of statistical properties:")
            print(stats_df.describe())

            # Save the statistical summary
            output_dir = Path("statistical-plot") / Path(csv_folder) / column
            output_dir.mkdir(parents=True, exist_ok=True)
            stats_df.to_csv(
                output_dir / 'statistical_summary.csv', index=False)
            print(
                f"\nStatistical summary saved to '{output_dir / 'statistical_summary.csv'}'")

            # Create visualizations
            print("\nGenerating distribution plots...")
            plot_statistical_distributions(stats_df, column, output_dir)

            print("\nAnalysis complete! Check the generated plots and CSV summary.")


if __name__ == "__main__":
    main()
