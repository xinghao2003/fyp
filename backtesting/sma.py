import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import warnings
import os
import json
from datetime import datetime

# --- 1. Define the Baseline Strategy (SMA Crossover) ---
# This strategy is taken directly from the Backtesting.py user guide.
# It buys when a short-term moving average crosses above a long-term one
# and sells when the short-term average crosses below.


class SmaCross(Strategy):
    """
    A simple moving average (MA) crossover strategy.

    Parameters
    ----------
    n1 : int
        The lookback period for the shorter (faster) moving average.
    n2 : int
        The lookback period for the longer (slower) moving average.
    """
    # Define the two MA lags as class variables for optimization
    n1 = 10
    n2 = 20

    def init(self):
        """
        Called once before the backtest begins. Pre-computes indicators.
        """
        # A helper function to compute the Simple Moving Average
        def sma(values, n):
            return pd.Series(values).rolling(n).mean()

        # Pre-compute the two moving averages
        self.sma1 = self.I(sma, self.data.Close, self.n1)
        self.sma2 = self.I(sma, self.data.Close, self.n2)

    def next(self):
        """
        Called on each candlestick bar. Defines the trading logic.
        """
        # If the fast MA crosses above the slow MA, go long
        if crossover(self.sma1, self.sma2):
            self.position.close()  # Close any existing short position
            self.buy()

        # If the fast MA crosses below the slow MA, go short
        elif crossover(self.sma2, self.sma1):
            self.position.close()  # Close any existing long position
            self.sell()

# --- 2. Load and Prepare Your Data ---
# IMPORTANT:
# 1. Replace 'folder_path' with the actual path to your folder containing CSV files.
# 2. Your CSV files must have columns named: 'date', 'open', 'high', 'low', 'close', 'volume'.
# 3. The script will automatically format the data for the backtesting library.


def process_csv_file(csv_file_path, folder_name):
    """
    Process a single CSV file and run backtest on it.

    Parameters
    ----------
    csv_file_path : str
        Path to the CSV file to process
    folder_name : str
        Name of the source folder for organizing results

    Returns
    -------
    dict
        Dictionary containing file path and backtest results
    """
    try:
        print(f"\n{'='*60}")
        print(f"Processing: {csv_file_path}")
        print(f"{'='*60}")

        # Load the data from the CSV file
        data = pd.read_csv(csv_file_path)

        # Convert the 'date' column to datetime objects
        data['date'] = pd.to_datetime(data['date'])

        # Remove timezone information to avoid numpy datetime64 warnings
        if data['date'].dt.tz is not None:
            data['date'] = data['date'].dt.tz_localize(None)

        # Set the date as the index
        data.set_index('date', inplace=True)

        # Rename columns to the required format for backtesting.py (TitleCase)
        data.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)

        # Keep only the required columns for backtesting
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = data[required_columns]

        print("Data loaded and prepared successfully.")
        print("Data head:")
        print(data.head())

        # --- 3. Run the Backtest ---
        # Instantiate the Backtest object with your data and the baseline strategy
        bt = Backtest(data, SmaCross, cash=10000, commission=.002)

        # Run the backtest
        print("\nRunning backtest...")
        stats = bt.run()

        # --- 4. Print and Plot the Results ---
        print("\nBacktest Results:")
        print(stats)

        # You can also access individual trade details
        print("\nTrade Details:")
        print(stats['_trades'])

        # --- 5. Save Results ---
        strategy_name = bt._strategy.__name__ if hasattr(
            bt._strategy, '__name__') else str(bt._strategy)
        saved_files = save_backtest_results(
            stats, bt, csv_file_path, folder_name, strategy_name)

        return {
            'file_path': csv_file_path,
            'status': 'success',
            'stats': stats,
            'saved_files': saved_files
        }

    except Exception as e:
        print(f"Error processing {csv_file_path}: {e}")
        return {
            'file_path': csv_file_path,
            'status': 'error',
            'error': str(e),
            'saved_files': {}
        }


def save_backtest_results(stats, bt, file_path, folder_name, strategy_name):
    """
    Save backtest results in multiple formats.

    Parameters
    ----------
    stats : pd.Series
        Backtest results from bt.run()
    bt : Backtest
        Backtest instance for generating plots
    file_path : str
        Original CSV file path (used for naming output files)
    folder_name : str
        Name of the source folder (e.g., '1d-2015')
    strategy_name : str
        Name of the strategy (e.g., 'SmaCross')

    Returns
    -------
    dict
        Dictionary with paths to saved files
    """
    # Extract filename without extension for folder naming
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # Create nested directory structure: result/folder_path/filename/strategy_name/
    output_dir = os.path.join("result", folder_name, base_name, strategy_name)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = {}

    try:
        # 1. Save main statistics as CSV
        stats_csv_path = os.path.join(output_dir, f"stats_{timestamp}.csv")
        # Convert stats to DataFrame for better CSV formatting
        stats_df = pd.DataFrame(stats).T
        stats_df.to_csv(stats_csv_path)
        saved_files['stats_csv'] = stats_csv_path

        # 2. Save trades data as CSV (if available)
        if '_trades' in stats and not stats['_trades'].empty:
            trades_csv_path = os.path.join(
                output_dir, f"trades_{timestamp}.csv")
            stats['_trades'].to_csv(trades_csv_path)
            saved_files['trades_csv'] = trades_csv_path

        # 3. Save equity curve data as CSV (if available)
        if '_equity_curve' in stats:
            equity_csv_path = os.path.join(
                output_dir, f"equity_{timestamp}.csv")
            equity_df = pd.DataFrame(stats['_equity_curve'])
            equity_df.to_csv(equity_csv_path)
            saved_files['equity_csv'] = equity_csv_path

        # 4. Save complete results as pickle (preserves all data types)
        pickle_path = os.path.join(output_dir, f"complete_{timestamp}.pkl")
        stats.to_pickle(pickle_path)
        saved_files['pickle'] = pickle_path

        # 5. Save key metrics as JSON for easy reading
        json_path = os.path.join(output_dir, f"summary_{timestamp}.json")
        # Convert stats to dict, handling non-serializable objects
        json_data = {}
        for key, value in stats.items():
            if key.startswith('_'):
                continue  # Skip internal objects for JSON
            try:
                # Try to convert to JSON-serializable format
                if pd.isna(value):
                    json_data[key] = None
                elif isinstance(value, (int, float, str, bool)):
                    json_data[key] = value
                elif hasattr(value, 'isoformat'):  # datetime objects
                    json_data[key] = value.isoformat()
                else:
                    json_data[key] = str(value)
            except:
                json_data[key] = str(value)

        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        saved_files['json'] = json_path

        # 6. Generate and save plot
        print("\nGenerating plot...")
        plot_path = os.path.join(output_dir, f"plot_{timestamp}")
        bt.plot(filename=plot_path, open_browser=False)
        saved_files['plot'] = plot_path

        print(f"Results saved to {output_dir}/")
        for file_type, path in saved_files.items():
            print(f"  - {file_type}: {os.path.basename(path)}")

    except Exception as e:
        print(f"Error saving results: {e}")

    return saved_files


def save_summary_results(results, folder_name, strategy_name="strategy"):
    """
    Save a summary of all backtest results.

    Parameters
    ----------
    results : list
        List of result dictionaries from process_csv_file
    folder_name : str
        Name of the source folder (e.g., '1d-2015')
    strategy_name : str
        Name of the strategy (e.g., 'SmaCross')
    """
    try:
        # Create summary directory: result/folder_path/
        output_dir = os.path.join("result", folder_name)
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # List of all possible stats to include in the summary
        all_stats = [
            'Start', 'End', 'Duration', 'Exposure Time [%]', 'Equity Final [$]', 'Equity Peak [$]',
            'Commissions [$]', 'Return [%]', 'Buy & Hold Return [%]', 'Return (Ann.) [%]',
            'Volatility (Ann.) [%]', 'CAGR [%]', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
            'Alpha [%]', 'Beta', 'Max. Drawdown [%]', 'Avg. Drawdown [%]', 'Max. Drawdown Duration',
            'Avg. Drawdown Duration', '# Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]',
            'Avg. Trade [%]', 'Max. Trade Duration', 'Avg. Trade Duration', 'Profit Factor',
            'Expectancy [%]', 'SQN', 'Kelly Criterion'
        ]

        summary_data = []
        for result in results:
            row = {'File': os.path.basename(result['file_path'])}
            if result['status'] == 'success':
                stats = result['stats']
                for stat in all_stats:
                    row[stat] = stats.get(stat, None)
                row['Status'] = 'Success'
            else:
                for stat in all_stats:
                    row[stat] = None
                row['Status'] = f"Error: {result['error']}"
            summary_data.append(row)

        # Save summary as CSV with strategy name in the file name
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(
            output_dir, f"backtest_summary_{strategy_name}_{timestamp}.csv")
        summary_df.to_csv(summary_path, index=False)

        print(f"\nSummary saved to: {summary_path}")
        return summary_path

    except Exception as e:
        print(f"Error saving summary: {e}")
        return None


def find_csv_files(folder_path):
    """
    Find all CSV files in a folder and its subfolders.

    Parameters
    ----------
    folder_path : str
        Path to the folder to search

    Returns
    -------
    list
        List of paths to CSV files
    """
    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files


# <--- CHANGE THIS TO YOUR FOLDER PATH
folder_path = r'1d-2015'

try:
    # Find all CSV files in the folder and subfolders
    csv_files = find_csv_files(folder_path)

    if not csv_files:
        print(f"No CSV files found in '{folder_path}' and its subfolders.")
    else:
        print(f"Found {len(csv_files)} CSV files to process:")
        for file in csv_files:
            print(f"  - {file}")

        # Extract folder name for organizing results
        folder_name = os.path.basename(os.path.normpath(folder_path))

        # Process each CSV file
        results = []
        for csv_file in csv_files:
            result = process_csv_file(csv_file, folder_name)
            results.append(result)

        # Save summary of all results
        strategy_name = SmaCross.__name__
        save_summary_results(results, folder_name, strategy_name)

        # Summary of results
        print(f"\n{'='*60}")
        print("PROCESSING SUMMARY")
        print(f"{'='*60}")
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'error')

        print(f"Total files processed: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

        if failed > 0:
            print("\nFailed files:")
            for result in results:
                if result['status'] == 'error':
                    print(f"  - {result['file_path']}: {result['error']}")

except FileNotFoundError:
    print(f"Error: The folder '{folder_path}' was not found.")
    print("Please make sure the folder path is correct.")
except Exception as e:
    print(f"An error occurred: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
