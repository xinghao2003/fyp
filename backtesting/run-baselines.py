from baselines import *
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import warnings
import os
import json
from datetime import datetime
import logging

# --- Logger Setup ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(
    log_dir, f"run_baselines_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import all baseline strategies

# --- Get all strategy classes from baselines ---


def get_all_strategies():
    """
    Get all strategy classes from the baselines module.

    Returns
    -------
    dict
        Dictionary mapping strategy names to strategy classes
    """
    import baselines
    strategies = {}

    for name in dir(baselines):
        obj = getattr(baselines, name)
        if (isinstance(obj, type) and
            issubclass(obj, Strategy) and
                obj != Strategy):
            strategies[name] = obj

    logger.info(
        f"Discovered {len(strategies)} strategies: {list(strategies.keys())}")
    return strategies

# --- 2. Load and Prepare Your Data ---
# IMPORTANT:
# 1. Replace 'folder_path' with the actual path to your folder containing CSV files.
# 2. Your CSV files must have columns named: 'date', 'open', 'high', 'low', 'close', 'volume'.
# 3. The script will automatically format the data for the backtesting library.


def process_csv_file(csv_file_path, folder_name):
    """
    Process a single CSV file and run backtest on it with all strategies.

    Parameters
    ----------
    csv_file_path : str
        Path to the CSV file to process
    folder_name : str
        Name of the source folder for organizing results

    Returns
    -------
    dict
        Dictionary containing file path and backtest results for all strategies
    """
    try:
        logger.info(f"{'='*60}")
        logger.info(f"Processing: {csv_file_path}")
        logger.info(f"{'='*60}")

        # Load the data from the CSV file
        data = pd.read_csv(csv_file_path)
        logger.info("CSV loaded successfully.")

        # Convert the 'date' column to datetime objects
        data['date'] = pd.to_datetime(data['date'])
        logger.info("Converted 'date' column to datetime.")

        # Remove timezone information to avoid numpy datetime64 warnings
        if data['date'].dt.tz is not None:
            data['date'] = data['date'].dt.tz_localize(None)
            logger.info("Removed timezone info from 'date' column.")

        # Set the date as the index
        data.set_index('date', inplace=True)
        logger.info("Set 'date' as index.")

        # Rename columns to the required format for backtesting.py (TitleCase)
        data.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        logger.info("Renamed columns for backtesting.py.")

        # Keep only the required columns for backtesting
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = data[required_columns]
        logger.info("Filtered required columns.")

        logger.debug(f"Data head:\n{data.head()}")

        # Get all strategies
        strategies = get_all_strategies()
        strategy_results = {}

        # --- Run backtest for each strategy ---
        for strategy_name, strategy_class in strategies.items():
            try:
                logger.info(f"Running backtest for strategy: {strategy_name}")

                # Instantiate the Backtest object with your data and the strategy
                bt = Backtest(data, strategy_class,
                              cash=100000, commission=.002)

                # Run the backtest
                stats = bt.run()

                logger.info(f"Backtest completed for {strategy_name}")
                logger.info(f"Final Return: {stats.get('Return [%]', 'N/A')}")

                # Save results for this strategy
                saved_files = save_backtest_results(
                    stats, bt, csv_file_path, folder_name, strategy_name)

                strategy_results[strategy_name] = {
                    'status': 'success',
                    'stats': stats,
                    'saved_files': saved_files
                }

            except Exception as strategy_error:
                logger.error(
                    f"Error running strategy {strategy_name}: {strategy_error}", exc_info=True)
                strategy_results[strategy_name] = {
                    'status': 'error',
                    'error': str(strategy_error),
                    'saved_files': {}
                }

        return {
            'file_path': csv_file_path,
            'status': 'success',
            'strategy_results': strategy_results
        }

    except Exception as e:
        logger.error(f"Error processing {csv_file_path}: {e}", exc_info=True)
        return {
            'file_path': csv_file_path,
            'status': 'error',
            'error': str(e),
            'strategy_results': {}
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
    output_dir = os.path.join("result", "baseline",
                              folder_name, base_name, strategy_name)
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
        logger.info(f"Saved stats CSV: {stats_csv_path}")

        # 2. Save trades data as CSV (if available)
        if '_trades' in stats and not stats['_trades'].empty:
            trades_csv_path = os.path.join(
                output_dir, f"trades_{timestamp}.csv")
            stats['_trades'].to_csv(trades_csv_path)
            saved_files['trades_csv'] = trades_csv_path
            logger.info(f"Saved trades CSV: {trades_csv_path}")

        # 3. Save equity curve data as CSV (if available)
        if '_equity_curve' in stats:
            equity_csv_path = os.path.join(
                output_dir, f"equity_{timestamp}.csv")
            equity_df = pd.DataFrame(stats['_equity_curve'])
            equity_df.to_csv(equity_csv_path)
            saved_files['equity_csv'] = equity_csv_path
            logger.info(f"Saved equity curve CSV: {equity_csv_path}")

        # 4. Save complete results as pickle (preserves all data types)
        pickle_path = os.path.join(output_dir, f"complete_{timestamp}.pkl")
        stats.to_pickle(pickle_path)
        saved_files['pickle'] = pickle_path
        logger.info(f"Saved pickle: {pickle_path}")

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
        logger.info(f"Saved summary JSON: {json_path}")

        # 6. Generate and save plot
        logger.info("Generating plot...")
        plot_path = os.path.join(output_dir, f"plot_{timestamp}")
        bt.plot(filename=plot_path, open_browser=False)
        saved_files['plot'] = plot_path
        logger.info(f"Saved plot: {plot_path}")

        logger.info(f"Results saved to {output_dir}/")
        for file_type, path in saved_files.items():
            logger.info(f"  - {file_type}: {os.path.basename(path)}")

    except Exception as e:
        logger.error(f"Error saving results: {e}", exc_info=True)

    return saved_files


def save_summary_results(results, folder_name):
    """
    Save a summary of all backtest results for all strategies.

    Parameters
    ----------
    results : list
        List of result dictionaries from process_csv_file
    folder_name : str
        Name of the source folder (e.g., '1d-2015')
    """
    try:
        # Create summary directory: result/folder_path/
        output_dir = os.path.join("result", "baseline", folder_name)
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

        # Collect all strategies used
        all_strategies = set()
        for result in results:
            if result['status'] == 'success':
                all_strategies.update(result['strategy_results'].keys())

        # Create summary for each strategy
        for strategy_name in all_strategies:
            summary_data = []
            for result in results:
                row = {'File': os.path.basename(result['file_path'])}

                if (result['status'] == 'success' and
                        strategy_name in result['strategy_results']):

                    strategy_result = result['strategy_results'][strategy_name]
                    if strategy_result['status'] == 'success':
                        stats = strategy_result['stats']
                        for stat in all_stats:
                            row[stat] = stats.get(stat, None)
                        row['Status'] = 'Success'
                    else:
                        for stat in all_stats:
                            row[stat] = None
                        row['Status'] = f"Error: {strategy_result['error']}"
                else:
                    for stat in all_stats:
                        row[stat] = None
                    row['Status'] = 'File Error'

                summary_data.append(row)

            # Save summary as CSV for this strategy
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(
                output_dir, f"backtest_summary_{strategy_name}_{timestamp}.csv")
            summary_df.to_csv(summary_path, index=False)
            logger.info(
                f"Summary for {strategy_name} saved to: {summary_path}")

        # Create overall summary comparing strategies
        create_strategy_comparison_summary(
            results, folder_name, timestamp, all_stats)

    except Exception as e:
        logger.error(f"Error saving summary: {e}", exc_info=True)
        return None


def create_strategy_comparison_summary(results, folder_name, timestamp, all_stats):
    """Create a comparison summary across all strategies and files."""
    try:
        output_dir = os.path.join("result", "baseline", folder_name)

        comparison_data = []
        for result in results:
            if result['status'] == 'success':
                file_name = os.path.basename(result['file_path'])
                for strategy_name, strategy_result in result['strategy_results'].items():
                    row = {
                        'File': file_name,
                        'Strategy': strategy_name
                    }

                    if strategy_result['status'] == 'success':
                        stats = strategy_result['stats']
                        for stat in all_stats:
                            row[stat] = stats.get(stat, None)
                        row['Status'] = 'Success'
                    else:
                        for stat in all_stats:
                            row[stat] = None
                        row['Status'] = f"Error: {strategy_result['error']}"

                    comparison_data.append(row)

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_path = os.path.join(
                output_dir, f"strategy_comparison_{timestamp}.csv")
            comparison_df.to_csv(comparison_path, index=False)
            logger.info(f"Strategy comparison saved to: {comparison_path}")

    except Exception as e:
        logger.error(f"Error creating strategy comparison: {e}", exc_info=True)


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
    logger.info(f"Found {len(csv_files)} CSV files in '{folder_path}'.")
    return csv_files


# <--- CHANGE THIS TO YOUR FOLDER PATH
folder_path = r'1d-2015'

try:
    logger.info(f"Searching for CSV files in: {folder_path}")
    csv_files = find_csv_files(folder_path)

    if not csv_files:
        logger.warning(
            f"No CSV files found in '{folder_path}' and its subfolders.")
        print(f"No CSV files found in '{folder_path}' and its subfolders.")
    else:
        strategies = get_all_strategies()
        logger.info(
            f"Found {len(strategies)} strategies to test: {list(strategies.keys())}")
        print(f"Found {len(strategies)} strategies to test:")
        for strategy_name in strategies.keys():
            print(f"  - {strategy_name}")

        print(f"\nFound {len(csv_files)} CSV files to process:")
        for file in csv_files:
            print(f"  - {file}")

        folder_name = os.path.basename(os.path.normpath(folder_path))

        results = []
        for csv_file in csv_files:
            logger.info(f"Processing file: {csv_file}")
            result = process_csv_file(csv_file, folder_name)
            results.append(result)

        save_summary_results(results, folder_name)

        logger.info(f"{'='*60}")
        logger.info("PROCESSING SUMMARY")
        logger.info(f"{'='*60}")

        print(f"\n{'='*60}")
        print("PROCESSING SUMMARY")
        print(f"{'='*60}")

        successful_files = sum(1 for r in results if r['status'] == 'success')
        failed_files = sum(1 for r in results if r['status'] == 'error')

        logger.info(f"Total files processed: {len(results)}")
        logger.info(f"Successful files: {successful_files}")
        logger.info(f"Failed files: {failed_files}")

        print(f"Total files processed: {len(results)}")
        print(f"Successful files: {successful_files}")
        print(f"Failed files: {failed_files}")

        # Count strategy successes/failures
        strategy_stats = {}
        for result in results:
            if result['status'] == 'success':
                for strategy_name, strategy_result in result['strategy_results'].items():
                    if strategy_name not in strategy_stats:
                        strategy_stats[strategy_name] = {
                            'success': 0, 'error': 0}

                    if strategy_result['status'] == 'success':
                        strategy_stats[strategy_name]['success'] += 1
                    else:
                        strategy_stats[strategy_name]['error'] += 1

        logger.info("Strategy Success/Failure Summary:")
        print(f"\nStrategy Success/Failure Summary:")
        for strategy_name, stats in strategy_stats.items():
            total = stats['success'] + stats['error']
            logger.info(
                f"  {strategy_name}: {stats['success']}/{total} successful")
            print(f"  {strategy_name}: {stats['success']}/{total} successful")

        if failed_files > 0:
            logger.warning("Failed files:")
            print("\nFailed files:")
            for result in results:
                if result['status'] == 'error':
                    logger.warning(
                        f"  - {result['file_path']}: {result['error']}")
                    print(f"  - {result['file_path']}: {result['error']}")

except FileNotFoundError:
    logger.error(f"Error: The folder '{folder_path}' was not found.")
    print(f"Error: The folder '{folder_path}' was not found.")
    print("Please make sure the folder path is correct.")
except Exception as e:
    logger.error(f"An error occurred: {e}", exc_info=True)
    print(f"An error occurred: {e}")
