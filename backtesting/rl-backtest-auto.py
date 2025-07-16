# File: fyp-code/backtesting/rl-backtest-auto.py

import argparse
import json
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from backtesting import Backtest, Strategy
from sb3_contrib import RecurrentPPO

# --- Logger Setup ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(
    log_dir, f"rl_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RLStrategy(Strategy):
    """
    A backtesting.py Strategy that uses a trained Stable-Baselines3
    RecurrentPPO agent to make trading decisions.
    """
    # --- Parameters passed from Backtest instance ---
    model_path = None
    params_path = None

    # This must map the agent's discrete output to the position values.
    # Based on your environment (`positions=[-1, 0, 1]`), the mapping is:
    # action 0 -> position -1 (SHORT)
    # action 1 -> position 0  (OUT)
    # action 2 -> position 1  (LONG)
    positions_map = {0: -1, 1: 0, 2: 1}

    def init(self):
        """
        Called once at the start of the backtest.
        This is where we load the model and prepare the data.
        """
        logger.info("=== Initializing RL Strategy ===")
        logger.debug(f"Model path: {self.model_path}")
        logger.debug(f"Params path: {self.params_path}")

        # 1. Load the trained RL model
        if not self.model_path or not os.path.exists(self.model_path):
            logger.error(f"Model file not found at: {self.model_path}")
            raise FileNotFoundError(
                f"Model file not found at: {self.model_path}")

        logger.info(f"Loading model from: {self.model_path}")
        try:
            self.model = RecurrentPPO.load(self.model_path, device="cpu")
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # 2. Load the parameters and feature configuration from training
        if not self.params_path or not os.path.exists(self.params_path):
            logger.error(f"Params JSON file not found at: {self.params_path}")
            raise FileNotFoundError(
                f"Params JSON file not found at: {self.params_path}")

        logger.info(f"Loading params from: {self.params_path}")
        try:
            with open(self.params_path, 'r') as f:
                params = json.load(f)
            logger.debug(f"Loaded params: {json.dumps(params, indent=2)}")
        except Exception as e:
            logger.error(f"Failed to load params: {e}")
            raise

        self.feature_config = params['feature_config']
        self.windows = params['hyperparameters']['windows']
        self.static_feature_columns = []

        logger.info(f"Agent trained with window size: {self.windows}")
        logger.debug(f"Feature config: {self.feature_config}")

        # 3. Recreate the 'preprocess' logic to generate feature columns
        data_df = self.data.df.copy()
        logger.info(f"Original data shape: {data_df.shape}")
        logger.debug(f"Data columns: {list(data_df.columns)}")

        logger.info("Applying preprocessing to generate features...")

        feature_map = {
            'use_volume': 'norm_volume', 'use_high': 'norm_high', 'use_low': 'norm_low',
            'use_open': 'norm_open', 'use_macd': 'norm_macd', 'use_rsi': 'norm_rsi',
            'use_sma': 'norm_close_10_sma', 'use_ema': 'norm_close_10_ema', 'use_adx': 'norm_adx',
            'use_bb_upper': 'norm_boll_ub', 'use_bb_lower': 'norm_boll_lb', 'use_bb_middle': 'norm_boll',
            'use_stoch_k': 'norm_kdjk', 'use_stoch_d': 'norm_kdjd', 'use_stoch_j': 'norm_kdjj',
            'use_atr': 'norm_atr'
        }

        # Always include the base price feature
        if 'norm_close' in data_df.columns:
            data_df["feature_close"] = data_df["norm_close"]
            self.static_feature_columns.append("feature_close")
            logger.debug("Added feature_close")
        else:
            logger.warning("norm_close column not found in data")

        for config_key, df_col_name in feature_map.items():
            if self.feature_config.get(config_key, False):
                if df_col_name in data_df.columns:
                    feature_name = f"feature_{config_key.replace('use_', '')}"
                    data_df[feature_name] = data_df[df_col_name]
                    self.static_feature_columns.append(feature_name)
                    logger.debug(f"Added {feature_name} from {df_col_name}")
                else:
                    logger.warning(
                        f"Column {df_col_name} not found for feature {config_key}")

        # Store the preprocessed data with all static features
        self.processed_data = data_df[self.static_feature_columns].values
        logger.info(
            f"Generated {len(self.static_feature_columns)} static features: {self.static_feature_columns}")
        logger.debug(f"Processed data shape: {self.processed_data.shape}")

        # 4. Initialize recurrent (LSTM) states for the model
        self.lstm_states = None
        self.episode_starts = np.array([True])

        # Initialize step counter for detailed logging
        self.step_count = 0

        # Initialize CSV logging lists
        self.decisions_log = []
        self.trades_log = []

        logger.info("RL Strategy initialization completed successfully")

    def next(self):
        """
        Called at each candlestick bar (each step) of the backtest.
        """
        self.step_count += 1
        current_index = len(self.data) - 1
        current_time = self.data.index[-1] if hasattr(
            self.data.index[-1], 'strftime') else f"Step {current_index}"

        logger.debug(
            f"\n--- Step {self.step_count} | Index {current_index} | Time {current_time} ---")
        logger.debug(f"Current price: {self.data.Close[-1]:.4f}")
        logger.debug(f"Current equity: {self.equity:.2f}")
        logger.debug(f"Current position size: {self.position.size}")

        # 1. Ensure we have enough data for the full window
        if current_index < self.windows - 1:
            logger.debug(
                f"Insufficient data: need {self.windows}, have {current_index + 1}. Skipping.")
            return

        # 2. Assemble the observation for the RL model
        logger.debug("Assembling observation for RL model...")

        # Part A: Get the static features from the pre-calculated array
        static_obs_data = self.processed_data[current_index -
                                              self.windows + 1: current_index + 1]
        logger.debug(f"Static observation shape: {static_obs_data.shape}")
        logger.debug(
            f"Static features range: [{static_obs_data.min():.4f}, {static_obs_data.max():.4f}]")

        # Part B: Simulate the two dynamic features from the training environment

        # Dynamic feature 1: `dynamic_feature_last_position_taken`
        if self.position.is_long:
            last_pos_value = 1
            pos_description = "LONG"
        elif self.position.is_short:
            last_pos_value = -1
            pos_description = "SHORT"
        else:
            last_pos_value = 0
            pos_description = "FLAT"

        last_action_index = next(
            (k for k, v in self.positions_map.items() if v == last_pos_value), 1)
        logger.debug(
            f"Current position: {pos_description} (value: {last_pos_value}, action_index: {last_action_index})")

        # Dynamic feature 2: `dynamic_feature_real_position` (leverage)
        if self.equity > 0:
            current_price = self.data.Close[-1]
            real_position = (self.position.size * current_price) / self.equity
        else:
            real_position = 0.0

        logger.debug(f"Real position (leverage): {real_position:.4f}")

        dynamic_features = np.array([last_action_index, real_position])
        dynamic_obs_data = np.tile(dynamic_features, (self.windows, 1))
        logger.debug(f"Dynamic observation shape: {dynamic_obs_data.shape}")

        # Part C: Combine static and dynamic features to form the final observation
        observation = np.hstack([static_obs_data, dynamic_obs_data])
        observation = np.expand_dims(observation, axis=0)

        logger.debug(f"Final observation shape: {observation.shape}")
        logger.debug(
            f"Observation stats - min: {observation.min():.4f}, max: {observation.max():.4f}, mean: {observation.mean():.4f}")

        # 3. Get the action from the RL model
        logger.debug("Querying RL model for action...")
        try:
            action, self.lstm_states = self.model.predict(
                observation,
                state=self.lstm_states,
                episode_start=self.episode_starts,
                deterministic=True
            )

            action_value = action.item()
            target_position = self.positions_map[action_value]

            # Log decision to CSV data (removed from main log to reduce spam)
            decision_record = {
                'timestamp': current_time,
                'step': self.step_count,
                'price': self.data.Close[-1],
                'equity': self.equity,
                'current_position': pos_description,
                'current_position_size': self.position.size,
                'action': action_value,
                'target_position': target_position,
                'real_position_leverage': real_position,
                'observation_min': observation.min(),
                'observation_max': observation.max(),
                'observation_mean': observation.mean()
            }
            self.decisions_log.append(decision_record)

            logger.debug(
                f"LSTM states updated: {self.lstm_states is not None}")

        except Exception as e:
            logger.error(f"Error during model prediction: {e}")
            logger.debug(f"Observation that caused error: {observation}")
            raise

        # It's no longer the start after the first step
        self.episode_starts = np.array([False])

        # 4. Translate the action to a target position and execute trades
        position_before = "LONG" if self.position.is_long else "SHORT" if self.position.is_short else "FLAT"
        position_size_before = self.position.size
        equity_before = self.equity

        trade_executed = False
        trade_action = "NONE"

        if target_position == 1:      # Target: LONG
            if not self.position.is_long:
                logger.info(
                    f"EXECUTING BUY: {position_before} -> LONG at price {self.data.Close[-1]:.4f}")
                self.buy()
                trade_executed = True
                trade_action = "BUY"
            else:
                logger.debug("Already LONG, no action needed")
        elif target_position == -1:   # Target: SHORT
            if not self.position.is_short:
                logger.info(
                    f"EXECUTING SELL: {position_before} -> SHORT at price {self.data.Close[-1]:.4f}")
                self.sell()
                trade_executed = True
                trade_action = "SELL"
            else:
                logger.debug("Already SHORT, no action needed")
        elif target_position == 0:    # Target: OUT/FLAT
            if self.position.size != 0:
                logger.info(
                    f"EXECUTING CLOSE: {position_before} -> FLAT at price {self.data.Close[-1]:.4f}")
                self.position.close()
                trade_executed = True
                trade_action = "CLOSE"
            else:
                logger.debug("Already FLAT, no action needed")

        position_after = "LONG" if self.position.is_long else "SHORT" if self.position.is_short else "FLAT"
        position_size_after = self.position.size
        equity_after = self.equity

        # Log trade execution to CSV data
        if trade_executed:
            trade_record = {
                'timestamp': current_time,
                'step': self.step_count,
                'price': self.data.Close[-1],
                'trade_action': trade_action,
                'position_before': position_before,
                'position_after': position_after,
                'position_size_before': position_size_before,
                'position_size_after': position_size_after,
                'equity_before': equity_before,
                'equity_after': equity_after,
                'model_action': action_value,
                'target_position': target_position
            }
            self.trades_log.append(trade_record)

            logger.info(
                f"Position changed: {position_before} -> {position_after}")

        logger.debug(
            f"Step {self.step_count} completed. New position size: {self.position.size}, Equity: {self.equity:.2f}")


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


def process_csv_file(csv_file_path, folder_name, model_path, params_path, model_name):
    """
    Process a single CSV file and run RL backtest on it.

    Parameters
    ----------
    csv_file_path : str
        Path to the CSV file to process
    folder_name : str
        Name of the source folder for organizing results
    model_path : str
        Path to the trained model file
    params_path : str
        Path to the params JSON file
    model_name : str
        Name of the model for organizing results

    Returns
    -------
    dict
        Dictionary containing file path and backtest results
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
        original_columns = list(data.columns)
        data.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        }, inplace=True, errors='ignore')

        renamed_columns = list(data.columns)
        if original_columns != renamed_columns:
            logger.debug(f"Renamed columns for backtesting.py compatibility")

        logger.debug(f"Data head:\n{data.head()}")

        # Run RL backtest
        logger.info(f"Running RL backtest with model: {model_name}")

        # Instantiate the Backtest object
        bt = Backtest(data, RLStrategy, cash=100000, commission=.002)

        # Run the backtest
        stats = bt.run(model_path=model_path, params_path=params_path)

        logger.info(f"Backtest completed for {model_name}")
        logger.info(f"Final Return: {stats.get('Return [%]', 'N/A')}")

        # Save results
        saved_files = save_backtest_results(
            stats, bt, csv_file_path, folder_name, model_name)

        return {
            'file_path': csv_file_path,
            'status': 'success',
            'stats': stats,
            'saved_files': saved_files
        }

    except Exception as e:
        logger.error(f"Error processing {csv_file_path}: {e}", exc_info=True)
        return {
            'file_path': csv_file_path,
            'status': 'error',
            'error': str(e),
            'saved_files': {}
        }


def save_backtest_results(stats, bt, file_path, folder_name, model_name):
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
    model_name : str
        Name of the model (e.g., 'best_model')

    Returns
    -------
    dict
        Dictionary with paths to saved files
    """
    # Extract filename without extension for folder naming
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(
        "result", "rl", folder_name, base_name, model_name)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = {}

    try:
        # Get the strategy instance to access decision and trade logs
        strategy_instance = bt._strategy

        # 1. Save main statistics as CSV
        stats_csv_path = os.path.join(output_dir, f"stats_{timestamp}.csv")
        # Convert stats to DataFrame for better CSV formatting
        stats_df = pd.DataFrame(stats).T
        stats_df.to_csv(stats_csv_path)
        saved_files['stats_csv'] = stats_csv_path
        logger.info(f"Saved stats CSV: {stats_csv_path}")

        # 2. Save model decisions log as CSV
        try:
            if hasattr(strategy_instance, 'decisions_log') and strategy_instance.decisions_log:
                decisions_csv_path = os.path.join(
                    output_dir, f"decisions_{timestamp}.csv")
                decisions_df = pd.DataFrame(strategy_instance.decisions_log)
                decisions_df.to_csv(decisions_csv_path, index=False)
                saved_files['decisions_csv'] = decisions_csv_path
                logger.info(f"Saved decisions CSV: {decisions_csv_path}")
                logger.info(
                    f"Total decisions logged: {len(strategy_instance.decisions_log)}")
        except Exception as e:
            logger.warning(f"Could not save decisions CSV: {e}")

        # 3. Save trade executions log as CSV
        try:
            if hasattr(strategy_instance, 'trades_log') and strategy_instance.trades_log:
                trades_exec_csv_path = os.path.join(
                    output_dir, f"trade_executions_{timestamp}.csv")
                trades_exec_df = pd.DataFrame(strategy_instance.trades_log)
                trades_exec_df.to_csv(trades_exec_csv_path, index=False)
                saved_files['trade_executions_csv'] = trades_exec_csv_path
                logger.info(
                    f"Saved trade executions CSV: {trades_exec_csv_path}")
                logger.info(
                    f"Total trade executions logged: {len(strategy_instance.trades_log)}")
        except Exception as e:
            logger.warning(f"Could not save trade executions CSV: {e}")

        # 4. Save backtesting.py trades data as CSV (if available)
        try:
            if '_trades' in stats and not stats['_trades'].empty:
                trades_csv_path = os.path.join(
                    output_dir, f"backtest_trades_{timestamp}.csv")
                stats['_trades'].to_csv(trades_csv_path)
                saved_files['backtest_trades_csv'] = trades_csv_path
                logger.info(f"Saved backtest trades CSV: {trades_csv_path}")
        except Exception as e:
            logger.warning(f"Could not save backtest trades CSV: {e}")

        # 5. Save equity curve data as CSV (if available)
        try:
            if '_equity_curve' in stats:
                equity_csv_path = os.path.join(
                    output_dir, f"equity_{timestamp}.csv")
                equity_df = pd.DataFrame(stats['_equity_curve'])
                equity_df.to_csv(equity_csv_path)
                saved_files['equity_csv'] = equity_csv_path
                logger.info(f"Saved equity curve CSV: {equity_csv_path}")
        except Exception as e:
            logger.warning(f"Could not save equity curve CSV: {e}")

        # 6. Save key metrics as JSON for easy reading
        try:
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
        except Exception as e:
            logger.warning(f"Could not save summary JSON: {e}")

        # 7. Generate and save plot
        try:
            logger.info("Generating plot...")
            plot_path = os.path.join(output_dir, f"plot_{timestamp}")
            bt.plot(filename=plot_path, open_browser=False)
            saved_files['plot'] = plot_path
            logger.info(f"Saved plot: {plot_path}")
        except Exception as e:
            logger.warning(f"Could not save plot: {e}")

        logger.info(f"Results saved to {output_dir}/")
        for file_type, path in saved_files.items():
            logger.info(f"  - {file_type}: {os.path.basename(path)}")

    except Exception as e:
        logger.error(f"Error saving results: {e}", exc_info=True)

    return saved_files


def save_summary_results(results, folder_name, model_name):
    """
    Save a summary of all backtest results for the RL model.

    Parameters
    ----------
    results : list
        List of result dictionaries from process_csv_file
    folder_name : str
        Name of the source folder (e.g., '1d-2015')
    model_name : str
        Name of the model used
    """
    try:
        # Create summary directory: result/rl/folder_name/
        output_dir = os.path.join("result", "rl", folder_name)
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

        # Create summary data
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

        # Save summary as CSV
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(
            output_dir, f"rl_backtest_summary_{model_name}_{timestamp}.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary saved to: {summary_path}")

        # Create overall statistics summary
        create_performance_summary(results, folder_name, model_name, timestamp)

    except Exception as e:
        logger.error(f"Error saving summary: {e}", exc_info=True)
        return None


def create_performance_summary(results, folder_name, model_name, timestamp):
    """Create a performance summary with key metrics."""
    try:
        output_dir = os.path.join("result", "rl", folder_name)

        successful_results = [r for r in results if r['status'] == 'success']

        if not successful_results:
            logger.warning("No successful results to summarize")
            return

        # Calculate aggregate statistics
        returns = [r['stats'].get('Return [%]', 0) for r in successful_results]
        sharpe_ratios = [r['stats'].get(
            'Sharpe Ratio', 0) for r in successful_results if r['stats'].get('Sharpe Ratio') is not None]
        max_drawdowns = [r['stats'].get('Max. Drawdown [%]', 0)
                         for r in successful_results]
        win_rates = [r['stats'].get('Win Rate [%]', 0) for r in successful_results if r['stats'].get(
            'Win Rate [%]') is not None]

        performance_summary = {
            'Model': model_name,
            'Total_Files_Processed': len(results),
            'Successful_Backtests': len(successful_results),
            'Failed_Backtests': len(results) - len(successful_results),
            'Avg_Return_Percent': np.mean(returns) if returns else 0,
            'Median_Return_Percent': np.median(returns) if returns else 0,
            'Best_Return_Percent': max(returns) if returns else 0,
            'Worst_Return_Percent': min(returns) if returns else 0,
            'Avg_Sharpe_Ratio': np.mean(sharpe_ratios) if sharpe_ratios else None,
            'Avg_Max_Drawdown_Percent': np.mean(max_drawdowns) if max_drawdowns else 0,
            'Avg_Win_Rate_Percent': np.mean(win_rates) if win_rates else None,
            'Timestamp': timestamp
        }

        # Save performance summary
        perf_summary_path = os.path.join(
            output_dir, f"performance_summary_{model_name}_{timestamp}.json")

        with open(perf_summary_path, 'w') as f:
            json.dump(performance_summary, f, indent=2)

        logger.info(f"Performance summary saved to: {perf_summary_path}")

        # Log key performance metrics
        logger.info(f"=== Performance Summary for {model_name} ===")
        logger.info(
            f"Successful backtests: {len(successful_results)}/{len(results)}")
        logger.info(
            f"Average return: {performance_summary['Avg_Return_Percent']:.2f}%")
        logger.info(
            f"Best return: {performance_summary['Best_Return_Percent']:.2f}%")
        logger.info(
            f"Worst return: {performance_summary['Worst_Return_Percent']:.2f}%")
        if performance_summary['Avg_Sharpe_Ratio']:
            logger.info(
                f"Average Sharpe ratio: {performance_summary['Avg_Sharpe_Ratio']:.3f}")

    except Exception as e:
        logger.error(f"Error creating performance summary: {e}", exc_info=True)


def run_batch_backtest():
    parser = argparse.ArgumentParser(
        description="Backtest a trained RL agent on multiple files in a folder.")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained .zip model file (e.g., best_model.zip).")
    parser.add_argument("--params", type=str, required=True,
                        help="Path to the best_params.json file for the model.")
    parser.add_argument("--data_folder", type=str, required=True,
                        help="Path to the folder containing CSV files for backtesting.")

    args = parser.parse_args()

    logger.info("=== Starting RL Agent Batch Backtest ===")
    logger.info(f"Model: {args.model}")
    logger.info(f"Params: {args.params}")
    logger.info(f"Data folder: {args.data_folder}")

    # Validate input files
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        raise FileNotFoundError(f"Model file not found: {args.model}")

    if not os.path.exists(args.params):
        logger.error(f"Params file not found: {args.params}")
        raise FileNotFoundError(f"Params file not found: {args.params}")

    if not os.path.exists(args.data_folder):
        logger.error(f"Data folder not found: {args.data_folder}")
        raise FileNotFoundError(f"Data folder not found: {args.data_folder}")

    # Find all CSV files
    logger.info(f"Searching for CSV files in: {args.data_folder}")
    csv_files = find_csv_files(args.data_folder)

    if not csv_files:
        logger.warning(
            f"No CSV files found in '{args.data_folder}' and its subfolders.")
        print(
            f"No CSV files found in '{args.data_folder}' and its subfolders.")
        return

    print(f"\nFound {len(csv_files)} CSV files to process:")
    for file in csv_files:
        print(f"  - {file}")

    # Extract folder name and model name for organizing results
    folder_name = os.path.basename(os.path.normpath(args.data_folder))
    model_name = os.path.splitext(os.path.basename(args.model))[0]

    logger.info(f"Processing {len(csv_files)} files with model: {model_name}")

    results = []
    for i, csv_file in enumerate(csv_files, 1):
        logger.info(f"Processing file {i}/{len(csv_files)}: {csv_file}")
        result = process_csv_file(
            csv_file, folder_name, args.model, args.params, model_name)
        results.append(result)

    # Save summary results
    save_summary_results(results, folder_name, model_name)

    # Print final summary
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

    if failed_files > 0:
        logger.warning("Failed files:")
        print("\nFailed files:")
        for result in results:
            if result['status'] == 'error':
                logger.warning(f"  - {result['file_path']}: {result['error']}")
                print(f"  - {result['file_path']}: {result['error']}")

    logger.info("=== RL Batch Backtest Session Completed ===")


def run_single_backtest():
    """Run backtest on a single file (original functionality)"""
    parser = argparse.ArgumentParser(
        description="Backtest a trained RL agent on a single file.")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained .zip model file (e.g., best_model.zip).")
    parser.add_argument("--params", type=str, required=True,
                        help="Path to the best_params.json file for the model.")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to the .csv data file for backtesting (e.g., from the test set).")

    args = parser.parse_args()

    logger.info("=== Starting RL Agent Single Backtest ===")
    logger.info(f"Model: {args.model}")
    logger.info(f"Params: {args.params}")
    logger.info(f"Data: {args.data}")

    # --- Load Data ---
    logger.info(f"Loading data from: {args.data}")
    try:
        data = pd.read_csv(args.data)
        logger.info(f"Data loaded successfully. Shape: {data.shape}")
        logger.debug(f"Data columns: {list(data.columns)}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

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

    # `backtesting.py` expects column names in TitleCase.
    original_columns = list(data.columns)
    data.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume'
    }, inplace=True, errors='ignore')

    renamed_columns = list(data.columns)
    if original_columns != renamed_columns:
        logger.debug(f"Renamed columns for backtesting.py compatibility")

    # --- Instantiate Backtest ---
    logger.info("Setting up Backtest...")
    try:
        bt = Backtest(
            data,
            RLStrategy,
            cash=100000,
            commission=.002
        )
        logger.info("Backtest instance created successfully")
        logger.info(f"Initial cash: 100000, Commission: 0.2%")
    except Exception as e:
        logger.error(f"Failed to create Backtest instance: {e}")
        raise

    # --- Run the Backtest ---
    logger.info("Starting backtest execution...")
    try:
        stats = bt.run(
            model_path=args.model,
            params_path=args.params
        )
        logger.info("Backtest completed successfully")
    except Exception as e:
        logger.error(f"Backtest execution failed: {e}")
        raise

    # --- Print and Plot Results ---
    logger.info("=== Backtest Results ===")
    print("\n--- Backtest Results ---")
    print(stats)

    # Log key performance metrics
    logger.info(f"Final Return: {stats['Return [%]']:.2f}%")
    logger.info(f"Sharpe Ratio: {stats.get('Sharpe Ratio', 'N/A')}")
    logger.info(f"Max Drawdown: {stats.get('Max. Drawdown [%]', 'N/A')}%")
    logger.info(f"Total Trades: {stats.get('# Trades', 'N/A')}")

    model_name = os.path.splitext(os.path.basename(args.model))[0]
    data_name = os.path.splitext(os.path.basename(args.data))[0]
    plot_filename = f"backtest_{model_name}_on_{data_name}.html"

    logger.info(f"Saving plot to {plot_filename}...")
    try:
        bt.plot(filename=plot_filename, open_browser=False)
        logger.info(f"Plot saved successfully: {plot_filename}")
    except Exception as e:
        logger.warning(f"Failed to save plot: {e}")

    logger.info("=== Backtest Session Completed ===")


if __name__ == "__main__":
    import sys

    # Check if --data_folder is provided to determine batch vs single mode
    if "--data_folder" in sys.argv:
        run_batch_backtest()
    else:
        run_single_backtest()
