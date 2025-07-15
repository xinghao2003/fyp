# File: fyp-code-report/backtesting/backtest_rl_agent.py

import argparse
import json
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from backtesting import Backtest, Strategy
from sb3_contrib import RecurrentPPO

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            f'rl_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- The Bridge: RL Agent Strategy for backtesting.py ---
# This class wraps the trained Stable-Baselines3 agent so it can be used
# by the backtesting.py library.


class RLStrategy(Strategy):
    """
    A backtesting.py Strategy that uses a trained Stable-Baselines3
    RecurrentPPO agent to make trading decisions.
    """
    # --- Parameters passed from Backtest instance ---
    # These will be set when we instantiate the Backtest object.
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

            logger.info(
                f"Model decision - Action: {action_value} -> Target position: {target_position}")
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

        if target_position == 1:      # Target: LONG
            if not self.position.is_long:
                logger.info(
                    f"EXECUTING BUY: {position_before} -> LONG at price {current_price:.4f}")
                self.buy()
            else:
                logger.debug("Already LONG, no action needed")
        elif target_position == -1:   # Target: SHORT
            if not self.position.is_short:
                logger.info(
                    f"EXECUTING SELL: {position_before} -> SHORT at price {current_price:.4f}")
                self.sell()
            else:
                logger.debug("Already SHORT, no action needed")
        elif target_position == 0:    # Target: OUT/FLAT
            if self.position.size != 0:
                logger.info(
                    f"EXECUTING CLOSE: {position_before} -> FLAT at price {current_price:.4f}")
                self.position.close()
            else:
                logger.debug("Already FLAT, no action needed")

        position_after = "LONG" if self.position.is_long else "SHORT" if self.position.is_short else "FLAT"
        if position_before != position_after:
            logger.info(
                f"Position changed: {position_before} -> {position_after}")

        logger.debug(
            f"Step {self.step_count} completed. New position size: {self.position.size}, Equity: {self.equity:.2f}")

# --- Main Execution Block ---


def run_backtest():
    parser = argparse.ArgumentParser(
        description="Backtest a trained RL agent.")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained .zip model file (e.g., best_model.zip).")
    parser.add_argument("--params", type=str, required=True,
                        help="Path to the best_params.json file for the model.")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to the .csv data file for backtesting (e.g., from the test set).")

    args = parser.parse_args()

    logger.info("=== Starting RL Agent Backtest ===")
    logger.info(f"Model: {args.model}")
    logger.info(f"Params: {args.params}")
    logger.info(f"Data: {args.data}")

    # --- 1. Load Data ---
    logger.info(f"Loading data from: {args.data}")
    try:
        data = pd.read_csv(args.data)
        logger.info(f"Data loaded successfully. Shape: {data.shape}")
        logger.debug(f"Data columns: {list(data.columns)}")
        logger.debug(
            f"Data date range: {data.index[0] if hasattr(data, 'index') else 'N/A'} to {data.index[-1] if hasattr(data, 'index') else 'N/A'}")
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

    # --- 2. Instantiate Backtest ---
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

    # --- 3. Run the Backtest ---
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

    # --- 4. Print and Plot Results ---
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
    run_backtest()
