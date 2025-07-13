import gym_trading_env
import gymnasium as gym
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import numpy as np
import random
import os
from datetime import datetime
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import logging
import shutil
import traceback
import sys
import math

# Import base reward function
# from reward import reward_function_5


def reward_function_5(
    history,
    window: int = 30,            # look-back for risk metrics
    r_free: float = 0.0,         # risk-free rate per step
    w_return: float = 1.00,      # weights for each component
    w_risk: float = 0.30,
    w_drawdown: float = 0.20,
    w_cost: float = 0.001,
    w_alpha: float = 0.50,
    clip_value: float = 1.0,
    eps: float = 1e-8
):
    """
    A robust reward that stays numerically stable and combines:
        • immediate log-return
        • risk-adjusted return        (Sharpe-like)
        • draw-down penalty
        • turnover / transaction cost penalty
        • market out-performance bonus (alpha)

    All terms are internally normalised and the final reward is
    clipped to [-clip_value, clip_value] to avoid gradient explosions.
    """

    # -------------- Safety checks --------------
    if len(history) < 2:
        return 0.0

    # ---------- 1. Immediate (current) return ----------
    curr_val = history["portfolio_valuation", -1]
    prev_val = history["portfolio_valuation", -2]
    r_t = np.log(curr_val / prev_val)                # robust to scale

    # ---------- 2. Risk-adjusted return (Sharpe) ----------
    # Take the last `window` log-returns
    values = np.asarray(history["portfolio_valuation"], dtype=np.float64)
    returns = np.diff(np.log(values[-(window + 1):]))
    if returns.size > 1:
        sharpe = (returns.mean() - r_free) / (returns.std() + eps)
    else:
        sharpe = 0.0

    # ---------- 3. Draw-down ----------
    peak = values.max()
    drawdown = (peak - curr_val) / (peak + eps)      # ∈ [0,1]

    # ---------- 4. Transaction-cost penalty ----------
    pos_now = history["position", -1]
    pos_prev = history["position", -2] if len(history) > 2 else pos_now
    # 0 (no trade) … 2 (full flip)
    turnover = abs(pos_now - pos_prev)

    # ---------- 5. Market out-performance (alpha) ----------
    if "data_close" in history.columns:
        m_ret = np.log(history["data_close", -1] / history["data_close", -2])
        alpha = r_t - m_ret
    else:
        alpha = 0.0

    # ---------- Final weighted reward ----------
    reward = (
        w_return * r_t +
        w_risk * sharpe -
        w_drawdown * drawdown -
        w_cost * turnover +
        w_alpha * alpha
    )

    # ---------- Numerical housekeeping ----------
    if np.isnan(reward) or np.isinf(reward):
        reward = 0.0
    reward = float(np.clip(reward, -clip_value, clip_value))
    return reward


# Helper functions for metrics
def calculate_max_drawdown(history):
    portfolio_valuations = np.asarray(
        history['portfolio_valuation'], dtype=np.float64)
    if len(portfolio_valuations) < 2:
        return 0.0
    peaks = np.maximum.accumulate(portfolio_valuations)
    drawdowns = (peaks - portfolio_valuations) / (peaks + 1e-8)
    return np.max(drawdowns)


def calculate_annualized_return(history):
    portfolio_valuations = np.asarray(
        history['portfolio_valuation'], dtype=np.float64)
    if len(portfolio_valuations) < 2:
        return 0.0

    total_return = (
        portfolio_valuations[-1] - portfolio_valuations[0]) / portfolio_valuations[0]

    start_date = pd.to_datetime(history['date', 0])
    end_date = pd.to_datetime(history['date', -1])
    duration_in_days = (end_date - start_date).days

    if duration_in_days <= 0:
        return 0.0

    duration_in_years = duration_in_days / 365.25
    annualized_return = (1 + total_return) ** (1 / duration_in_years) - 1
    return annualized_return


def calculate_sharpe_ratio(history, risk_free_rate=0.0404):
    portfolio_valuations = np.asarray(
        history['portfolio_valuation'], dtype=np.float64)
    if len(portfolio_valuations) < 2:
        return 0.0

    returns = np.diff(portfolio_valuations) / portfolio_valuations[:-1]

    start_date = pd.to_datetime(history['date', 0])
    end_date = pd.to_datetime(history['date', -1])
    duration_in_days = (end_date - start_date).days

    if duration_in_days <= 0:
        return 0.0

    # Assuming daily data for simplicity in calculating daily risk-free rate
    daily_risk_free_rate = (1 + risk_free_rate)**(1/365.25) - 1

    excess_returns = returns - daily_risk_free_rate

    mean_excess_return = np.mean(excess_returns)
    std_dev_returns = np.std(returns)

    if std_dev_returns < 1e-8:
        return 0.0

    # Annualize Sharpe Ratio
    sharpe_ratio = mean_excess_return / std_dev_returns
    annualized_sharpe = sharpe_ratio * \
        np.sqrt(365.25)  # Assuming daily returns
    return annualized_sharpe


# Generate unique timestamp-based ID for this run
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Please record this ID for tracking: {RUN_ID}")
# Configure logging with more detailed format
os.makedirs('optuna_logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'optuna_logs/debug_{RUN_ID}.log')
    ]
)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
SEED = 42


def set_seeds(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


set_seeds(SEED)

# Custom preprocessing function


def create_preprocess_function(feature_config):
    """Create a preprocessing function based on feature configuration"""
    def preprocess(df: pd.DataFrame):
        # Create your features based on the configuration
        try:
            # Log detailed DataFrame information for debugging
            logger.debug(f"Input DataFrame shape: {df.shape}")
            logger.debug(f"Available columns: {df.columns.tolist()}")
            logger.debug(
                f"DataFrame index range: {df.index.min()} to {df.index.max()}")

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"DataFrame info:\n{df.info()}")

            # Check for empty DataFrame
            if df.empty:
                logger.error("Input DataFrame is empty!")
                raise ValueError("Input DataFrame is empty")

            # Check for NaN values
            used_columns = ['open', 'close', 'high', 'low', 'volume', 'macd', 'rsi', 'close_10_sma',
                            'close_10_ema', 'adx', 'boll_ub', 'boll_lb', 'boll', 'kdjk', 'kdjd', 'kdjj', 'atr']
            used_columns = [f'norm_{col}' for col in used_columns]
            nan_counts = df[used_columns].isnull().sum()
            if nan_counts.any():
                logger.warning(
                    f"NaN values found in columns: {nan_counts[nan_counts > 0].to_dict()}")

            # Basic price features (always included for trading)
            if 'norm_close' in df.columns:
                if df['norm_close'].isna().all():
                    logger.error(
                        "Critical: 'norm_close' column is entirely NaN!")
                    raise ValueError("'norm_close' column is entirely NaN")
                df["feature_close"] = df["norm_close"]
            else:
                logger.error(
                    "Critical: 'norm_close' column not found in dataset!")
                raise ValueError(
                    "'norm_close' column is required but not found in dataset")

            # Optional features based on trial suggestions with validation
            if feature_config.get('use_volume', True):
                if 'norm_volume' in df.columns:
                    if df['norm_volume'].isna().all():
                        logger.warning(
                            "'norm_volume' column is entirely NaN, skipping")
                    else:
                        df["feature_volume"] = df["norm_volume"]
                else:
                    logger.debug("'norm_volume' column not found in dataset")

            if feature_config.get('use_high', True):
                if 'norm_high' in df.columns:
                    if df['norm_high'].isna().all():
                        logger.warning(
                            "'norm_high' column is entirely NaN, skipping")
                    else:
                        df["feature_high"] = df["norm_high"]
                else:
                    logger.debug("'norm_high' column not found in dataset")

            if feature_config.get('use_low', True):
                if 'norm_low' in df.columns:
                    if df['norm_low'].isna().all():
                        logger.warning(
                            "'norm_low' column is entirely NaN, skipping")
                    else:
                        df["feature_low"] = df["norm_low"]
                else:
                    logger.debug("'norm_low' column not found in dataset")

            if feature_config.get('use_open', True):
                if 'norm_open' in df.columns:
                    if df['norm_open'].isna().all():
                        logger.warning(
                            "'norm_open' column is entirely NaN, skipping")
                    else:
                        df["feature_open"] = df["norm_open"]
                else:
                    logger.debug("'norm_open' column not found in dataset")

            # Technical indicators with validation
            if feature_config.get('use_macd', True):
                if 'norm_macd' in df.columns:
                    if df['norm_macd'].isna().all():
                        logger.warning(
                            "'norm_macd' column is entirely NaN, skipping")
                    else:
                        df["feature_macd"] = df["norm_macd"]
                else:
                    logger.debug("'norm_macd' column not found in dataset")

            if feature_config.get('use_rsi', False):
                if 'norm_rsi' in df.columns:
                    if df['norm_rsi'].isna().all():
                        logger.warning(
                            "'norm_rsi' column is entirely NaN, skipping")
                    else:
                        df["feature_rsi"] = df["norm_rsi"]
                else:
                    logger.debug("'norm_rsi' column not found in dataset")

            if feature_config.get('use_sma', False):
                if 'norm_close_10_sma' in df.columns:
                    if df['norm_close_10_sma'].isna().all():
                        logger.warning(
                            "'norm_close_10_sma' column is entirely NaN, skipping")
                    else:
                        df["feature_sma"] = df["norm_close_10_sma"]
                else:
                    logger.debug(
                        "'norm_close_10_sma' column not found in dataset")

            if feature_config.get('use_ema', False):
                if 'norm_close_10_ema' in df.columns:
                    if df['norm_close_10_ema'].isna().all():
                        logger.warning(
                            "'norm_close_10_ema' column is entirely NaN, skipping")
                    else:
                        df["feature_ema"] = df["norm_close_10_ema"]
                else:
                    logger.debug(
                        "'norm_close_10_ema' column not found in dataset")

            if feature_config.get('use_adx', False):
                if 'norm_adx' in df.columns:
                    if df['norm_adx'].isna().all():
                        logger.warning(
                            "'norm_adx' column is entirely NaN, skipping")
                    else:
                        df["feature_adx"] = df["norm_adx"]
                else:
                    logger.debug("'norm_adx' column not found in dataset")

            if feature_config.get('use_bb_upper', False):
                if 'norm_boll_ub' in df.columns:
                    if df['norm_boll_ub'].isna().all():
                        logger.warning(
                            "'norm_boll_ub' column is entirely NaN, skipping")
                    else:
                        df["feature_bb_upper"] = df["norm_boll_ub"]
                else:
                    logger.debug("'norm_boll_ub' column not found in dataset")

            if feature_config.get('use_bb_lower', False):
                if 'norm_boll_lb' in df.columns:
                    if df['norm_boll_lb'].isna().all():
                        logger.warning(
                            "'norm_boll_lb' column is entirely NaN, skipping")
                    else:
                        df["feature_bb_lower"] = df["norm_boll_lb"]
                else:
                    logger.debug("'norm_boll_lb' column not found in dataset")

            if feature_config.get('use_bb_middle', False):
                if 'norm_boll' in df.columns:
                    if df['norm_boll'].isna().all():
                        logger.warning(
                            "'norm_boll' column is entirely NaN, skipping")
                    else:
                        df["feature_bb_middle"] = df["norm_boll"]
                else:
                    logger.debug("'norm_boll' column not found in dataset")

            if feature_config.get('use_stoch_k', False):
                if 'norm_kdjk' in df.columns:
                    if df['norm_kdjk'].isna().all():
                        logger.warning(
                            "'norm_kdjk' column is entirely NaN, skipping")
                    else:
                        df["feature_stoch_k"] = df["norm_kdjk"]
                else:
                    logger.debug("'norm_kdjk' column not found in dataset")

            if feature_config.get('use_stoch_d', False):
                if 'norm_kdjd' in df.columns:
                    if df['norm_kdjd'].isna().all():
                        logger.warning(
                            "'norm_kdjd' column is entirely NaN, skipping")
                    else:
                        df["feature_stoch_d"] = df["norm_kdjd"]
                else:
                    logger.debug("'norm_kdjd' column not found in dataset")

            if feature_config.get('use_stoch_j', False):
                if 'norm_kdjj' in df.columns:
                    if df['norm_kdjj'].isna().all():
                        logger.warning(
                            "'norm_kdjj' column is entirely NaN, skipping")
                    else:
                        df["feature_stoch_j"] = df["norm_kdjj"]
                else:
                    logger.debug("'norm_kdjj' column not found in dataset")

            if feature_config.get('use_atr', False):
                if 'norm_atr' in df.columns:
                    if df['norm_atr'].isna().all():
                        logger.warning(
                            "'norm_atr' column is entirely NaN, skipping")
                    else:
                        df["feature_atr"] = df["norm_atr"]
                else:
                    logger.debug("'norm_atr' column not found in dataset")

            # Validate that we have at least some features
            feature_columns = [
                col for col in df.columns if col.startswith('feature_')]
            if len(feature_columns) == 0:
                logger.error(
                    "No feature columns created! Check dataset and feature configuration.")
                raise ValueError(
                    "No valid features could be created from the dataset")

            logger.debug(
                f"Created {len(feature_columns)} features: {feature_columns}")
            logger.debug(f"Output DataFrame shape: {df.shape}")
            logger.debug(
                f"Feature columns data types: {df[feature_columns].dtypes.to_dict()}")

        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            logger.error(f"Available columns: {df.columns.tolist()}")
            logger.error(f"DataFrame shape: {df.shape}")
            logger.error(f"Feature config: {feature_config}")
            raise
        return df

    return preprocess


def create_tunable_reward_function(trial):
    """Create a reward function with tunable weights based on trial suggestions"""

    # Suggest weight parameters for different reward components
    w_return = trial.suggest_float('w_return', 0.5, 2.0)
    w_risk = trial.suggest_float('w_risk', 0.0, 1.0)
    w_drawdown = trial.suggest_float('w_drawdown', 0.0, 1.0)
    w_cost = trial.suggest_float('w_cost', 0.0001, 0.01)
    w_alpha = trial.suggest_float('w_alpha', 0.0, 1.0)

    # Other reward function parameters
    window = trial.suggest_int('reward_window', 10, 50)
    clip_value = trial.suggest_float('clip_value', 0.5, 2.0)

    def tunable_reward_function(history):
        try:
            # Add validation before calling reward function
            if history is None:
                logger.error("History is None in reward function")
                raise ValueError("History cannot be None")

            # Log history information for debugging
            if hasattr(history, 'shape'):
                logger.debug(f"History shape: {history.shape}")
            elif hasattr(history, '__len__'):
                logger.debug(f"History length: {len(history)}")
            else:
                logger.debug(f"History type: {type(history)}")

            # Check if history is a pandas DataFrame or Series
            if hasattr(history, 'index'):
                logger.debug(f"History index length: {len(history.index)}")
                if hasattr(history, 'columns'):
                    logger.debug(
                        f"History columns: {history.columns.tolist()}")

            return reward_function_5(
                history=history,
                window=window,
                w_return=w_return,
                w_risk=w_risk,
                w_drawdown=w_drawdown,
                w_cost=w_cost,
                w_alpha=w_alpha,
                clip_value=clip_value
            )
        except Exception as e:
            logger.error(f"Error in tunable_reward_function: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            logger.error(f"History type: {type(history)}")
            if hasattr(history, 'shape'):
                logger.error(f"History shape: {history.shape}")
            elif hasattr(history, '__len__'):
                logger.error(f"History length: {len(history)}")
            raise

    return tunable_reward_function


def evaluate_sharpe_ratio(model, eval_env, n_episodes=10, base_seed=42, annual_risk_free_rate=0.0404):
    """
    Evaluate the trained model using Sharpe ratio as a consistent metric.
    This function evaluates the actual trading performance independent of the reward function.

    IMPORTANT: All trials use the SAME sequence of episodes for fair comparison.
    Each trial evaluates on identical datasets/starting points to ensure that
    performance differences are due to hyperparameters, not random evaluation conditions.

    NOTICE: Current implementation has some issue on statistical properties of Sharpe ratio. But with the fixed ranges of validation dataset,
    the Sharpe ratio should be calculated in a mostly consistent manner across trials.

    Args:
        model: Trained RL model
        eval_env: Evaluation environment
        n_episodes: Number of episodes to evaluate
        base_seed: Base seed for consistent evaluation across all trials
        annual_risk_free_rate (float): The annualized risk-free rate (e.g., 0.0404 for 4.04%, taken from annual performance of US 1-year Treasury).

    Returns:
        float: Sharpe ratio of the portfolio returns
    """
    portfolio_values = []
    episode_returns = []
    episode_excess_returns = []

    for episode in range(n_episodes):
        # Use SAME seed sequence for ALL trials - ensures fair comparison
        # All trials evaluate on identical episodes:
        # - Trial A, Episode 0: base_seed + 0
        # - Trial A, Episode 1: base_seed + 1
        # - Trial B, Episode 0: base_seed + 0  (SAME as Trial A)
        # - Trial B, Episode 1: base_seed + 1  (SAME as Trial A)
        # This guarantees fair comparison across trials
        episode_seed = base_seed + episode
        logger.info(f"Starting episode {episode} with seed {episode_seed}")

        obs, _ = eval_env.reset(seed=episode_seed)
        done = False
        initial_value = None
        step_count = 0

        # Clear portfolio values for the new episode
        portfolio_values.clear()

        # Get initial episode information after reset
        try:
            if hasattr(eval_env, 'unwrapped') and hasattr(eval_env.unwrapped, 'historical_info'):
                history = eval_env.unwrapped.historical_info

                # Get symbol information
                symbol = "Unknown"
                if 'data_symbol' in history.columns:
                    symbol = history['data_symbol', -
                                     1] if len(history) > 0 else "Unknown"

                # Get start date
                start_date = "Unknown"
                if 'date' in history.columns and len(history) > 0:
                    start_date = pd.to_datetime(
                        history['date', -1]).strftime('%Y-%m-%d')

                logger.info(
                    f"Episode {episode}: Symbol={symbol}, Start_date={start_date}")

        except Exception as e:
            logger.warning(
                f"Episode {episode}: Could not get initial episode info: {e}")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            step_count += 1

            # Track portfolio valuation
            if hasattr(eval_env, 'unwrapped') and hasattr(eval_env.unwrapped, 'historical_info'):
                current_value = eval_env.unwrapped.historical_info["portfolio_valuation", -1]
                logger.debug(
                    f"Episode {episode}, Step {step_count}: Portfolio value: {current_value:.2f}")
                if initial_value is None:
                    initial_value = current_value
                    logger.info(
                        f"Episode {episode}: Initial portfolio value: {initial_value:.2f}")
                portfolio_values.append(current_value)

        # Get final episode information
        try:
            if hasattr(eval_env, 'unwrapped') and hasattr(eval_env.unwrapped, 'historical_info'):
                history = eval_env.unwrapped.historical_info

                # Get symbol information (should be same as start)
                symbol = "Unknown"
                if 'data_symbol' in history.columns:
                    symbol = history['data_symbol', -
                                     1] if len(history) > 0 else "Unknown"

                # Get end date
                end_date = "Unknown"
                if 'date' in history.columns and len(history) > 0:
                    end_date = pd.to_datetime(
                        history['date', -1]).strftime('%Y-%m-%d')

                # Get episode length from history
                episode_length = len(history) if hasattr(
                    history, '__len__') else step_count

                logger.info(f"Episode {episode}: Symbol={symbol}, End_date={end_date}, "
                            f"Episode_length={episode_length}, Steps_taken={step_count}")

        except Exception as e:
            logger.warning(
                f"Episode {episode}: Could not get final episode info: {e}")

        # Calculate episode return and risk-adjusted return
        if initial_value is not None and len(portfolio_values) > 1:
            final_value = portfolio_values[-1]
            episode_return = (final_value - initial_value) / initial_value
            episode_returns.append(episode_return)

            logger.info(f"Episode {episode}: Initial_value={initial_value:.2f}, "
                        f"Final_value={final_value:.2f}, Episode_return={episode_return:.4f}")

            # Calculate episode duration in years to adjust the risk-free rate
            try:
                history = eval_env.unwrapped.historical_info
                start_date_obj = pd.to_datetime(history['date', 0])
                end_date_obj = pd.to_datetime(history['date', -1])
                duration_in_days = (end_date_obj - start_date_obj).days

                logger.debug(f"Episode {episode}: Duration={duration_in_days} days "
                             f"({start_date_obj.strftime('%Y-%m-%d')} to {end_date_obj.strftime('%Y-%m-%d')})")

                # Avoid division by zero if episode is less than a day
                if duration_in_days > 0:
                    duration_in_years = duration_in_days / 365.25
                    # De-annualize the risk-free rate for the episode's duration
                    episode_risk_free_return = (
                        1 + annual_risk_free_rate)**duration_in_years - 1
                else:
                    episode_risk_free_return = 0.0

                # Calculate excess return over the risk-free rate
                excess_return = episode_return - episode_risk_free_return
                episode_excess_returns.append(excess_return)

                logger.info(f"Episode {episode}: Risk_free_return={episode_risk_free_return:.4f}, "
                            f"Excess_return={excess_return:.4f}")

            except Exception as e:
                logger.warning(
                    f"Episode {episode}: Could not calculate duration/excess return: {e}")
                # Fallback: assume zero risk-free return
                excess_return = episode_return
                episode_excess_returns.append(excess_return)
        else:
            logger.warning(f"Episode {episode}: Insufficient data for return calculation "
                           f"(initial_value={initial_value}, portfolio_values_len={len(portfolio_values)})")

    # Calculate Sharpe ratio from episode returns
    if len(episode_returns) > 1:
        # Use mean of excess returns
        mean_excess_return = np.mean(episode_excess_returns)
        # Use std of portfolio returns (standard definition of Sharpe Ratio)
        std_return = np.std(episode_returns)

        # Avoid division by zero
        if std_return > 1e-8:
            sharpe_ratio = mean_excess_return / std_return
        else:
            # If no volatility, return mean excess return
            sharpe_ratio = mean_excess_return
    else:
        # Not enough episodes to calculate meaningful Sharpe ratio
        sharpe_ratio = 0.0

    logger.info(
        f"Final evaluation results across {len(episode_returns)} episodes:")
    logger.info(f"  Mean excess return: {np.mean(episode_excess_returns):.4f}")
    logger.info(f"  Std return: {np.std(episode_returns):.4f}")
    logger.info(f"  Sharpe ratio: {sharpe_ratio:.4f}")
    logger.info(f"  Episode returns: {[f'{r:.4f}' for r in episode_returns]}")

    return sharpe_ratio


class OptunaPruningCallback(BaseCallback):
    """Custom callback for Optuna pruning during training"""

    def __init__(self, trial: optuna.Trial, eval_env, model, base_seed=42, verbose: int = 0):
        super().__init__(verbose)
        self.trial = trial
        self.eval_env = eval_env
        self.model = model
        self.base_seed = base_seed

    def _on_step(self) -> bool:
        # This callback is called after each evaluation by EvalCallback
        # We'll evaluate using Sharpe ratio for pruning decisions
        if hasattr(self.parent, 'n_calls') and self.parent.n_calls > 0:
            logger.info(
                f"Pruning: Evaluating trial {self.trial.number} at step {self.parent.n_calls}")
            # Only evaluate for pruning every few evaluations to save computation
            if self.parent.n_calls % 2 == 0:  # Every 2nd evaluation
                try:
                    logger.info(
                        f"Pruning: Evaluating trial {self.trial.number} for pruning at step {self.parent.n_calls}")
                    # Quick Sharpe ratio evaluation for pruning (fewer episodes)
                    # Use consistent base seed for fair comparison across all trials
                    sharpe_ratio = evaluate_sharpe_ratio(
                        model=self.model,
                        eval_env=self.eval_env,
                        # Fewer episodes for faster pruning evaluation
                        n_episodes=math.ceil(65 * 0.1),
                        base_seed=self.base_seed  # Same episodes for all trials
                    )

                    # Optuna expects steps to be monotonically increasing
                    step = self.parent.n_calls

                    self.trial.report(sharpe_ratio, step)

                    if self.trial.should_prune():
                        raise optuna.TrialPruned()

                except Exception as e:
                    # If evaluation fails, don't prune (continue training)
                    logger.warning(f"Pruning evaluation failed: {e}")
                    pass

        return True


def objective(trial):
    """
    Optuna objective function for hyperparameter optimization.

    IMPORTANT: This function evaluates trials using Sharpe ratio instead of mean reward
    to avoid the circular dependency problem where the reward function weights are being
    optimized while simultaneously being used as the evaluation metric. 

    The Sharpe ratio provides a consistent, meaningful metric across all trials that
    measures risk-adjusted returns independent of the reward function formulation.
    """
    train_env = None
    eval_env = None
    trial_dir = None
    model = None

    try:
        logger.info(f"Starting trial {trial.number}")

        # Suggest hyperparameters
        learning_rate = trial.suggest_float(
            'learning_rate', 1e-5, 1e-2, log=True)
        n_steps = trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096])
        batch_size = trial.suggest_categorical(
            'batch_size', [32, 64, 128, 256])
        n_epochs = trial.suggest_int('n_epochs', 3, 30)
        gamma = trial.suggest_float('gamma', 0.9, 0.9999)
        gae_lambda = trial.suggest_float('gae_lambda', 0.8, 0.99)
        clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
        ent_coef = trial.suggest_float('ent_coef', 1e-4, 1e-2, log=True)
        vf_coef = trial.suggest_float('vf_coef', 0.1, 1.0)

        # PPO-specific hyperparameters
        # Increased minimum window size
        windows = trial.suggest_int('windows', 10, 60)
        trading_fees = trial.suggest_float('trading_fees', 0.0005, 0.002)
        borrow_interest_rate = trial.suggest_float(
            'borrow_interest_rate', 0.0001, 0.0005)

        logger.info(f"Trial {trial.number}: Suggested hyperparameters: "
                    f"learning_rate={learning_rate}, n_steps={n_steps}, batch_size={batch_size}, "
                    f"n_epochs={n_epochs}, gamma={gamma}, gae_lambda={gae_lambda}, "
                    f"clip_range={clip_range}, ent_coef={ent_coef}, vf_coef={vf_coef}, "
                    f"windows={windows}, trading_fees={trading_fees}, borrow_interest_rate={borrow_interest_rate}")

        # Feature selection hyperparameters
        # Note: 'close' price is always included as it's essential for trading
        # Other features are optional and will be optimized by Optuna
        feature_config = {
            # Basic OHLCV features (volume, high, low, open are optional)
            'use_volume': trial.suggest_categorical('use_volume', [True, False]),
            'use_high': trial.suggest_categorical('use_high', [True, False]),
            'use_low': trial.suggest_categorical('use_low', [True, False]),
            'use_open': trial.suggest_categorical('use_open', [True, False]),

            # Momentum Indicators
            # Relative Strength Index
            'use_rsi': trial.suggest_categorical('use_rsi', [True, False]),
            # Moving Average Convergence Divergence
            'use_macd': trial.suggest_categorical('use_macd', [True, False]),
            # Stochastic Oscillator %K
            'use_stoch_k': trial.suggest_categorical('use_stoch_k', [True, False]),
            # Stochastic Oscillator %D
            'use_stoch_d': trial.suggest_categorical('use_stoch_d', [True, False]),
            # Stochastic Oscillator %J
            'use_stoch_j': trial.suggest_categorical('use_stoch_j', [True, False]),

            # Trend Indicators
            # Simple Moving Average
            'use_sma': trial.suggest_categorical('use_sma', [True, False]),
            # Exponential Moving Average
            'use_ema': trial.suggest_categorical('use_ema', [True, False]),
            # Average Directional Index
            'use_adx': trial.suggest_categorical('use_adx', [True, False]),

            # Volatility Indicators
            # Bollinger Bands Upper
            'use_bb_upper': trial.suggest_categorical('use_bb_upper', [True, False]),
            # Bollinger Bands Lower
            'use_bb_lower': trial.suggest_categorical('use_bb_lower', [True, False]),
            # Bollinger Bands Middle
            'use_bb_middle': trial.suggest_categorical('use_bb_middle', [True, False]),
            # Average True Range
            'use_atr': trial.suggest_categorical('use_atr', [True, False]),
        }

        # Create preprocessing function with selected features
        preprocess_func = create_preprocess_function(feature_config)

        # Get reward function with tunable weights
        custom_reward_function = create_tunable_reward_function(trial)

        # Log selected features and reward weights for this trial
        selected_features = [feature for feature, enabled in feature_config.items()
                             if enabled and not feature.endswith('_window')]
        reward_weights = {
            'w_return': trial.params.get('w_return'),
            'w_risk': trial.params.get('w_risk'),
            'w_drawdown': trial.params.get('w_drawdown'),
            'w_cost': trial.params.get('w_cost'),
            'w_alpha': trial.params.get('w_alpha'),
            'reward_window': trial.params.get('reward_window'),
            'clip_value': trial.params.get('clip_value')
        }
        logger.info(
            f"Trial {trial.number}: Selected features: {selected_features}")
        logger.info(
            f"Trial {trial.number}: Reward weights: {reward_weights}")
        logger.info(f"Trial {trial.number}: Windows size: {windows}")

        # Create trial-specific directory
        trial_dir = f"./optuna_trials/{RUN_ID}/trial_{trial.number}"
        os.makedirs(trial_dir, exist_ok=True)

        # Training environment with better error handling
        try:
            logger.info(
                f"Trial {trial.number}: Creating training environment...")
            train_env = gym.make('MultiDatasetTradingEnv',
                                 dataset_dir='dataset/1d-2005/train/*.pkl',
                                 reward_function=custom_reward_function,
                                 preprocess=preprocess_func,
                                 windows=windows,
                                 positions=[-1, 0, 1],
                                 trading_fees=trading_fees,
                                 borrow_interest_rate=borrow_interest_rate,
                                 )

            train_env.add_metric(
                'Symbol', lambda history: history['data_symbol', -1] if 'data_symbol' in history.columns else 'Unknown')
            train_env.add_metric('Position Changes', lambda history: np.sum(
                np.diff(history['position']) != 0))
            train_env.add_metric(
                'Episode Length', lambda history: len(history['position']))
            train_env.add_metric(
                'Max Drawdown', lambda history: f"{calculate_max_drawdown(history) * 100:.2f}%")
            train_env.add_metric(
                'Annualized Return', lambda history: f"{calculate_annualized_return(history) * 100:.2f}%")
            train_env.add_metric(
                'Sharpe Ratio', lambda history: f"{calculate_sharpe_ratio(history):.2f}")

            logger.info(
                f"Trial {trial.number}: Resetting training environment...")
            obs, info = train_env.reset(seed=SEED + trial.number)
            logger.info(
                f"Trial {trial.number}: Training env reset successful. Obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
            logger.debug(
                f"Trial {trial.number}: Training environment info: {info}")

        except Exception as e:
            logger.error(
                f"Trial {trial.number}: Failed to create/reset training environment")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise

        # Create evaluation environment with better error handling
        try:
            logger.info(
                f"Trial {trial.number}: Creating evaluation environment...")
            eval_env = gym.make('MultiDatasetTradingEnv',
                                dataset_dir='dataset/1d-2005/val/*.pkl',
                                reward_function=custom_reward_function,
                                preprocess=preprocess_func,
                                windows=windows,
                                positions=[-1, 0, 1],
                                trading_fees=trading_fees,
                                borrow_interest_rate=borrow_interest_rate,
                                )

            eval_env.add_metric(
                'Symbol', lambda history: history['data_symbol', -1] if 'data_symbol' in history.columns else 'Unknown')
            eval_env.add_metric('Position Changes', lambda history: np.sum(
                np.diff(history['position']) != 0))
            eval_env.add_metric(
                'Episode Length', lambda history: len(history['position']))
            eval_env.add_metric(
                'Max Drawdown', lambda history: f"{calculate_max_drawdown(history) * 100:.2f}%")
            eval_env.add_metric(
                'Annualized Return', lambda history: f"{calculate_annualized_return(history) * 100:.2f}%")
            eval_env.add_metric(
                'Sharpe Ratio', lambda history: f"{calculate_sharpe_ratio(history):.2f}")

            logger.info(
                f"Trial {trial.number}: Resetting evaluation environment...")
            obs, info = eval_env.reset(seed=SEED + trial.number)
            logger.info(
                f"Trial {trial.number}: Eval env reset successful. Obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
            eval_env = Monitor(eval_env)

        except Exception as e:
            logger.error(
                f"Trial {trial.number}: Failed to create/reset evaluation environment")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            if train_env:
                train_env.close()
            raise

        # Create model with suggested hyperparameters
        try:
            logger.info(f"Trial {trial.number}: Creating model...")
            model = RecurrentPPO("MlpLstmPolicy",
                                 train_env,
                                 learning_rate=learning_rate,
                                 n_steps=n_steps,
                                 batch_size=batch_size,
                                 n_epochs=n_epochs,
                                 gamma=gamma,
                                 gae_lambda=gae_lambda,
                                 clip_range=clip_range,
                                 ent_coef=ent_coef,
                                 vf_coef=vf_coef,
                                 verbose=0,
                                 seed=SEED + trial.number,
                                 device="cpu",
                                 )
            logger.info(f"Trial {trial.number}: Model created successfully")
        except Exception as e:
            logger.error(f"Trial {trial.number}: Failed to create model")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            if train_env:
                train_env.close()
            if eval_env:
                eval_env.close()
            raise

        trial_total_timesteps = 2000000
        total_datasets = 65

        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=4, min_evals=4, verbose=1)

        # Create pruning callback with Sharpe ratio
        pruning_callback = OptunaPruningCallback(
            trial=trial,
            eval_env=eval_env,
            model=model,
            base_seed=SEED
        )

        # Chain the callbacks
        callback_list = CallbackList([stop_callback, pruning_callback])

        eval_callback = EvalCallback(eval_env,
                                     best_model_save_path=trial_dir,
                                     log_path=trial_dir,
                                     # Evaluate every 2.5% of total timesteps
                                     eval_freq=math.ceil(
                                         trial_total_timesteps * 0.025),
                                     n_eval_episodes=math.ceil(
                                         total_datasets * 0.1),  # 10% of datasets
                                     deterministic=True,
                                     render=False,
                                     callback_after_eval=callback_list,
                                     verbose=0)

        # Train the model with pruning enabled
        try:
            logger.info(f"Trial {trial.number}: Starting training...")
            model.learn(total_timesteps=trial_total_timesteps,
                        callback=eval_callback, progress_bar=True)
            logger.info(
                f"Trial {trial.number}: Training completed successfully")
        except Exception as e:
            logger.error(f"Trial {trial.number}: Training failed")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            # Additional context for training errors
            try:
                logger.error(
                    f"Training environment state: action_space={train_env.action_space}, observation_space={train_env.observation_space}")
                logger.error(f"Model parameters: {model.get_parameters()}")
            except:
                logger.error("Could not retrieve additional context")
            raise

        # Evaluate using Sharpe ratio instead of mean reward
        # This provides a consistent, meaningful metric across all trials
        logger.info(f"Trial {trial.number}: Evaluating with Sharpe ratio...")
        try:
            sharpe_ratio = evaluate_sharpe_ratio(
                model=model,
                eval_env=eval_env,
                n_episodes=math.ceil(total_datasets * 0.2),  # 20% of datasets
                base_seed=SEED  # Same episodes for all trials - fair comparison
            )
            logger.info(
                f"Trial {trial.number}: Sharpe ratio: {sharpe_ratio:.4f}")
        except Exception as e:
            logger.error(
                f"Trial {trial.number}: Sharpe ratio evaluation failed: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            sharpe_ratio = -np.inf

        # Clean up environments
        if train_env:
            train_env.close()
        if eval_env:
            eval_env.close()

        # Clean up trial directory if not the best (with safe best_value access)
        try:
            current_best = trial.study.best_value
            if sharpe_ratio < current_best:
                shutil.rmtree(trial_dir, ignore_errors=True)
        except (AttributeError, ValueError):
            # No best value exists yet or other database issue
            # Keep the trial directory for now
            pass

        logger.info(
            f"Trial {trial.number}: Completed with Sharpe ratio: {sharpe_ratio:.4f}")
        return sharpe_ratio

    except optuna.TrialPruned:
        logger.info(f"Trial {trial.number}: Pruned")
        # Clean up on pruning
        if train_env:
            train_env.close()
        if eval_env:
            eval_env.close()
        if trial_dir:
            shutil.rmtree(trial_dir, ignore_errors=True)
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {str(e)}")
        logger.error(f"Trial {trial.number} error type: {type(e).__name__}")
        logger.error(
            f"Full traceback for trial {trial.number}:\n{traceback.format_exc()}")

        # Additional debugging information
        try:
            logger.error(f"Trial {trial.number} parameters: {trial.params}")
            if train_env:
                logger.error(
                    f"Training env action_space: {train_env.action_space}")
                logger.error(
                    f"Training env observation_space: {train_env.observation_space}")
            if eval_env:
                logger.error(f"Eval env action_space: {eval_env.action_space}")
                logger.error(
                    f"Eval env observation_space: {eval_env.observation_space}")
        except Exception as debug_e:
            logger.error(f"Could not get additional debug info: {debug_e}")

        # Clean up on error
        if train_env:
            try:
                train_env.close()
            except:
                pass
        if eval_env:
            try:
                eval_env.close()
            except:
                pass
        if trial_dir:
            shutil.rmtree(trial_dir, ignore_errors=True)
        return -np.inf


def run_optuna_optimization(number_of_trials=50):
    """Run Optuna hyperparameter optimization"""
    # Create necessary directories
    os.makedirs('optuna_studies', exist_ok=True)
    os.makedirs(f'optuna_trials/{RUN_ID}', exist_ok=True)

    # Create study with better error handling
    study_db_path = f'sqlite:///optuna_studies/{RUN_ID}_study.db'

    try:
        study = optuna.create_study(
            direction='maximize',
            # TODO: Possibly use RUN_ID as seed, introduce randomness
            sampler=TPESampler(seed=SEED),
            pruner=MedianPruner(
                n_startup_trials=number_of_trials * 0.1,    # Increased for better baseline
                n_warmup_steps=10,      # Increased for more stable pruning
                interval_steps=5        # Check every 5 evaluations
            ),
            # Self define the RUN_ID for resume
            study_name=f"ppo_trading_{RUN_ID}",
            storage=study_db_path,
            load_if_exists=True,
        )
    except Exception as e:
        logger.warning(
            f"Failed to create study with database: {e}. Creating in-memory study.")
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=SEED),
            pruner=MedianPruner(
                n_startup_trials=number_of_trials * 0.1,
                n_warmup_steps=10,
                interval_steps=5
            ),
            study_name=f"ppo_trading_{RUN_ID}",
        )

    print(f"Starting Optuna optimization with study: {study.study_name}")
    print("Pruning enabled: Early stopping of unpromising trials")
    print("Evaluation metric: Sharpe ratio (consistent across all trials)")

    # Optimize with pruning
    study.optimize(objective, n_trials=number_of_trials,
                   timeout=number_of_trials * 3600)  # 1 hour per trial

    # Print results
    print("\nOptimization completed!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best Sharpe ratio: {study.best_value:.4f}")
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save study results
    study_path = f"./optuna_studies/{RUN_ID}_study.pkl"
    os.makedirs(os.path.dirname(study_path), exist_ok=True)
    with open(study_path, 'wb') as f:
        import pickle
        pickle.dump(study, f)

    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    best_trial = study.best_trial

    # Get best reward function with optimized weights
    temp_trial = optuna.trial.FixedTrial(best_trial.params)
    best_reward_function = create_tunable_reward_function(temp_trial)

    # Create best feature configuration
    best_feature_config = {
        'use_volume': best_trial.params.get('use_volume', True),
        'use_high': best_trial.params.get('use_high', True),
        'use_low': best_trial.params.get('use_low', True),
        'use_open': best_trial.params.get('use_open', True),
        # Momentum Indicators
        'use_rsi': best_trial.params.get('use_rsi', False),
        'use_macd': best_trial.params.get('use_macd', True),
        'use_stoch_k': best_trial.params.get('use_stoch_k', False),
        'use_stoch_d': best_trial.params.get('use_stoch_d', False),
        'use_stoch_j': best_trial.params.get('use_stoch_j', False),
        # Trend Indicators
        'use_sma': best_trial.params.get('use_sma', False),
        'use_ema': best_trial.params.get('use_ema', False),
        'use_adx': best_trial.params.get('use_adx', False),
        # Volatility Indicators
        'use_bb_upper': best_trial.params.get('use_bb_upper', False),
        'use_bb_lower': best_trial.params.get('use_bb_lower', False),
        'use_bb_middle': best_trial.params.get('use_bb_middle', False),
        'use_atr': best_trial.params.get('use_atr', False),
    }

    # Create best preprocessing function
    best_preprocess_func = create_preprocess_function(best_feature_config)

    # Create final environments
    final_train_env = gym.make('MultiDatasetTradingEnv',
                               dataset_dir='dataset/1d-2005/train/*.pkl',
                               reward_function=best_reward_function,
                               preprocess=best_preprocess_func,
                               windows=best_trial.params['windows'],
                               positions=[-1, 0, 1],
                               trading_fees=best_trial.params['trading_fees'],
                               borrow_interest_rate=best_trial.params['borrow_interest_rate'],
                               )

    final_train_env.add_metric(
        'Symbol', lambda history: history['data_symbol', -1] if 'data_symbol' in history.columns else 'Unknown')
    final_train_env.add_metric('Position Changes', lambda history: np.sum(
        np.diff(history['position']) != 0))
    final_train_env.add_metric(
        'Episode Length', lambda history: len(history['position']))
    final_train_env.add_metric(
        'Max Drawdown', lambda history: f"{calculate_max_drawdown(history) * 100:.2f}%")
    final_train_env.add_metric(
        'Annualized Return', lambda history: f"{calculate_annualized_return(history) * 100:.2f}%")
    final_train_env.add_metric(
        'Sharpe Ratio', lambda history: f"{calculate_sharpe_ratio(history):.2f}")

    final_eval_env = gym.make('MultiDatasetTradingEnv',
                              dataset_dir='dataset/1d-2005/val/*.pkl',
                              reward_function=best_reward_function,
                              preprocess=best_preprocess_func,
                              windows=best_trial.params['windows'],
                              positions=[-1, 0, 1],
                              trading_fees=best_trial.params['trading_fees'],
                              borrow_interest_rate=best_trial.params['borrow_interest_rate'],
                              )

    final_eval_env.add_metric(
        'Symbol', lambda history: history['data_symbol', -1] if 'data_symbol' in history.columns else 'Unknown')
    final_eval_env.add_metric('Position Changes', lambda history: np.sum(
        np.diff(history['position']) != 0))
    final_eval_env.add_metric(
        'Episode Length', lambda history: len(history['position']))
    final_eval_env.add_metric(
        'Max Drawdown', lambda history: f"{calculate_max_drawdown(history) * 100:.2f}%")
    final_eval_env.add_metric(
        'Annualized Return', lambda history: f"{calculate_annualized_return(history) * 100:.2f}%")
    final_eval_env.add_metric(
        'Sharpe Ratio', lambda history: f"{calculate_sharpe_ratio(history):.2f}")

    final_train_env.reset(seed=SEED)
    final_eval_env.reset(seed=SEED)
    final_eval_env = Monitor(final_eval_env)

    # Create final model
    final_model = RecurrentPPO("MlpLstmPolicy",
                               final_train_env,
                               learning_rate=best_trial.params['learning_rate'],
                               n_steps=best_trial.params['n_steps'],
                               batch_size=best_trial.params['batch_size'],
                               n_epochs=best_trial.params['n_epochs'],
                               gamma=best_trial.params['gamma'],
                               gae_lambda=best_trial.params['gae_lambda'],
                               clip_range=best_trial.params['clip_range'],
                               ent_coef=best_trial.params['ent_coef'],
                               vf_coef=best_trial.params['vf_coef'],
                               verbose=1,
                               tensorboard_log="./runs",
                               seed=SEED,
                               device="cpu",
                               )

    # Final training callbacks
    final_stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=4, min_evals=4, verbose=1)

    final_eval_callback = EvalCallback(final_eval_env,
                                       best_model_save_path=f'./model/{RUN_ID}/',
                                       log_path=f'./eval_logs/{RUN_ID}/',
                                       eval_freq=5000000 * 0.05,  # Every 5% of total timesteps
                                       n_eval_episodes=math.ceil(
                                           65 * 0.2),  # 20% of datasets
                                       deterministic=True,
                                       render=False,
                                       callback_after_eval=final_stop_callback,
                                       verbose=1)

    # Train final model
    print(f"Training final optimized model... [id: {RUN_ID}]")
    final_model.learn(total_timesteps=5000000,
                      tb_log_name=f"{RUN_ID}_optimized",
                      callback=final_eval_callback)

    final_model.save(f"./model/{RUN_ID}/final_optimized_model")

    # Save best parameters including feature configuration and reward weights
    import json

    # Extract reward weights from best parameters
    reward_weights = {
        'w_return': best_trial.params.get('w_return'),
        'w_risk': best_trial.params.get('w_risk'),
        'w_drawdown': best_trial.params.get('w_drawdown'),
        'w_cost': best_trial.params.get('w_cost'),
        'w_alpha': best_trial.params.get('w_alpha'),
        'reward_window': best_trial.params.get('reward_window'),
        'clip_value': best_trial.params.get('clip_value')
    }

    complete_config = {
        'hyperparameters': best_trial.params,
        'feature_config': best_feature_config,
        'reward_weights': reward_weights,
        'run_id': RUN_ID,
        'study_name': study.study_name,
        'best_sharpe_ratio': study.best_value,
        'best_trial_number': best_trial.number,
        'evaluation_metric': 'sharpe_ratio'
    }

    with open(f"./model/{RUN_ID}/best_params.json", 'w') as f:
        json.dump(complete_config, f, indent=2)

    # Also save feature configuration separately for easy access
    with open(f"./model/{RUN_ID}/feature_config.json", 'w') as f:
        json.dump(best_feature_config, f, indent=2)

    # Also save reward weights separately for easy access
    with open(f"./model/{RUN_ID}/reward_weights.json", 'w') as f:
        json.dump(reward_weights, f, indent=2)

    print(f"Optimization complete! Best model saved with ID: {RUN_ID}")
    print(f"Best feature configuration:")
    for feature, enabled in best_feature_config.items():
        if enabled and not feature.endswith('_window'):
            print(f"  - {feature}: {enabled}")

    print(f"\nBest reward weights:")
    for weight_name, weight_value in reward_weights.items():
        print(f"  - {weight_name}: {weight_value}")

    return study


if __name__ == "__main__":
    # Run Optuna optimization
    study = run_optuna_optimization()
