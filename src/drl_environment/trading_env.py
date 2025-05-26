# Trading environment using gym_anytrading
import numpy as np
import pandas as pd
import gymnasium as gym
import gym_anytrading
from gym_anytrading.envs import StocksEnv, ForexEnv, Actions, Positions
from gymnasium import Env, spaces
from typing import Callable, Optional, Dict, Any, Tuple, Union, Sequence, cast
import logging
from src.preprocessing.normalizers import TradingDataNormalizer, create_market_agnostic_features

logger = logging.getLogger(__name__)


class TradingEnv(Env):
    """
    A wrapper around gym_anytrading environments for DRL agents.
    Provides compatibility with both PPO and SAC agents while leveraging
    the proven gym_anytrading implementation.

    Now includes data normalization support for better generalization.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 df: pd.DataFrame,
                 initial_cash: float = 1000.0,
                 window_size: int = 10,
                 frame_bound: Optional[Tuple[int, int]] = None,
                 reward_fn: Optional[Callable] = None,
                 transaction_cost_pct: float = 0.001,
                 action_space_type: str = 'multidiscrete',
                 normalize_data: bool = True,
                 normalization_method: str = 'percentage_change',
                 add_market_agnostic_features: bool = True,
                 seed: int = 42,
                 **kwargs):
        """
        Initialize the trading environment using gym_anytrading.

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            initial_cash: Starting cash amount
            window_size: Number of previous time steps to include in observation
            frame_bound: (start, end) indices for data slice
            reward_fn: Custom reward function (optional)
            transaction_cost_pct: Transaction cost percentage
            action_space_type: Type of action space ('multidiscrete' or 'discrete')
            normalize_data: Whether to normalize the input data
            normalization_method: Method for normalization ('percentage_change', 'minmax', 'standard', 'robust')
            add_market_agnostic_features: Whether to add market-agnostic features
        """
        super().__init__()

        # Store original data for potential inverse transforms
        self.original_df = df.copy()
        self.normalize_data = normalize_data
        self.normalization_method = normalization_method
        self.add_market_agnostic_features = add_market_agnostic_features

        # Process data with normalization if enabled
        processed_df = self._preprocess_data(df)

        # Validate DataFrame columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in processed_df.columns for col in required_cols):
            raise ValueError(
                f"DataFrame must contain columns: {required_cols}")

        self.df = processed_df.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.window_size = window_size
        self.transaction_cost_pct = transaction_cost_pct
        self.action_space_type = action_space_type.lower()

        # Set frame_bound if not provided
        if frame_bound is None:
            frame_bound = (window_size, len(df))
        self.frame_bound = frame_bound

        # Create the base gym_anytrading environment
        self._create_base_env()
        # Set seed for environment and base_env
        self.seed(seed)

        # Override action space if needed for compatibility
        self._setup_action_space()

        # Track additional metrics
        self.history = []
        self.current_step = 0

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data with normalization and feature engineering.
        
        Args:
            df: Raw DataFrame with OHLCV data
            
        Returns:
            Processed DataFrame ready for training
        """
        processed_df = df.copy()
        
        # Add market-agnostic features if requested
        if self.add_market_agnostic_features:
            processed_df = create_market_agnostic_features(processed_df)
            logger.info("Added market-agnostic features")
        
        # Apply normalization if requested
        if self.normalize_data:
            self.normalizer = TradingDataNormalizer(method=self.normalization_method)
            
            if self.normalization_method == 'percentage_change':
                # For percentage change, we work with the features directly
                processed_df = self.normalizer.fit_transform(processed_df)
                logger.info(f"Applied {self.normalization_method} normalization")
            else:
                # For other methods, normalize core OHLCV columns
                core_cols = ['Open', 'High', 'Low', 'Close']
                if 'Volume' in processed_df.columns:
                    core_cols.append('Volume')
                
                # Fit and transform core columns
                core_data = processed_df[core_cols]
                normalized_core = self.normalizer.fit_transform(core_data)
                
                # Replace core columns with normalized versions
                for col in core_cols:
                    processed_df[col] = normalized_core[col]
                    
                logger.info(f"Applied {self.normalization_method} normalization to OHLCV data")
        
        return processed_df

    def _create_base_env(self):
        """Create the underlying gym_anytrading environment."""
        try:
            # Use stocks environment from gym_anytrading
            self.base_env = gym.make(
                'stocks-v0',
                df=self.df,
                window_size=self.window_size,
                frame_bound=self.frame_bound
            )
        except Exception as e:
            logger.warning(f"Failed to create stocks-v0 environment: {e}")
            # Fallback: create StocksEnv directly
            self.base_env = StocksEnv(
                df=self.df,
                window_size=self.window_size,
                frame_bound=self.frame_bound
            )

    def _setup_action_space(self):
        """Setup action space for compatibility with PPO/SAC."""
        if self.action_space_type == 'multidiscrete':
            # MultiDiscrete: [position_type, confidence/amount]
            # position_type: 0=hold, 1=buy, 2=sell
            # confidence: 0-10 (can be used to determine trade size)
            self.action_space = spaces.MultiDiscrete([3, 11])
        else:
            # Use the default action space from gym_anytrading (typically Discrete(3))
            self.action_space = self.base_env.action_space

        # Use observation space from base environment
        self.observation_space = self.base_env.observation_space

    def _convert_action(self, action: Union[int, np.ndarray, Sequence[int]]) -> int:
        """Convert action to format expected by gym_anytrading."""
        if self.action_space_type == 'multidiscrete':
            if isinstance(action, (np.ndarray, list, tuple)):
                action_array = np.asarray(action).flatten()
                if len(action_array) >= 2:
                    action_type, confidence = int(
                        action_array[0]), int(action_array[1])
                else:
                    action_type, confidence = int(
                        action_array[0]), 5  # Default confidence
            else:
                action_type, confidence = int(action), 5

            # Convert to gym_anytrading action format
            if action_type == 0:  # Hold
                return 0  # Actions.Hold
            elif action_type == 1:  # Buy
                return 1  # Actions.Buy
            else:  # Sell
                return 2  # Actions.Sell
        else:
            # Direct mapping for discrete actions
            return int(action)

    def reset(self, *, seed: Optional[int] = 42, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        if seed is not None:
            # Use gymnasium's random number generator
            super().reset(seed=seed)
            # For older gym_anytrading compatibility
            if hasattr(self.base_env, 'np_random'):
                from gymnasium.utils import seeding
                self.base_env.np_random, _ = seeding.np_random(seed)

        # Reset base environment
        obs = self.base_env.reset()

        # Handle different gym versions
        if isinstance(obs, tuple):
            obs, info = obs
        else:
            info = {}

        self.current_step = 0
        self.history = []

        return obs, info

    def step(self, action: Union[int, np.ndarray, Sequence[int]]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        # Convert action to gym_anytrading format
        converted_action = self._convert_action(action)

        # Execute step in base environment
        result = self.base_env.step(converted_action)

        # Handle different gym API versions
        if len(result) == 4:
            obs, reward, done, info = result
            truncated = False
        else:
            obs, reward, done, truncated, info = result

        # Apply transaction costs if specified
        if (hasattr(self.base_env, '_position') and
            hasattr(self.base_env, '_last_position') and
                getattr(self.base_env, '_position') != getattr(self.base_env, '_last_position')):
            # Position changed, apply transaction cost
            if isinstance(reward, (int, float)) and reward > 0:
                reward = float(reward) - \
                    (self.transaction_cost_pct * float(reward))

        # Track history
        self.current_step += 1
        self.history.append({
            'step': self.current_step,
            'action': action,
            'converted_action': converted_action,
            'reward': reward,
            'total_profit': getattr(self.base_env, '_total_profit', 0),
            'position': getattr(self.base_env, '_position', 0)
        })

        return obs, float(reward), done, truncated, info

    def render(self, mode='human'):
        """Render the environment."""
        if hasattr(self.base_env, 'render'):
            try:
                return self.base_env.render()
            except TypeError:
                # Handle environments that don't accept mode parameter
                return self.base_env.render()
        else:
            # Fallback rendering
            total_profit = getattr(self.base_env, '_total_profit', 0)
            position = getattr(self.base_env, '_position', 0)
            logger.info(
                f"Step: {self.current_step}, Total Profit: {total_profit:.2f}, Position: {position}")

    def close(self):
        """Close the environment."""
        if hasattr(self.base_env, 'close'):
            self.base_env.close()

    def get_history(self) -> pd.DataFrame:
        """Get trading history as DataFrame."""
        return pd.DataFrame(self.history)

    @property
    def total_asset(self) -> float:
        """Get total asset value for compatibility."""
        return getattr(self.base_env, '_total_profit', 0) + self.initial_cash

    @property
    def cash(self) -> float:
        """Get current cash for compatibility."""
        # Approximate cash based on position and total profit
        return self.initial_cash + getattr(self.base_env, '_total_profit', 0)

    @property
    def position(self) -> float:
        """Get current position for compatibility."""
        return getattr(self.base_env, '_position', 0)

    def seed(self, seed=42):
        import random
        import numpy as np
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        random.seed(seed)
        np.random.seed(seed)
        if hasattr(self.base_env, 'np_random'):
            from gymnasium.utils import seeding
            self.base_env.np_random, _ = seeding.np_random(seed)
        return [seed]
