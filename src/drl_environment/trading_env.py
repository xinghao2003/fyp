# trading_env.py
# Core DRL trading environment for stock trading simulation
import numpy as np
import pandas as pd
from gymnasium import Env, spaces
from typing import Callable, Optional, Dict, Any, Tuple, Union


class TradingEnv(Env):
    """
    A flexible trading environment for DRL agents.
    Supports single-asset trading with extensibility for multi-asset and more realistic features.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 df: pd.DataFrame,
                 initial_cash: float = 1000.0,
                 max_steps: Optional[int] = None,
                 reward_fn: Optional[Callable] = None,
                 transaction_cost_pct: float = 0.001,
                 allow_short: bool = False,
                 max_stock_per_trade: int = 100,
                 **kwargs):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.transaction_cost_pct = transaction_cost_pct
        self.allow_short = allow_short
        self.max_steps = max_steps or len(df) - 1
        self.reward_fn = reward_fn
        self.current_step = 0
        self.cash = initial_cash
        # Number of shares held (can be negative if allow_short)
        self.position = 0
        self.position_value = 0
        self.total_asset = initial_cash
        self.history = []
        # Action space: [action_type, units]
        # action_type: 0 = hold, 1 = buy, 2 = sell
        # units: 1 to max_stock_per_trade
        self.max_stock_per_trade = max_stock_per_trade
        self.action_space = spaces.MultiDiscrete([3, max_stock_per_trade + 1])
        # Observation: [price, cash, position]
        # Use concrete large negative number instead of -inf to avoid dtype issues
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1e10], dtype=np.float32),
            high=np.array([1e10, 1e10, 1e10], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed, options=options)

        self.current_step = 0
        self.cash = self.initial_cash
        self.position = 0
        self.position_value = 0
        self.total_asset = self.initial_cash
        self.history = []

        info = {}  # Additional info
        return self._get_obs(), info

    def _get_obs(self) -> np.ndarray:
        price = self._get_price()
        return np.array([price, self.cash, self.position], dtype=np.float32)

    def _get_price(self) -> float:
        # Prevent out-of-bounds access
        if self.current_step >= len(self.df):
            # Optionally, return the last price or handle as done
            return float(self.df.iloc[-1]['close'])
        return float(self.df.loc[self.current_step, 'close'])

    def step(self, action: Union[int, np.ndarray, tuple]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Accepts action as [action_type, units]
        if isinstance(action, (np.ndarray, list, tuple)):
            action_type, units = int(action[0]), int(action[1])
        else:
            # fallback for legacy single int actions
            action_type, units = int(action), 1
        assert self.action_space.contains(
            [action_type, units]), f"Invalid action: {[action_type, units]}"
        price = self._get_price()
        done = False
        truncated = False
        info: Dict[str, Any] = {}

        prev_asset = self.cash + self.position * price
        if action_type == 1:  # Buy
            max_affordable = int(
                self.cash // (price * (1 + self.transaction_cost_pct)))
            buy_units = min(units, max_affordable, self.max_stock_per_trade)
            if buy_units > 0:
                cost = buy_units * price * (1 + self.transaction_cost_pct)
                self.cash -= cost
                self.position += buy_units
        elif action_type == 2:  # Sell
            sell_units = min(units, abs(self.position))
            if self.position > 0 and sell_units > 0:
                proceeds = sell_units * price * (1 - self.transaction_cost_pct)
                self.cash += proceeds
                self.position -= sell_units
            elif self.allow_short and self.position <= 0:
                # Implement short selling logic if needed
                pass
        # else: hold

        self.current_step += 1
        if self.current_step >= self.max_steps or self.current_step >= len(self.df):
            done = True

        next_price = self._get_price() if not done else price
        self.position_value = self.position * next_price
        self.total_asset = self.cash + self.position_value

        # Reward
        if self.reward_fn:
            reward = self.reward_fn(self, prev_asset)
        else:
            reward = self.total_asset - prev_asset

        self.history.append({
            'step': self.current_step,
            'action': [action_type, units],
            'price': price,
            'cash': self.cash,
            'position': self.position,
            'total_asset': self.total_asset,
            'reward': reward
        })

        return self._get_obs(), float(reward), done, truncated, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Price: {self._get_price():.2f}, Cash: {self.cash:.2f}, Position: {self.position}, Total Asset: {self.total_asset:.2f}")

    def get_history(self):
        return pd.DataFrame(self.history)

    def seed(self, seed=None):
        np.random.seed(seed)


"""
Custom trading environment compatible with DRL libraries (e.g., extending FinRL or OpenAI Gym).
Defines state spaces, action spaces, and reward mechanism.
"""

# import gym # or from finrl.env.env_stocktrading import StockTradingEnv

# class CustomTradingEnv(gym.Env): # or class CustomTradingEnv(StockTradingEnv):
#     def __init__(self, df, **kwargs):
#         super().__init__(df, **kwargs)
#         # Define action and observation space
#         # They must be gym.spaces objects
#         # Example when using discrete actions:
#         # self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
#         # Example for using image as input (channel-first; channel-last also works):
#         # self.observation_space = spaces.Box(low=0, high=255,
#         #                                     shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)
#         pass

#     def step(self, action):
#         # Execute one time step within the environment
#         pass

#     def reset(self):
#         # Reset the state of the environment to an initial state
#         pass

#     def render(self, mode='human', close=False):
#         # Render the environment to the screen
#         pass

pass
