# Core DRL trading environment for stock trading simulation
import numpy as np
import pandas as pd
from gymnasium import Env, spaces
from typing import Callable, Optional, Dict, Any, Tuple, Union, Sequence, cast


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
                 action_space_type: str = 'multidiscrete',
                 **kwargs):
        # Allow agent_type to override action_space_type if passed in kwargs
        agent_type = kwargs.pop('agent_type', None)
        # Only PPO and SAC supported; always use multidiscrete
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
        self.action_space_type = action_space_type.lower()

        if self.action_space_type == 'multidiscrete':
            # MultiDiscrete action space: [action_type, units]
            self.action_space = spaces.MultiDiscrete(
                [3, max_stock_per_trade + 1])
        else:
            # Discrete action space: converted to (3 * (max_stock_per_trade + 1)) options
            # 0 = hold with 0 units
            # 1 to max_stock_per_trade = buy with 1 to max_stock_per_trade units
            # max_stock_per_trade+1 to 2*max_stock_per_trade = sell with 1 to max_stock_per_trade units
            self.action_space = spaces.Discrete(1 + 2 * max_stock_per_trade)

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
        return np.array([float(price), float(self.cash), float(self.position)], dtype=np.float32)

    def _get_price(self) -> float:
        """
        Return the close price for the current bar.
        We use `cast` only to convince the static type checker that the value
        really is float-like; at run time a NumPy scalar is perfectly valid.
        """
        idx = min(self.current_step, len(self.df) - 1)
        price_scalar = self.df.at[idx, 'close']            # type: Any
        return float(cast(float, price_scalar))

    def step(
        self, action: Union[int, np.ndarray, Sequence[int]]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Accepts action as [action_type, units] or as a discrete value
        if self.action_space_type == 'multidiscrete':
            # Handle different types of actions
            if isinstance(action, (np.ndarray, list, tuple)):
                # Convert numpy arrays to list and ensure we have at least 2 values
                action_list = np.array(action).flatten().tolist()
                if len(action_list) >= 2:
                    action_type, units = int(
                        action_list[0]), int(action_list[1])
                else:
                    # Only one value provided, assume it's action_type with units=1
                    action_type, units = int(action_list[0]), 1
            elif isinstance(action, (int, np.integer, float)):
                # Single scalar value, treat as action_type with units=1
                action_type, units = int(action), 1
            else:
                # Fallback for other types - log and use a default
                print(
                    f"Warning: Unexpected action type: {type(action)}, action: {action}")
                action_type, units = 0, 1  # Default to hold
        else:  # discrete action space
            # Make absolutely sure the object really is an int for the checker
            if not isinstance(action, (int, np.integer)):
                raise TypeError(
                    f"Discrete action space expects an int, got {type(action)}"
                )
            action_int: int = int(action)

            # Handle different discrete action interpretations:
            # 0 = hold with 0 units
            # 1 to max_stock_per_trade = buy with 1 to max_stock_per_trade units
            # max_stock_per_trade+1 to 2*max_stock_per_trade = sell with 1 to max_stock_per_trade units
            if action_int == 0:
                action_type, units = 0, 0  # Hold
            elif 1 <= action_int <= self.max_stock_per_trade:
                action_type, units = 1, action_int  # Buy 1 to max_stock_per_trade
            else:
                # Sell actions
                action_type = 2
                units = action_int - self.max_stock_per_trade
                if units > self.max_stock_per_trade:
                    units = self.max_stock_per_trade

        # Validate action format for each action space type
        if self.action_space_type == 'multidiscrete':
            assert self.action_space.contains(
                [action_type, units]), f"Invalid action: {[action_type, units]}"
        else:
            assert self.action_space.contains(
                action), f"Invalid action: {action}"
        price = self._get_price()
        done = False
        truncated = False
        info: Dict[str, Any] = {}

        prev_asset = self.cash + self.position * price
        if action_type == 1:  # Buy
            max_affordable = int(
                self.cash // (price * (1 + self.transaction_cost_pct)))
            buy_units = min(int(units), max_affordable,
                            self.max_stock_per_trade)
            if buy_units > 0:
                cost = buy_units * price * (1 + self.transaction_cost_pct)
                self.cash -= cost
                self.position += buy_units
        elif action_type == 2:  # Sell
            sell_units = min(int(units), abs(self.position))
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
        import logging
        logging.info(
            f"Step: {self.current_step}, Price: {self._get_price():.2f}, Cash: {self.cash:.2f}, Position: {self.position}, Total Asset: {self.total_asset:.2f}")

    def get_history(self):
        return pd.DataFrame(self.history)

    def seed(self, seed=None):
        np.random.seed(seed)
