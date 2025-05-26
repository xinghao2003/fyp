import numpy as np

from time import time
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym


class Actions(Enum):
    Sell = 0
    Buy = 1
    Hold = 2  # New action to hold current position


class Positions(Enum):
    Short = 0
    Long = 1
    Neutral = 2  # New neutral position

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long

    def is_active(self):
        return self == Positions.Long or self == Positions.Short


class TradingEnv(gym.Env):

    metadata = {'render_modes': ['human'], 'render_fps': 3}

    def __init__(self, df, window_size, render_mode=None):
        assert df.ndim == 2
        assert render_mode is None or render_mode in self.metadata['render_modes']

        self.render_mode = render_mode

        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = gym.spaces.Discrete(len(Actions))
        INF = 1e10
        self.observation_space = gym.spaces.Box(
            low=-INF, high=INF, shape=self.shape, dtype=np.float32,
        )

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._truncated = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.action_space.seed(
            int((self.np_random.uniform(0, seed if seed is not None else 1))))

        self._truncated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Short
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def step(self, action):
        self._truncated = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._truncated = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(action)

        trade = False
        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            trade = True

        if trade:
            self._position = self._position.opposite()
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        if self.render_mode == 'human':
            self._render_frame()

        return observation, step_reward, False, self._truncated, info

    def _get_info(self):
        return dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position
        )

    def _get_observation(self):
        return self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _render_frame(self):
        self.render()

    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            marker = 'o'
            if position == Positions.Short:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            elif position == Positions.Neutral:
                color = 'blue'
                marker = '.'
            if color:
                plt.scatter(tick, self.prices[tick],
                            color=color, marker=marker)

        start_time = time()

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        end_time = time()
        process_time = end_time - start_time

        pause_time = (1 / self.metadata['render_fps']) - process_time
        if pause_time <= 0:
            pause_time = 0.001

        plt.pause(pause_time)

    def render_all(self, title=None):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        neutral_ticks = []
        for i, tick in enumerate(window_ticks):
            if i < len(self._position_history) and self._position_history[i] is not None:
                if self._position_history[i] == Positions.Short:
                    short_ticks.append(tick)
                elif self._position_history[i] == Positions.Long:
                    long_ticks.append(tick)
                elif self._position_history[i] == Positions.Neutral:
                    neutral_ticks.append(tick)

        valid_short_ticks = [t for t in short_ticks if t < len(self.prices)]
        valid_long_ticks = [t for t in long_ticks if t < len(self.prices)]
        valid_neutral_ticks = [
            t for t in neutral_ticks if t < len(self.prices)]

        if valid_short_ticks:
            plt.plot(valid_short_ticks,
                     self.prices[valid_short_ticks], 'ro', markersize=5, label='Short')
        if valid_long_ticks:
            plt.plot(
                valid_long_ticks, self.prices[valid_long_ticks], 'go', markersize=5, label='Long')
        if valid_neutral_ticks:
            plt.plot(valid_neutral_ticks,
                     self.prices[valid_neutral_ticks], 'bo', markersize=2, label='Neutral')

        if title:
            plt.title(title)
        plt.legend()
        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self):
        raise NotImplementedError

    def _calculate_reward(self, action):
        raise NotImplementedError

    def _update_profit(self, action):
        raise NotImplementedError

    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError


class StocksEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, render_mode=None):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size, render_mode)

        # Override action space for 3 actions
        self.action_space = gym.spaces.Discrete(len(Actions))

        # Fees
        self.buy_fee_percent = 0.005   # 0.5% cost when buying
        self.sell_fee_percent = 0.01   # 1.0% cost when selling

        # Penalties
        self.neutral_hold_penalty = -0.0001  # Small penalty for staying neutral
        self.active_hold_penalty = 0         # Penalty for redundant actions

        self._last_trade_price = 0.  # Price at which current position was entered

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)

        # Override to start in Neutral position
        self._position = Positions.Neutral
        self._last_trade_price = 0.

        # Update position history
        if self._position_history:
            self._position_history[-1] = self._position
        else:
            self._position_history = (
                self.window_size * [None]) + [self._position]

        info = self._get_info()
        return observation, info

    def step(self, action_value):
        action = Actions(action_value)
        self._truncated = False
        self._current_tick += 1

        if self._current_tick >= self._end_tick:
            self._truncated = True

        previous_position = self._position
        current_price = self.prices[self._current_tick]

        step_reward = self._calculate_reward(
            action, previous_position, current_price)
        self._total_reward += step_reward

        self._update_profit(action, previous_position, current_price)

        # Position transition logic
        if previous_position == Positions.Neutral:
            if action == Actions.Buy:
                self._position = Positions.Long
                self._last_trade_price = current_price
                self._last_trade_tick = self._current_tick
            elif action == Actions.Sell:
                self._position = Positions.Short
                self._last_trade_price = current_price
                self._last_trade_tick = self._current_tick
            # If Hold, position remains Neutral

        elif previous_position == Positions.Long:
            if action == Actions.Sell:  # Closing Long
                self._position = Positions.Neutral
            # If Buy or Hold, position remains Long

        elif previous_position == Positions.Short:
            if action == Actions.Buy:  # Closing Short
                self._position = Positions.Neutral
            # If Sell or Hold, position remains Short

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        if self.render_mode == 'human':
            self._render_frame()

        return observation, step_reward, False, self._truncated, info

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        # validate index (TODO: Improve validation)
        prices[self.frame_bound[0] - self.window_size]
        prices = prices[self.frame_bound[0] -
                        self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _calculate_reward(self, action_taken: Actions, previous_position: Positions, current_price: float):
        step_reward = 0.

        # Penalty for holding Neutral
        if previous_position == Positions.Neutral and action_taken == Actions.Hold:
            step_reward += self.neutral_hold_penalty

        # Penalty for redundant actions
        if previous_position == Positions.Long:
            if action_taken == Actions.Buy or action_taken == Actions.Hold:
                step_reward += self.active_hold_penalty
        elif previous_position == Positions.Short:
            if action_taken == Actions.Sell or action_taken == Actions.Hold:
                step_reward += self.active_hold_penalty

        # Reward/Loss from closing positions
        if previous_position == Positions.Long and action_taken == Actions.Sell:
            # Selling to close Long position
            cost_to_buy_one_share = self._last_trade_price * \
                (1 + self.buy_fee_percent)
            proceeds_from_selling_one_share = current_price * \
                (1 - self.sell_fee_percent)
            profit_from_trade_one_share = proceeds_from_selling_one_share - cost_to_buy_one_share
            step_reward += profit_from_trade_one_share

        elif previous_position == Positions.Short and action_taken == Actions.Buy:
            # Buying to close Short position
            proceeds_from_initial_short_sell = self._last_trade_price * \
                (1 - self.sell_fee_percent)
            cost_to_buy_back_one_share = current_price * \
                (1 + self.buy_fee_percent)
            profit_from_trade_one_share = proceeds_from_initial_short_sell - \
                cost_to_buy_back_one_share
            step_reward += profit_from_trade_one_share

        return step_reward

    def _update_profit(self, action_taken: Actions, previous_position: Positions, current_price: float):
        closed_long_this_step = (
            previous_position == Positions.Long and action_taken == Actions.Sell)
        closed_short_this_step = (
            previous_position == Positions.Short and action_taken == Actions.Buy)

        if closed_long_this_step:
            effective_entry_cost_per_share = self._last_trade_price * \
                (1 + self.buy_fee_percent)
            effective_exit_proceeds_per_share = current_price * \
                (1 - self.sell_fee_percent)
            if effective_entry_cost_per_share > 0:
                profit_factor = effective_exit_proceeds_per_share / effective_entry_cost_per_share
                self._total_profit *= profit_factor

        elif closed_short_this_step:
            if self._last_trade_price > 0:
                profit_from_trade_abs = (self._last_trade_price * (1 - self.sell_fee_percent)) - \
                                        (current_price * (1 + self.buy_fee_percent))
                profit_factor = 1 + \
                    (profit_from_trade_abs / self._last_trade_price)
                self._total_profit *= profit_factor

        elif self._truncated:
            # Episode ended with open position - liquidate at current price
            if previous_position == Positions.Long:
                effective_entry_cost_per_share = self._last_trade_price * \
                    (1 + self.buy_fee_percent)
                effective_exit_proceeds_per_share = current_price * \
                    (1 - self.sell_fee_percent)
                if effective_entry_cost_per_share > 0:
                    profit_factor = effective_exit_proceeds_per_share / effective_entry_cost_per_share
                    self._total_profit *= profit_factor

            elif previous_position == Positions.Short:
                if self._last_trade_price > 0:
                    profit_from_trade_abs = (self._last_trade_price * (1 - self.sell_fee_percent)) - \
                                            (current_price *
                                             (1 + self.buy_fee_percent))
                    profit_factor = 1 + \
                        (profit_from_trade_abs / self._last_trade_price)
                    self._total_profit *= profit_factor

    def max_possible_profit(self):
        current_profit = 1.0
        trade_entry_price = 0.0
        ideal_position = Positions.Neutral

        for idx in range(self._start_tick, self._end_tick + 1):
            price_at_idx = self.prices[idx]

            # Close existing position if optimal or necessary
            if ideal_position == Positions.Long:
                should_close_long = False
                if idx == self._end_tick:
                    should_close_long = True
                elif idx < self._end_tick and self.prices[idx+1] <= price_at_idx:
                    should_close_long = True

                if should_close_long:
                    entry_cost = trade_entry_price * (1 + self.buy_fee_percent)
                    exit_proceeds = price_at_idx * (1 - self.sell_fee_percent)
                    if entry_cost > 0:
                        current_profit *= (exit_proceeds / entry_cost)
                    ideal_position = Positions.Neutral
                    trade_entry_price = 0.0

            elif ideal_position == Positions.Short:
                should_close_short = False
                if idx == self._end_tick:
                    should_close_short = True
                elif idx < self._end_tick and self.prices[idx+1] >= price_at_idx:
                    should_close_short = True

                if should_close_short:
                    entry_proceeds = trade_entry_price * \
                        (1 - self.sell_fee_percent)
                    exit_cost = price_at_idx * (1 + self.buy_fee_percent)
                    if trade_entry_price > 0:
                        abs_profit_from_trade = entry_proceeds - exit_cost
                        current_profit *= (1 +
                                           (abs_profit_from_trade / trade_entry_price))
                    ideal_position = Positions.Neutral
                    trade_entry_price = 0.0

            # Open new position if optimal (and not the last tick)
            if ideal_position == Positions.Neutral and idx < self._end_tick:
                if self.prices[idx+1] > price_at_idx:
                    ideal_position = Positions.Long
                    trade_entry_price = price_at_idx
                elif self.prices[idx+1] < price_at_idx:
                    ideal_position = Positions.Short
                    trade_entry_price = price_at_idx

        return current_profit
