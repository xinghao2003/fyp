import numpy as np

from gym_anytrading.envs import TradingEnv, Actions, Positions


class StocksEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, render_mode=None):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size, render_mode)

        # Renamed for clarity: fees on transaction value
        # 0.5% cost when buying (was trade_fee_ask_percent)
        self.buy_fee_percent = 0.005
        # 1.0% cost when selling (was trade_fee_bid_percent)
        self.sell_fee_percent = 0.01

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        # validate index (TODO: Improve validation)
        prices[self.frame_bound[0] - self.window_size]
        prices = prices[self.frame_bound[0] -
                        self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _calculate_reward(self, action):
        step_reward = 0

        trade = False
        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                # Calculate net profit considering fees for a round trip
                # Cost of original purchase: last_trade_price * (1 + self.buy_fee_percent)
                # Proceeds from sale: current_price * (1 - self.sell_fee_percent)
                cost_to_buy_one_share = last_trade_price * \
                    (1 + self.buy_fee_percent)
                proceeds_from_selling_one_share = current_price * \
                    (1 - self.sell_fee_percent)
                profit_from_trade_one_share = proceeds_from_selling_one_share - cost_to_buy_one_share
                step_reward = profit_from_trade_one_share

        return step_reward

    def _update_profit(self, action):
        trade = False
        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            trade = True

        if trade or self._truncated:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                # Cost per share at entry: last_trade_price * (1 + self.buy_fee_percent)
                # Proceeds per share at exit: current_price * (1 - self.sell_fee_percent)
                effective_entry_cost_per_share = last_trade_price * \
                    (1 + self.buy_fee_percent)
                effective_exit_proceeds_per_share = current_price * \
                    (1 - self.sell_fee_percent)

                if effective_entry_cost_per_share > 0:  # Avoid division by zero
                    profit_factor_for_this_trade = effective_exit_proceeds_per_share / \
                        effective_entry_cost_per_share
                    self._total_profit *= profit_factor_for_this_trade

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]

                # Apply fees to the profit calculation
                effective_entry_cost_per_share = last_trade_price * \
                    (1 + self.buy_fee_percent)
                effective_exit_proceeds_per_share = current_price * \
                    (1 - self.sell_fee_percent)

                if effective_entry_cost_per_share > 0:
                    profit_factor = effective_exit_proceeds_per_share / effective_entry_cost_per_share
                    profit *= profit_factor

            last_trade_tick = current_tick - 1

        return profit
