import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover

# --- 0. Benchmark Strategy: Buy & Hold ---
# This strategy provides a crucial baseline. It performs a single 'buy'
# transaction at the beginning and holds the asset for the entire duration
# of the backtest. Its performance represents the raw return of the asset itself.


class BuyAndHold(Strategy):
    """
    A benchmark strategy that buys on the first data point and holds
    until the end of the backtest.
    """

    def init(self):
        pass

    def next(self):
        # If not already invested, buy the asset.
        if not self.position:
            self.buy()

# --- 1. Trend-Following: Simple Moving Average Crossover ---
# This is a classic trend-following strategy. It operates on the principle
# that an asset's momentum will continue in its current direction. A "golden cross"
# (short-term average crossing above long-term) is a buy signal, while a
# "death cross" (short-term crossing below) is a sell signal.


class SmaCross(Strategy):
    """
    A classic trend-following strategy based on the crossover of two
    Simple Moving Averages (SMAs).

    Parameters
    ----------
    n_fast : int
        The lookbook period for the shorter (faster) moving average.
    n_slow : int
        The lookback period for the longer (slower) moving average.
    """
    # Define parameters for potential optimization
    n_fast = 10
    n_slow = 20

    def init(self):
        """
        Pre-computes the moving averages once before the backtest begins
        for efficiency.
        """
        # A helper function from the pandas library to compute the SMA
        def sma(values, n):
            return pd.Series(values).rolling(n).mean()

        # Use the self.I() method to efficiently calculate and plot indicators
        self.sma_fast = self.I(sma, self.data.Close, self.n_fast)
        self.sma_slow = self.I(sma, self.data.Close, self.n_slow)

    def next(self):
        """
        Defines the trading logic that is executed on each candlestick bar.
        """
        # Buy signal: when the fast MA crosses above the slow MA.
        if crossover(self.sma_fast, self.sma_slow):
            self.position.close()  # Ensure any short position is closed
            self.buy()

        # Sell signal: when the fast MA crosses below the slow MA.
        elif crossover(self.sma_slow, self.sma_fast):
            self.position.close()  # Ensure any long position is closed
            self.sell()

# --- 2. Mean-Reversion: Relative Strength Index (RSI) ---
# This strategy is based on the mean-reversion principle, which assumes
# that asset prices will revert to their historical average. The RSI is an
# oscillator that indicates "oversold" or "overbought" conditions. The strategy
# buys when the asset is deemed oversold and sells when it's overbought.


class RsiReversion(Strategy):
    """
    A mean-reversion strategy using the Relative Strength Index (RSI).
    It buys when the asset is oversold and sells when it's overbought.

    Parameters
    ----------
    oversold_level : int
        The RSI level below which the asset is considered oversold.
    overbought_level : int
        The RSI level above which the asset is considered overbought.
    rsi_window : int
        The lookback period for calculating the RSI.
    """
    oversold_level = 30
    overbought_level = 70
    rsi_window = 14

    def init(self):
        """
        Pre-computes the RSI indicator.
        """
        def rsi(close, window):
            """Calculate Relative Strength Index"""
            close_series = pd.Series(close)
            delta = close_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        self.rsi = self.I(rsi, self.data.Close, self.rsi_window)

    def next(self):
        """
        Defines the trading logic based on RSI levels.
        """
        # Buy signal: if the RSI crosses below the oversold level.
        if crossover(self.oversold_level, self.rsi):
            self.position.close()
            self.buy()

        # Sell signal: if the RSI crosses above the overbought level.
        elif crossover(self.rsi, self.overbought_level):
            self.position.close()
            self.sell()

# --- 3. Volatility-Based: Bollinger Bands Mean Reversion ---
# This strategy also follows a mean-reversion philosophy but uses volatility
# to generate signals. Bollinger Bands consist of a moving average (the middle band)
# and two outer bands representing standard deviations. The strategy assumes that
# when the price touches an outer band, it's likely to revert toward the middle band.


class BollingerReversion(Strategy):
    """
    A volatility-based, mean-reversion strategy using Bollinger Bands.
    It sells when the price hits the upper band and buys when it hits the lower band.

    Parameters
    ----------
    bb_window : int
        The lookback period for the moving average and standard deviation.
    bb_std : int
        The number of standard deviations for the upper and lower bands.
    """
    bb_window = 20
    bb_std = 2

    def init(self):
        """
        Pre-computes the Bollinger Bands.
        """
        # Custom function to calculate bands
        def bollinger_bands(price, n, std):
            series = pd.Series(price)
            ma = series.rolling(n).mean()
            std_dev = series.rolling(n).std()
            return ma + std * std_dev, ma - std * std_dev

        # self.I can compute multiple return values at once
        self.upper_band, self.lower_band = self.I(
            bollinger_bands, self.data.Close, self.bb_window, self.bb_std)

    def next(self):
        """
        Defines the trading logic based on price touching the bands.
        """
        # Sell signal: if the price crosses above the upper band.
        if crossover(self.data.Close, self.upper_band):
            self.position.close()
            self.sell()

        # Buy signal: if the price crosses below the lower band.
        elif crossover(self.lower_band, self.data.Close):
            self.position.close()
            self.buy()


# --- 4. Momentum/Breakout: Donchian Channel Breakout ---
# A classic momentum strategy made famous by the Turtle Traders. It is a pure
# trend-following system that buys when the price breaks above its highest high
# over a certain period and sells when it breaks below its lowest low. This
# strategy is designed to catch the beginning of major trends.

class DonchianBreakout(Strategy):
    """
    A momentum/breakout strategy using Donchian Channels. It buys when
    the price makes a new high and sells on a new low.

    Parameters
    ----------
    lookback_period : int
        The lookback period for determining the highest high and lowest low.
    """
    lookback_period = 20

    def init(self):
        """
        Pre-computes the Donchian Channel highs and lows.
        """
        self.donchian_high = self.I(
            lambda: pd.Series(self.data.High).rolling(self.lookback_period).max())
        self.donchian_low = self.I(
            lambda: pd.Series(self.data.Low).rolling(self.lookback_period).min())

    def next(self):
        """
        Defines the trading logic based on price breaking the channel.
        Note: We check against the previous bar's high/low to enter on a breakout.
        """
        # Buy signal: if price closes above the previous high of the channel.
        if self.data.Close[-1] > self.donchian_high[-2]:
            self.position.close()
            self.buy()

        # Sell signal: if price closes below the previous low of the channel.
        elif self.data.Close[-1] < self.donchian_low[-2]:
            self.position.close()
            self.sell()
