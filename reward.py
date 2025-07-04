import numpy as np

# Basic reward function: log return of portfolio valuation


def reward_function(history):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])

# Custom reward function: risk-adjusted return


def reward_function_1(history):
    # Get portfolio valuation history as numpy array
    values = history["portfolio_valuation"]

    # Need at least 2 points to calculate returns
    if len(values) < 2:
        return 0.0

    # Ensure values is a numpy array and has valid numbers
    values = np.asarray(values, dtype=np.float64)

    # Calculate log returns
    log_returns = np.diff(np.log(values))
    current_return = log_returns[-1]

    # Calculate rolling volatility (risk measure)
    window = min(20, len(log_returns))
    if window > 1:
        volatility = np.std(log_returns[-window:])
    else:
        volatility = 0.0

    # Risk-adjusted reward: return minus volatility penalty
    # The 0.5 factor can be tuned to adjust risk aversion
    risk_adjusted_reward = current_return - 0.5 * volatility

    return risk_adjusted_reward


def reward_function_2(history):
    """
    Aggressive reward function that encourages risk-taking and active trading.
    This function:
    1. Amplifies returns - making positive returns more rewarding
    2. Rewards larger positions - encouraging full market exposure
    3. Adds trend-following bonus - rewarding riding profitable trends
    4. Penalizes being in cash - pushing for market participation
    """
    # Get portfolio valuation history
    values = history["portfolio_valuation"]

    # Need at least 2 points to calculate returns
    if len(values) < 2:
        return 0.0

    # Calculate base return
    current_return = np.log(values[-1] / values[-2])

    # Amplify returns with power function (preserves sign but increases magnitude)
    # Higher exponent = more aggressive return amplification
    amplified_return = np.sign(current_return) * np.abs(current_return) ** 1.5

    # Get position information
    current_position = history["position", -1]
    real_position = history["real_position", -1]

    # Reward for larger absolute positions (being fully invested)
    position_size_bonus = 0.001 * abs(real_position)

    # Bonus for correct directional bets
    # If return is positive and long position, or return is negative and short position
    if (current_return > 0 and real_position > 0) or (current_return < 0 and real_position < 0):
        direction_bonus = 0.002 * abs(real_position)
    else:
        direction_bonus = 0

    # Trend-following bonus - reward for maintaining profitable positions
    trend_bonus = 0.0
    if len(values) > 3 and len(history["position"]) > 1:
        previous_position = history["position", -2]
        previous_return = np.log(values[-2] / values[-3])

        # Reward for staying in a position that's working
        if abs(current_position - previous_position) < 0.1 and current_return > 0 and previous_return > 0:
            trend_bonus = 0.003

    # Penalty for staying in cash (near-zero position)
    cash_penalty = -0.001 if abs(real_position) < 0.1 else 0

    # Combine all components for final reward
    total_reward = amplified_return + position_size_bonus + \
        direction_bonus + trend_bonus + cash_penalty

    return total_reward


def reward_function_3(history):
    """
    A robust reward function that combines multiple factors to guide optimal trading performance:
    1. Portfolio returns (main objective)
    2. Risk-adjusted returns (Sharpe-like ratio)
    3. Transaction cost penalty
    4. Drawdown penalty
    5. Position stability bonus
    """

    # Get current and previous portfolio values
    current_value = history["portfolio_valuation", -1]
    previous_value = history["portfolio_valuation", -2]

    # 1. Basic return component (log returns for numerical stability)
    basic_return = np.log(current_value / previous_value)

    # 2. Risk-adjusted return component
    if len(history) >= 20:  # Need sufficient history for volatility calculation
        # Calculate rolling volatility of returns
        # Get all values first, then slice the last 20
        portfolio_values = history["portfolio_valuation"]
        recent_values = np.array(portfolio_values[-20:], dtype=np.float64)
        returns = np.diff(np.log(recent_values))
        volatility = np.std(returns) if len(
            returns) > 1 else 0.01  # Avoid division by zero

        # Sharpe-like adjustment (assuming 0 risk-free rate for simplicity)
        # Small epsilon to avoid division by zero
        risk_adjusted_return = basic_return / (volatility + 1e-6)
    else:
        risk_adjusted_return = basic_return

    # 3. Transaction cost penalty
    current_position = history["position", -1]
    previous_position = history["position", -
                                2] if len(history) > 1 else current_position
    position_changed = float(current_position != previous_position)
    transaction_penalty = -0.001 * position_changed  # Penalty for changing positions

    # 4. Drawdown penalty
    if len(history) >= 10:
        # Get all values first, then slice the last 10
        portfolio_values = history["portfolio_valuation"]
        recent_values = np.array(portfolio_values[-10:], dtype=np.float64)
        peak_value = np.max(recent_values)
        current_drawdown = (peak_value - current_value) / peak_value
        drawdown_penalty = -0.5 * \
            max(0, current_drawdown - 0.05)  # Penalize drawdowns > 5%
    else:
        drawdown_penalty = 0

    # 5. Position stability bonus (encourage holding profitable positions)
    if len(history) >= 5 and not position_changed:
        # Check if the position has been profitable over recent steps
        recent_returns = []
        for i in range(-5, -1):
            if i >= -len(history) + 1:
                ret = history["portfolio_valuation", i] / \
                    history["portfolio_valuation", i-1] - 1
                recent_returns.append(ret)

        if len(recent_returns) > 0 and np.mean(recent_returns) > 0:
            stability_bonus = 0.0001  # Small bonus for holding winning positions
        else:
            stability_bonus = 0
    else:
        stability_bonus = 0

    # 6. Market-relative performance bonus
    if "data_close" in history.columns:
        current_price = history["data_close", -1]
        previous_price = history["data_close", -2]
        market_return = np.log(current_price / previous_price)

        # Reward for outperforming the market
        alpha = basic_return - market_return
        market_bonus = 0.5 * alpha  # Weight the market outperformance
    else:
        market_bonus = 0

    # Combine all components with weights
    total_reward = (
        1.0 * basic_return +                    # Main objective
        0.3 * risk_adjusted_return +            # Risk adjustment
        1.0 * transaction_penalty +             # Transaction costs
        1.0 * drawdown_penalty +                # Drawdown control
        1.0 * stability_bonus +                 # Position stability
        0.5 * market_bonus                      # Market outperformance
    )

    # Clip extreme rewards to prevent instability
    total_reward = np.clip(total_reward, -0.1, 0.1)

    return total_reward


def reward_function_4(history,              # mandatory argument
                      window: int = 20,     # how many past steps for the Sharpe computation
                      drawdown_penalty: float = 0.1,
                      eps: float = 1e-8):
    """
    Risk-adjusted reward with draw-down penalty.

    Reward  =  Sharpe_t(window)  –  λ · current_drawdown

    where
    Sharpe_t(window)         = mean(R) / (std(R)+eps)   on the last `window` log returns
    current_drawdown         = (peak_equity - equity_t) / peak_equity
    λ (lambda)               = drawdown_penalty

    The function is robust to the first steps of an episode and
    to corner cases (zero variance, single step, etc.).
    """
    # We need at least one return to compute anything
    if len(history) < 2:
        return 0.0

    # ------------- 1. Compute log-returns -------------
    # history['portfolio_valuation'] gives the whole equity curve
    equity_curve = history['portfolio_valuation']

    # Ensure equity_curve is a numpy array
    equity_curve = np.array(equity_curve, dtype=np.float64)

    # Slice only the available part then take the last `window+1` points
    equity_curve = equity_curve[-(window + 1):]

    # If we have less than two points, again just return 0
    if equity_curve.shape[0] < 2:
        return 0.0

    log_returns = np.diff(np.log(equity_curve))

    # ------------- 2. Short-term Sharpe ratio -------------
    mean_r = np.mean(log_returns)
    std_r = np.std(log_returns)

    sharpe = mean_r / (std_r + eps)          # eps avoids /0

    # ------------- 3. Current draw-down -------------
    # Peak since the beginning of the episode
    all_equity = np.array(history['portfolio_valuation'], dtype=np.float64)
    running_peak = np.max(all_equity)
    current_equity = history['portfolio_valuation', -1]
    drawdown = (running_peak - current_equity) / \
        (running_peak + eps)  # eps avoids /0

    # ------------- 4. Final reward -------------
    reward = sharpe - drawdown_penalty * drawdown

    # Guard against NaNs that can occasionally appear in the very first steps
    if np.isnan(reward) or np.isinf(reward):
        return 0.0

    return float(reward)


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


def reward_function_5_less_risk_averse(
    history,
    window: int = 30,            # look-back for risk metrics
    r_free: float = 0.0,         # risk-free rate per step
    w_return: float = 1.0,       # weights for each component
    w_risk: float = 0.15,        # Lower risk aversion
    w_drawdown: float = 0.1,     # Lower drawdown penalty
    w_cost: float = 0.001,
    w_alpha: float = 0.50,
    clip_value: float = 1.0,
    eps: float = 1e-8
):
    """
    A less risk-averse version of reward_function_5.
    This version reduces the penalties for risk and drawdown,
    potentially encouraging the agent to explore more aggressive strategies.
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


def reward_function_5_aggressive(
    history,
    window: int = 30,
    r_free: float = 0.0,
    # — you may want to *reduce* these if you currently over-penalize churning
    w_return: float = 1.00,
    w_risk: float = 0.10,    # ← lower Sharpe penalty
    w_drawdown: float = 0.10,    # ← lower drawdown penalty
    w_cost: float = 0.000,   # ← essentially remove the “turnover cost”
    # — *new* bonus term for actually *trading*
    w_trade: float = 0.20,    # ← reward for any position change
    w_alpha: float = 0.50,
    clip_value: float = 1.0,
    eps: float = 1e-8,
):
    if len(history) < 2:
        return 0.0

    # 1) immediate log‐return
    curr = history["portfolio_valuation", -1]
    prev = history["portfolio_valuation", -2]
    r_t = np.log(curr/prev)

    # 2) Sharpe over last window
    vals = np.asarray(history["portfolio_valuation"], dtype=np.float64)
    rets = np.diff(np.log(vals[-(window+1):]))
    sharpe = 0.0
    if rets.size > 1:
        sharpe = (rets.mean() - r_free)/(rets.std()+eps)

    # 3) draw‐down
    peak = vals.max()
    drawdown = (peak - curr)/(peak + eps)

    # 4) turnover *cost* (we’re going to zero this out)
    pos_now = history["position", -1]
    pos_prev = history["position", -2]
    turnover = abs(pos_now - pos_prev)

    # 5) market‐outperformance
    alpha = 0.0
    if "data_close" in history.columns:
        m_ret = np.log(history["data_close", -1] / history["data_close", -2])
        alpha = r_t - m_ret

    # 6) NEW: bonus for trading
    trade_bonus = float(turnover > 0) * w_trade

    reward = (
        w_return * r_t
        + w_risk * sharpe
        - w_drawdown * drawdown
        - w_cost * turnover
        + w_alpha * alpha
        + trade_bonus
    )

    if not np.isfinite(reward):
        reward = 0.0
    return float(np.clip(reward, -clip_value, clip_value))


def reward_function_6(history):
    """
    A robust reward function that guides the model towards high-performance trading.
    It is based on a Sharpe-like ratio calculated over a rolling window, which
    balances returns against risk (volatility). It also includes a penalty
    for frequent trading.

    Args:
        history (History): The history object of the environment, providing access
                           to past portfolio valuations, positions, etc.

    Returns:
        float: The calculated reward for the current step.
    """
    # --- Parameters ---
    # Look back over the last N steps to calculate the Sharpe-like ratio.
    # A window of 20-50 is a reasonable starting point.
    sharpe_window = 30
    # A small penalty to discourage excessive trading.
    transaction_penalty_factor = 0.0001

    current_step = len(history)

    # --- 1. Handle edge cases at the start of an episode ---

    # At the very first step (or if history is empty), no reward can be calculated.
    if current_step < 2:
        return 0.0

    # --- 2. Calculate the transaction penalty for changing position ---

    transaction_penalty = 0.0
    # history['position', -1] is the action taken at the previous state that led to the current state.
    # history['position', -2] is the action taken before that.
    if history['position', -1] != history['position', -2]:
        transaction_penalty = transaction_penalty_factor

    # --- 3. Calculate the core reward component (profit vs. risk) ---

    # For the first few steps, use a simpler log return because there isn't enough
    # data for a stable Sharpe ratio calculation.
    if current_step <= sharpe_window:
        # Simple log return of portfolio valuation.
        # This is robust and avoids the user's reported error.
        try:
            val_now = float(history["portfolio_valuation", -1])
            val_before = float(history["portfolio_valuation", -2])

            # Avoid log(0) or division by zero.
            if val_before <= 0 or val_now <= 0:
                return -1.0  # Severe penalty for losing everything

            profit_reward = np.log(val_now / val_before)
        except (ValueError, TypeError):
            # If casting fails for any reason, return a neutral reward.
            profit_reward = 0.0
    else:
        # Use a Sharpe-like ratio for more mature episodes.

        # Get portfolio valuations for the window and ensure they are floats.
        # This is the key fix for the traceback error `TypeError: loop of ufunc does not support ...`
        valuations = np.asarray(
            history['portfolio_valuation', -sharpe_window:], dtype=np.float64)

        # Check for invalid portfolio values (e.g., bankruptcy).
        if np.any(valuations <= 0):
            return -1.0  # Penalize heavily for wiping out the portfolio.

        # Calculate log returns for each step in the window.
        log_returns = np.diff(np.log(valuations))

        # Ensure we have returns to process.
        if len(log_returns) == 0:
            return 0.0 - transaction_penalty

        # Calculate mean and standard deviation of log returns.
        mean_log_return = np.mean(log_returns)
        std_log_return = np.std(log_returns)

        # Calculate the Sharpe-like metric. Add a small epsilon to avoid division by zero.
        # This metric rewards higher average returns and penalizes higher volatility.
        sharpe_like_ratio = mean_log_return / (std_log_return + 1e-9)

        # Use tanh to squash the reward into a normalized range [-1, 1].
        # This improves training stability for many RL algorithms.
        profit_reward = np.tanh(sharpe_like_ratio)

    # --- 4. Combine profit component and transaction penalty ---

    final_reward = profit_reward - transaction_penalty

    # Ensure the final return value is a standard Python float.
    return float(final_reward)


def reward_function_7(history):
    """
    Robust reward function combining logarithmic returns with risk-adjusted metrics.
    Features:
    1. Handles portfolio valuation arrays safely
    2. Incorporates volatility penalty
    3. Includes drawdown penalty
    4. Ensures numerical stability
    """
    # Extract portfolio valuations as float array
    portfolio_valuations = np.array(
        history['portfolio_valuation'], dtype=np.float64)

    # Return 0 if insufficient data
    if len(portfolio_valuations) < 2:
        return 0.0

    # Calculate logarithmic returns safely
    current_value = portfolio_valuations[-1]
    previous_value = portfolio_valuations[-2]

    # Handle zero/negative values to avoid numerical issues
    if previous_value <= 0 or current_value <= 0:
        return -10  # Large penalty for invalid values

    log_return = np.log(current_value / previous_value)

    # Calculate volatility penalty (last 10 steps)
    returns = []
    for i in range(1, min(11, len(portfolio_valuations))):
        if portfolio_valuations[-i-1] > 0 and portfolio_valuations[-i] > 0:
            returns.append(
                np.log(portfolio_valuations[-i] / portfolio_valuations[-i-1]))

    volatility = np.std(returns) if returns else 0

    # Calculate drawdown penalty
    peak = np.max(portfolio_valuations)
    current_drawdown = (peak - current_value) / peak if peak > 0 else 0

    # Combine components (weights can be adjusted)
    reward = (
        log_return
        - 0.5 * volatility  # Volatility penalty
        - 2.0 * current_drawdown  # Drawdown penalty
    )

    return reward
