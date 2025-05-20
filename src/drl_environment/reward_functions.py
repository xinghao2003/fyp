# reward_functions.py
# Reward function utilities for trading environments

def simple_profit_reward(env, prev_asset):
    """
    Reward is the change in total asset value.
    """
    return env.total_asset - prev_asset


# More reward functions can be added here (e.g., risk-adjusted, Sharpe, etc.)
"""
Implementation of reward functions (e.g., Sharpe ratio-based).
"""

# Placeholder for reward function implementations


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculates the Sharpe ratio."""
    # Placeholder implementation
    return 0.0


pass
