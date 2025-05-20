"""
Utilities for plotting results (e.g., equity curves, performance charts).
"""


import matplotlib.pyplot as plt
import numpy as np


def plot_trading_evaluation(price_series, actions, portfolio_values, action_labels=None, title='Trading Evaluation', save_path=None):
    """
    Plots the stock price with action markers and the portfolio value (profit curve) below.
    price_series: pd.Series or np.ndarray of prices
    actions: list/array of actions (0=hold, 1=buy, 2=sell)
    portfolio_values: list/array of portfolio values
    action_labels: optional, list of action names
    """
    import pandas as pd
    if action_labels is None:
        action_labels = ['Hold', 'Buy', 'Sell']
    price_series = np.asarray(price_series)
    actions = np.asarray(actions)
    portfolio_values = np.asarray(portfolio_values)
    steps = np.arange(len(price_series))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                   sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Plot price
    ax1.plot(steps, price_series, label='Price', color='black')
    # Mark actions
    for action_type, marker, color in zip([1, 2], ['^', 'v'], ['green', 'red']):
        idxs = np.where(actions == action_type)[0]
        ax1.scatter(idxs, price_series[idxs], marker=marker, color=color,
                    label=action_labels[action_type], s=60, alpha=0.7)
    ax1.set_ylabel('Stock Price')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True)

    # Plot portfolio value
    ax2.plot(steps, portfolio_values, label='Portfolio Value', color='blue')
    ax2.set_ylabel('Portfolio Value')
    ax2.set_xlabel('Step')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_equity_curve(portfolio_values, title='Equity Curve'):
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.show()

# def plot_performance_summary(metrics_dict):
#     """Plots a summary of performance metrics."""
#     # Placeholder for plotting various metrics
#     pass


pass
