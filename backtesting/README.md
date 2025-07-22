# Backtesting Framework

Comprehensive system for comparing traditional trading strategies and deep reinforcement learning (DRL) models.

## Overview

This backtesting framework provides a complete solution for evaluating and comparing different trading strategies. It includes implementations of traditional baseline strategies, specialized backtesting for DRL models, and comprehensive visualization tools for performance analysis.

## Features

- **Traditional strategies** - Implementation of 5 baseline trading strategies
- **DRL model support** - Specialized backtesting for Stable-Baselines3 models
- **Comprehensive metrics** - 15+ performance indicators including risk-adjusted returns
- **Batch processing** - Automated processing of multiple data files
- **Advanced visualization** - Radar charts, heatmaps, and statistical analysis
- **Statistical testing** - Significance testing between strategies
- **Detailed logging** - Timestamped logs with error handling

## Components

### 1. Baseline Strategy Implementation (`baselines.py`)

Traditional trading strategies used as benchmarks:

- **BuyAndHold**: Benchmark strategy that buys at the start and holds until the end
- **SmaCross**: Trend-following strategy using Simple Moving Average crossover (10/20 periods)
- **RsiReversion**: Mean-reversion strategy using RSI oscillator (30/70 levels)
- **BollingerReversion**: Volatility-based mean-reversion using Bollinger Bands (20-period, 2-std)
- **DonchianBreakout**: Momentum/breakout strategy using Donchian Channels (20-period)

Automated batch processing for all baseline strategies with comprehensive reporting and error handling.

Specialized backtesting system for trained DRL models with LSTM state management and environment recreation.

Advanced visualization suite for strategy performance comparison with statistical analysis and multi-dimensional charts.

## Usage

### Baseline Strategy Backtesting

```bash
python 1-run-baselines.py "C:\path\to\your\data"
```

### DRL Model Backtesting

```bash
# Single file backtesting
python 2-rl-backtest.py --model best_model.zip --params best_params.json --data test_data.csv

# Batch processing (multiple files in a folder)
python 2-rl-backtest.py --model best_model.zip --params best_params.json --data_folder test_data_folder
```

### Results Visualization

```bash
# Process CSV files or folders
python 3-visualize.py strategy1.csv strategy2.csv --output-dir plots
python 3-visualize.py /path/to/csv/folder --output-dir plots

# Custom settings
python 3-visualize.py folder --figsize 16 10 --dpi 300 --output-dir high_res_plots
```

## Configuration

### Backtesting Parameters
- Initial cash: $100,000
- Commission: 0.2% per trade
- No slippage or margin requirements

## Requirements

### CSV File Format

All CSV files must contain the following columns:

- `date`: Timestamp (convertible to datetime)
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume

### For DRL Backtesting

Additional preprocessed columns required (generated during training):

- Normalized price features (`norm_close`, `norm_high`, etc.)
- Technical indicators (`norm_rsi`, `norm_macd`, etc.)
- Feature columns as specified in training parameters

## Key Performance Metrics

The system calculates comprehensive performance metrics including:

- **Return Metrics**: Total return, annualized return, CAGR
- **Risk Metrics**: Volatility, maximum drawdown, average drawdown
- **Risk-Adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Trading Efficiency**: Win rate, profit factor, average trade
- **Robustness**: Number of trades, SQN, expectancy

## Configuration

### Backtesting Parameters

- Initial cash: $100,000
- Commission: 0.2% per trade
- No slippage or margin requirements

### Logging

- All operations are logged with timestamps
- Logs saved to `logs/` directory
- Console and file output for real-time monitoring

## Error Handling

The system includes comprehensive error handling:

- File validation before processing
- Strategy-level error isolation
- Graceful degradation with detailed error logging
- Summary reports include success/failure statistics

## Best Practices

1. **Data Quality**: Ensure clean, continuous price data without gaps
2. **Folder Organization**: Use descriptive folder names (e.g., `1d-2015`, `4h-2020`)
3. **Parameter Tracking**: Keep DRL model parameters synchronized with training
4. **Regular Validation**: Compare results across different time periods
5. **Statistical Significance**: Use the built-in statistical tests to validate performance differences

## Output Interpretation

### Strategy Rankings

- Higher Sharpe ratio indicates better risk-adjusted returns
- Lower maximum drawdown indicates better risk management
- Higher win rate with reasonable trade frequency suggests consistency
- Statistical significance tests help validate performance differences

### Visualization Insights

- Radar charts show multi-dimensional strategy strengths/weaknesses
- Risk-return plots help identify efficient strategies
- Heatmaps reveal strategy performance across different market conditions
- Individual metric charts provide detailed performance breakdowns

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computations
- matplotlib/seaborn: Visualization
- backtesting: Backtesting framework
- stable-baselines3: DRL model loading (for RL backtesting)
- scipy: Statistical testing
- pathlib: File system operations

## Troubleshooting

Common issues and solutions:

1. **Column naming**: Ensure CSV columns match expected format
2. **Date parsing**: Verify date format is recognizable by pandas
3. **Memory usage**: For large datasets, process in batches
4. **Model compatibility**: Ensure DRL model and parameters are compatible
5. **Missing data**: Handle NaN values in technical indicators appropriately
