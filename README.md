# Deep Reinforcement Learning Trading System

A comprehensive framework for developing, training, and evaluating deep reinforcement learning models for algorithmic trading.

## Overview

This Final Year Project (FYP) implements an end-to-end pipeline for algorithmic trading using deep reinforcement learning. The system downloads financial data, preprocesses it for ML applications, trains RecurrentPPO models, and provides comprehensive backtesting against traditional strategies. The framework supports multi-asset trading environments with advanced risk management and performance analysis.

## Features

- **Multi-asset support** - Trade stocks, ETFs, commodities, crypto, and forex
- **Advanced preprocessing** - 15+ technical indicators with bias-free normalization
- **Deep RL training** - RecurrentPPO with LSTM networks and hyperparameter optimization
- **Comprehensive backtesting** - Compare DRL models against 5 traditional strategies
- **Risk management** - Custom reward functions with risk-adjusted metrics
- **Visualization suite** - Performance radar charts, heatmaps, and statistical analysis
- **Docker environment** - Containerized development with VS Code integration
- **Production ready** - Scalable architecture with proper logging and error handling

## Project Structure

```
fyp/
├── docker/                    # Containerized development environment
│   ├── Dockerfile            # Python 3.11 with VS Code server
│   ├── startup.sh            # Automated setup script
│   └── README.md             # Docker usage instructions
├── download/                  # Data acquisition tools
│   ├── 1-check-yahoo-symbols.py      # Symbol validation
│   ├── 2-download-yahoo-stock-data.py # Historical data download
│   ├── tickers.json          # 60+ symbols across 8 asset classes
│   └── README.md             # Download tools documentation
├── preprocessing/             # 6-step ML data pipeline
│   ├── 1-add-indicators.py   # Technical indicators (MACD, RSI, etc.)
│   ├── 2-normalization.py    # Feature-specific normalization
│   ├── 3-date-cap.py         # Date range filtering
│   ├── 4-cleaning.py         # NaN handling
│   ├── 5-split-fix.py        # Temporal train/val/test splits
│   ├── 6-prepare-gym-compatible-data.py # Gym environment formatting
│   └── README.md             # Preprocessing pipeline guide
├── backtesting/               # Strategy evaluation framework
│   ├── baselines.py          # 5 traditional trading strategies
│   ├── 1-run-baselines.py    # Baseline strategy backtesting
│   ├── 2-rl-backtest.py      # DRL model backtesting
│   ├── 3-visualize.py        # Performance visualization
│   └── README.md             # Backtesting framework guide
├── docs/                      # Documentation and guides
├── reward.py                  # Custom reward functions
├── rl-multi.py               # Multi-dataset RL training
├── rppo-mul-opt.py           # Hyperparameter optimization
├── toolkit.ipynb            # Development notebook
├── requirements.txt          # Python dependencies
└── LICENSE                   # MIT License
```

## Datasets and Pre-trained Models

### Example Dataset and Results

**🤗 Hugging Face Dataset**: [xinghao2003/fyp](https://huggingface.co/datasets/xinghao2003/fyp)

This repository includes preprocessed datasets, trained models, and comprehensive backtesting results:

- **Preprocessed Data** - Ready-to-use datasets for 16 representative symbols across all asset classes
- **Trained Models** - Pre-trained RecurrentPPO models with optimized hyperparameters  
- **Backtesting Results** - Complete performance analysis comparing DRL models against traditional strategies
- **Visualization Outputs** - Performance charts, heatmaps, and statistical analysis

The dataset demonstrates the complete end-to-end pipeline from raw financial data to trained models with backtesting results, making it easy to reproduce the research and build upon the work.

## Quick Start

### Option A: Use Pre-trained Models (Recommended for Quick Start)

```bash
# Download example dataset and models from Hugging Face
# Visit: https://huggingface.co/datasets/xinghao2003/fyp

# Extract and use pre-trained models for backtesting
cd backtesting/
python 2-rl-backtest.py --model downloaded_model.zip --params downloaded_params.json --data_folder test_data/
```

### Option B: Full Pipeline Setup

### 1. Environment Setup

**Docker (Recommended)**

```bash
# Build and run development container
cd docker/
docker build -t fyp-trading .
docker run -d --name fyp-dev fyp-trading
docker logs -f fyp-dev
# Follow GitHub device login instructions in logs
# Access via browser: https://vscode.dev/tunnel/fyp
```

**Local Installation**

```bash
# Clone repository
git clone <repository-url>
cd fyp

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Acquisition

```bash
# Validate symbols
cd download/
python 1-check-yahoo-symbols.py tickers.json

# Download historical data (daily, maximum period)
python 2-download-yahoo-stock-data.py tickers.json --period max --interval 1d
```

### 3. Data Preprocessing

```bash
cd preprocessing/
# Run complete 6-step pipeline
python 1-add-indicators.py ../download/data/
python 2-normalization.py ../download/data/
python 3-date-cap.py ../download/data/
python 4-cleaning.py ../download/data/
python 5-split-fix.py ../download/data/
python 6-prepare-gym-compatible-data.py ../download/data/
```

### 4. Model Training

```bash
# Train RecurrentPPO model
python rl-multi.py

# Hyperparameter optimization with Optuna
python rppo-mul-opt.py
```

### 5. Strategy Evaluation

```bash
cd backtesting/
# Run baseline strategies
python 1-run-baselines.py ../preprocessing/data/test/

# Backtest trained RL model
python 2-rl-backtest.py --model best_model.zip --params best_params.json --data_folder ../preprocessing/data/test/

# Generate comparison visualizations
python 3-visualize.py results/ --output-dir plots/
```

### Option C: Reproducible `uv` Environment (Preferred)

We maintain `pyproject.toml` and `uv.lock` so you can rely entirely on [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for dependency installation; nothing else is required. Follow the upstream installation guide to install `uv`, then execute:

1. **Verify `uv` is available on your PATH**  

   ```bash
   uv --version
   ```

   This step confirms that you followed the official installation instructions and ensures `uv` can run commands.
2. **Let `uv` set up the environment**  

   ```bash
   uv sync
   ```

   `uv sync` reads `pyproject.toml` and `uv.lock` so every transitive dependency is locked; the result works the same on every machine.
3. **Activate the environment**  

`uv` also respects the `.python-version` hint (Python 3.11), so you can skip manual Python installs when that matches your system. This pure `uv` workflow is now our default recommendation for new contributors.

## Components

### Data Pipeline

1. **Download Tools** - Acquire OHLCV data from Yahoo Finance for 60+ symbols
2. **Preprocessing** - 6-step pipeline adding technical indicators and normalization
3. **Quality Assurance** - NaN handling, date filtering, and temporal consistency

### Machine Learning

1. **Environment** - Multi-dataset trading environment with gym interface
2. **Model** - RecurrentPPO with LSTM networks for sequential decision making
3. **Optimization** - Optuna-based hyperparameter tuning with pruning
4. **Rewards** - Risk-adjusted reward functions with volatility penalties

### Evaluation

1. **Baseline Strategies** - 5 traditional strategies (Buy&Hold, SMA Cross, RSI, etc.)
2. **Performance Metrics** - 15+ metrics including Sharpe ratio, max drawdown, win rate
3. **Statistical Testing** - Significance tests for strategy comparison
4. **Visualization** - Comprehensive charts and performance analysis

## Configuration

### Key Parameters

```python
# Training Configuration
LOOKBACK_WINDOW = 60        # LSTM sequence length
TOTAL_TIMESTEPS = 1000000   # Training steps
LEARNING_RATE = 0.0003      # PPO learning rate
BATCH_SIZE = 64             # Training batch size

# Environment Settings
INITIAL_CASH = 100000       # Starting portfolio value
COMMISSION = 0.002          # 0.2% per trade
MAX_POSITIONS = 1           # Single position limit

# Data Splits
TRAIN_END = "2020-12-31"    # Training data cutoff
VAL_END = "2024-12-31"      # Validation data cutoff
TEST_START = "2025-01-01"   # Test data start
```

### Asset Classes

The system supports 8 asset categories:

- U.S. Equity Benchmarks (SPY, QQQ, IWM, DIA)
- Sector/Thematic ETFs (XLB, XLE, XLF, XLK, XLV, etc.)
- Large-Cap Stocks (AAPL, MSFT, GOOGL, AMZN, etc.)
- Fixed-Income (TLT, IEF, LQD, HYG)
- Commodities (GLD, SLV, USO, DBA)
- Volatility (^VIX, VXX)
- Currencies (EUR/USD, USD/JPY, GBP/USD)
- Cryptocurrencies (BTC-USD, ETH-USD, SOL-USD)

## Performance Metrics

The system calculates comprehensive performance statistics:

### Return Metrics

- Total Return, Annualized Return, CAGR
- Risk-free rate adjusted returns

### Risk Metrics

- Volatility (annualized standard deviation)
- Maximum Drawdown, Average Drawdown
- Value at Risk (VaR), Conditional VaR

### Risk-Adjusted Metrics

- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Information Ratio, Treynor Ratio

### Trading Metrics

- Win Rate, Profit Factor, Expectancy
- Average Trade, Best/Worst Trade
- Number of Trades, SQN Score

## Results and Visualization

The framework generates comprehensive visualizations:

1. **Performance Radar Charts** - Multi-dimensional strategy comparison
2. **Risk-Return Scatter Plots** - Efficient frontier analysis
3. **Performance Heatmaps** - Strategy performance across different assets
4. **Statistical Significance Tests** - Validate performance differences
5. **Equity Curves** - Portfolio value progression over time
6. **Drawdown Analysis** - Risk assessment and recovery periods

## Development Environment

### Docker Setup

- Python 3.11 with optimized dependencies
- VS Code Server with tunnel access for remote development
- Git integration with automatic repository updates
- Persistent workspace with volume mounting

### Local Development

- Jupyter notebook support for interactive development
- Comprehensive logging with timestamped files
- Error handling and graceful degradation
- Progress tracking and batch processing capabilities

## Dependencies

### Core Libraries

```
yfinance==0.2.65              # Financial data download
stockstats==0.6.5             # Technical indicators
stable-baselines3[extra]==2.6.0  # Deep RL algorithms
sb3-contrib==2.6.0            # Additional RL algorithms
gym-trading-env==0.3.5        # Trading environment
optuna==4.4.0                 # Hyperparameter optimization
backtesting==0.6.4            # Strategy backtesting
matplotlib==3.10.3            # Plotting
seaborn==0.13.2              # Statistical visualization
```

## Research Applications

This framework supports various research directions:

1. **Alternative Architectures** - Transformer-based models, attention mechanisms
2. **Multi-Agent Systems** - Portfolio allocation across multiple strategies
3. **Risk Management** - Dynamic position sizing, stop-loss optimization
4. **Market Regime Detection** - Adaptive strategies for different market conditions
5. **Feature Engineering** - Alternative data sources, sentiment analysis
6. **Execution Modeling** - Market impact, slippage, and realistic transaction costs

## Best Practices

### Data Quality

- Validate symbols before downloading
- Handle missing data and corporate actions
- Ensure temporal consistency across datasets
- Prevent lookahead bias in preprocessing

### Model Development

- Use proper train/validation/test splits
- Implement early stopping and model checkpointing
- Monitor overfitting with validation metrics
- Test on multiple market regimes

### Backtesting

- Use realistic transaction costs and slippage
- Account for survivorship bias
- Test across multiple time periods
- Compare against appropriate benchmarks

### Production Deployment

- Implement proper error handling and logging
- Use configuration management for parameters
- Monitor model performance in production
- Have rollback procedures for model failures

## Contributing

1. Follow the established code structure and documentation standards
2. Add comprehensive tests for new features
3. Update relevant README files
4. Use consistent coding style and naming conventions
5. Document any new dependencies or configuration changes

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Yahoo Finance for financial data access
- Stable-Baselines3 for deep RL implementations
- Optuna for hyperparameter optimization
- The open-source trading and ML community

---

**Note**: This system is for educational and research purposes. Past performance does not guarantee future results. Always conduct thorough testing before deploying trading strategies with real capital.
