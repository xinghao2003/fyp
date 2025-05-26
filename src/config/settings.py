"""
General project settings, API keys (use .env for sensitive data).
To use .env file for API keys:
1. Create a .env file in the project_root directory.
2. Add your API keys like: ALPHA_VANTAGE_API_KEY='YOUR_KEY'
3. In this file, load them using: 
   from dotenv import load_dotenv
   import os
   load_dotenv()
   ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
"""


# Load environment variables from .env
from dotenv import load_dotenv
import os
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Data paths
RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"

# Model paths
MODEL_SAVE_DIR = "checkpoints/"
LOG_DIR = "logs/"

# Data Preprocessing Settings
NORMALIZATION_CONFIG = {
    'enable_normalization': True,
    # 'percentage_change', 'minmax', 'standard', 'robust'
    'normalization_method': 'percentage_change',
    'add_market_agnostic_features': True,
    'feature_range': (0, 1),  # For MinMaxScaler
}

# Trading Environment Settings
TRADING_ENV_CONFIG = {
    'initial_cash': 1000.0,
    'transaction_cost_pct': 0.001,
    'window_size': 10,
    'action_space_type': 'multidiscrete',
}

# Training Settings
TRAINING_CONFIG = {
    'default_seed': 42,
    'enable_tensorboard': True,
    'save_freq': 10000,
}

pass
