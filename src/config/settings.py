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

pass
