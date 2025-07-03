import gym_trading_env
import gymnasium as gym
from gym_trading_env.downloader import download
import datetime
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import random
import os
from reward import reward_function_5 as custom_reward_function

# Set seeds for reproducibility
SEED = 42

# Set Python random seed
random.seed(SEED)

# Set NumPy random seed
np.random.seed(SEED)

# Set environment variable for Python hash randomization
os.environ['PYTHONHASHSEED'] = str(SEED)

# For PyTorch (if used by stable-baselines3 internally)
try:
    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except ImportError:
    pass

print(f"All random seeds set to: {SEED}")

# Custom preprocessing function


def preprocess(df: pd.DataFrame):
    # Create your features
    try:
        df["feature_macd"] = df["macd"]
    except Exception as e:
        print(f"Error during preprocessing: {e}")
    return df


# Create training environment
train_env = gym.make('MultiDatasetTradingEnv',
                     dataset_dir='dataset/1d-2005/*.pkl',
                     reward_function=custom_reward_function,
                     preprocess=preprocess,
                     # Allow short, neutral, and long positions
                     positions=[-1, 0, 1],
                     trading_fees=0.001,    # A realistic 0.1% fee for crypto exchanges
                     borrow_interest_rate=0.0003,  # A realistic daily borrow/funding rate
                     )

# Set seed for training environment
train_env.reset(seed=SEED)

# Create PPO model with seed
model = PPO("MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log="./ppo_trading_tensorboard/",
            seed=SEED)

# Train the model
print("Starting PPO training...")
model.learn(total_timesteps=3000000, tb_log_name="ppo_trading_custom_features")

model.save("ppo_trading_model")
