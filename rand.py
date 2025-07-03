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
from reward import reward_function_1 as custom_reward_function

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

# Available in the github repo : examples/data/BTC_USD-Hourly.csv
url = "https://raw.githubusercontent.com/ClementPerroud/Gym-Trading-Env/main/examples/data/BTC_USD-Hourly.csv"
df = pd.read_csv(url, parse_dates=["date"], index_col="date")
df.sort_index(inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

env = gym.make("TradingEnv",
               name="BTCUSD",
               df=df,  # Your dataset with your custom features
               positions=[-1, 0, 1],  # -1 (=SHORT), 0(=OUT), +1 (=LONG)
               # 0.01% per stock buy / sell (Binance fees)
               trading_fees=0.01/100,
               # 0.0003% per timestep (one timestep = 1h here)
               borrow_interest_rate=0.0003/100,
               reward_function=custom_reward_function,  # Use custom reward function
               )

# Split data into training (70%) and evaluation (30%)
split_index = int(len(df) * 0.7)
train_df = df.iloc[:split_index].copy()
eval_df = df.iloc[split_index:].copy()

print(f"Training data: {len(train_df)} samples")
print(f"Evaluation data: {len(eval_df)} samples")

# Create evaluation environment
eval_env = gym.make("TradingEnv",
                    name="BTCUSD_Eval",
                    df=eval_df,
                    positions=[-1, 0, 1],
                    trading_fees=0.01/100,
                    borrow_interest_rate=0.0003/100,
                    reward_function=custom_reward_function,
                    )

# Set seed for evaluation environment
eval_env.reset(seed=SEED)

# Compare with random baseline
print("Running random baseline for comparison...")
# Reset with seed for reproducible random baseline
obs, info = eval_env.reset(seed=SEED)
done, truncated = False, False
random_reward = 0

while not done and not truncated:
    action = eval_env.action_space.sample()
    obs, reward, done, truncated, info = eval_env.step(action)
    random_reward += reward

print(f"Random baseline reward: {random_reward:.4f}")
