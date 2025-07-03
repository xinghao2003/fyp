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

# Available in the github repo : examples/data/BTC_USD-Hourly.csv
url = "https://raw.githubusercontent.com/ClementPerroud/Gym-Trading-Env/main/examples/data/BTC_USD-Hourly.csv"
df = pd.read_csv(url, parse_dates=["date"], index_col="date")
df.sort_index(inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Split data into training (70%) and evaluation (30%)
split_index = int(len(df) * 0.7)
train_df = df.iloc[:split_index].copy()
eval_df = df.iloc[split_index:].copy()

print(f"Training data: {len(train_df)} samples")
print(f"Evaluation data: {len(eval_df)} samples")

# Create training environment
train_env = gym.make("TradingEnv",
                     name="BTCUSD_Train",
                     df=train_df,
                     positions=[-1, 0, 1],
                     trading_fees=0.01/100,
                     borrow_interest_rate=0.0003/100,
                     reward_function=custom_reward_function,
                     )

# Set seed for training environment
train_env.reset(seed=SEED)

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

# Create PPO model with seed
model = PPO("MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log="./ppo_trading_tensorboard/",
            seed=SEED)

# Train the model
print("Starting PPO training...")
model.learn(total_timesteps=1000000)

# Evaluate the trained model
print("Evaluating trained model on test data...")
# Reset with seed for reproducible evaluation
obs, info = eval_env.reset(seed=SEED)
done, truncated = False, False
total_reward = 0
episode_rewards = []

while not done and not truncated:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = eval_env.step(action)
    total_reward += reward

print(f"Evaluation completed. Total reward: {total_reward:.4f}")
