import gym_trading_env
import gymnasium as gym
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
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
        df["feature_close"] = df["close"]
        df["feature_volume"] = df["volume"]
        df["feature_high"] = df["high"]
        df["feature_low"] = df["low"]
        df["feature_open"] = df["open"]
        df["feature_macd"] = df["macd"]     # macd feature for trend detection
    except Exception as e:
        print(f"Error during preprocessing: {e}")
    return df


# Create training environment
train_env = gym.make('MultiDatasetTradingEnv',
                     dataset_dir='dataset/1d-2005/train/*.pkl',
                     reward_function=custom_reward_function,
                     preprocess=preprocess,
                     windows=20,  # Set window size for LSTM
                     # Allow short, neutral, and long positions
                     positions=[-1, 0, 1],
                     trading_fees=0.001,    # A realistic 0.1% fee for crypto exchanges
                     borrow_interest_rate=0.0003,  # A realistic daily borrow/funding rate
                     )

# Set seed for training environment
train_env.reset(seed=SEED)

# Create evaluation environment
eval_env = gym.make('MultiDatasetTradingEnv',
                    dataset_dir='dataset/1d-2005/val/*.pkl',  # Use validation dataset
                    reward_function=custom_reward_function,
                    preprocess=preprocess,
                    windows=20,
                    positions=[-1, 0, 1],
                    trading_fees=0.001,
                    borrow_interest_rate=0.0003,
                    )

eval_env.reset(seed=SEED)

# Create PPO model with seed
# Swap to "RecurrentPPO" to enable recurrent policies like LSTM
# Use "MlpLstmPolicy" for LSTM support, which is useful for trading tasks, where temporal dependencies are important
model = RecurrentPPO("MlpLstmPolicy",
                     train_env,
                     verbose=1,
                     tensorboard_log="./runs",
                     seed=SEED
                     )

# Set up early stopping callback
stop_callback = StopTrainingOnNoModelImprovement(
    max_no_improvement_evals=10, min_evals=5, verbose=1)

eval_callback = EvalCallback(eval_env,
                             best_model_save_path='./model/best/',
                             log_path='./eval_logs/',
                             eval_freq=10000,  # Evaluate every 10k steps
                             n_eval_episodes=5,
                             deterministic=True,
                             render=False,
                             callback_after_eval=stop_callback,
                             verbose=1)

# Train the model
print("Starting PPO training with early stopping...")
model.learn(total_timesteps=5000000,
            tb_log_name="recurrent_ppo", callback=eval_callback)

model.save("./model/recurrent_ppo_trading_model")
