import os
import logging
import optuna
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from src.drl_environment.trading_env import TradingEnv
import time


def tune_ppo(env=None, n_trials=20, tensorboard_log_dir=None):
    """
    Run Optuna hyperparameter tuning for PPOAgent. If env is None, loads default training env.
    Returns best hyperparameters as a dict.
    """
    def objective_with_env(trial):
        # Use provided env if given, else create new
        local_env = env
        if local_env is None:
            train_data_path = os.path.join(os.path.dirname(
                __file__), '../../data/processed/AAPL_alpha_vantage_train.csv')
            df = pd.read_csv(train_data_path)
            local_env = TradingEnv(df)

        # Hyperparameter search space
        learning_rate = trial.suggest_float(
            'learning_rate', 1e-5, 1e-2, log=True)
        n_steps = trial.suggest_categorical(
            'n_steps', [64, 128, 256, 512, 1024])
        gamma = trial.suggest_float('gamma', 0.90, 0.999)
        ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
        total_timesteps = trial.suggest_int('total_timesteps', 10000, 200000)

        model = PPO(
            'MlpPolicy',
            local_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            ent_coef=ent_coef,
            batch_size=batch_size,
            verbose=0,
            tensorboard_log=tensorboard_log_dir
        )
        model.learn(total_timesteps=total_timesteps,
                    reset_num_timesteps=True, progress_bar=True)

        # Evaluate (simple: final total asset)
        obs, _ = local_env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = action.tolist()
            obs, reward, done, *_ = local_env.step(action)
            total_reward += reward
        return total_reward

    study = optuna.create_study(
        direction='maximize', study_name=f"tune_ppo-{time.strftime('%Y%m%d-%H%M%S')}", storage="sqlite:///tune_ppo.db", load_if_exists=True)
    study.optimize(objective_with_env, n_trials=n_trials,
                   n_jobs=1, show_progress_bar=True)
    logging.info(f"Best trial: {study.best_trial.params}")
    return study.best_trial.params
