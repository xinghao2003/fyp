"""
Automated DQN hyperparameter tuning using Optuna.
Switch between auto and manual mode with a flag.
"""
import os
import logging
import optuna
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from src.drl_environment.trading_env import TradingEnv
from src.models.agents.dqn_agent import DQNAgent


def tune_dqn(env=None, n_trials=20, tensorboard_log_dir=None):
    """
    Run Optuna hyperparameter tuning for DQNAgent. If env is None, loads default training env.
    Returns best hyperparameters as a dict.
    """
    def objective_with_env(trial):
        local_env = env
        if local_env is None:
            train_data_path = os.path.join(os.path.dirname(
                __file__), '../../data/processed/AAPL_alpha_vantage_train.csv')
            df = pd.read_csv(train_data_path)
            local_env = TradingEnv(df)

        learning_rate = trial.suggest_float(
            'learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
        gamma = trial.suggest_float('gamma', 0.90, 0.999)
        exploration_fraction = trial.suggest_float(
            'exploration_fraction', 0.05, 0.5)
        exploration_final_eps = trial.suggest_float(
            'exploration_final_eps', 0.01, 0.2)
        total_timesteps = trial.suggest_int('total_timesteps', 10000, 200000)

        model = DQN(
            'MlpPolicy',
            local_env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=gamma,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            verbose=0,
            tensorboard_log=tensorboard_log_dir
        )
        model.learn(total_timesteps=total_timesteps,
                    reset_num_timesteps=True, tb_log_name="DQN")

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

    study = optuna.create_study(direction='maximize')

    def print_callback(study, trial):
        print(
            f"Trial {trial.number}: Value={trial.value}, Params={trial.params}")

    study.optimize(objective_with_env, n_trials=n_trials,
                   n_jobs=-1, show_progress_bar=True, callbacks=[print_callback])
    logging.info(f"Best trial: {study.best_trial.params}")
    return study.best_trial.params


def run_agent_training(auto_tune=False, n_trials=20):
    """
    For backward compatibility: runs manual or auto-tune DQN agent training.
    """
    if auto_tune:
        best_params = tune_dqn(n_trials=n_trials)
        logging.info(f"Best hyperparameters: {best_params}")
        return best_params
    else:
        train_data_path = os.path.join(os.path.dirname(
            __file__), '../../data/processed/AAPL_alpha_vantage_train.csv')
        df = pd.read_csv(train_data_path)
        env = TradingEnv(df)
        agent = DQNAgent(env)
        agent.train(total_timesteps=10000)
        agent.save('dqn_agent_manual.pth')
        logging.info("Manual training complete.")
        print('Manual training complete.')
        agent.save('dqn_agent_manual.pth')
        logging.info("Manual training complete.")
        print('Manual training complete.')
