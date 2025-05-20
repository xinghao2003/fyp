"""
Automated PPO hyperparameter tuning using Optuna.
Switch between auto and manual mode with a flag.
"""
import os
import optuna
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from src.drl_environment.trading_env import TradingEnv
from src.models.agents.ppo_agent import PPOAgent


def objective(trial):
    # Hyperparameter search space
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    n_steps = trial.suggest_categorical('n_steps', [64, 128, 256, 512, 1024])
    gamma = trial.suggest_float('gamma', 0.90, 0.999)
    ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    total_timesteps = trial.suggest_int('total_timesteps', 10000, 200000)

    # Load data
    train_data_path = os.path.join(os.path.dirname(
        __file__), '../../data/processed/AAPL_alpha_vantage_train.csv')
    df = pd.read_csv(train_data_path)
    env = TradingEnv(df)

    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        ent_coef=ent_coef,
        batch_size=batch_size,
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps)

    # Evaluate (simple: final total asset)
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        if np.isscalar(action):
            action = np.array([action, 1], dtype=int)
        obs, reward, done, *_ = env.step(action)
        total_reward += reward
    return total_reward


def tune_ppo(env=None, n_trials=20):
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
            from src.drl_environment.trading_env import TradingEnv
            local_env = TradingEnv(df)
        # Hyperparameter search space
        learning_rate = trial.suggest_float(
            'learning_rate', 1e-5, 1e-2, log=True)
        n_steps = trial.suggest_categorical(
            'n_steps', [64, 128, 256, 512, 1024])
        gamma = trial.suggest_float('gamma', 0.90, 0.999)
        ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-2, log=True)
        batch_size = trial.suggest_categorical(
            'batch_size', [8, 16, 32, 64])
        total_timesteps = trial.suggest_int('total_timesteps', 10000, 200000)
        from stable_baselines3 import PPO
        model = PPO(
            'MlpPolicy',
            local_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            ent_coef=ent_coef,
            batch_size=batch_size,
            verbose=0,
        )
        model.learn(total_timesteps=total_timesteps)
        # Evaluate (simple: final total asset)
        obs, _ = local_env.reset()
        done = False
        total_reward = 0
        import numpy as np
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if np.isscalar(action):
                action = np.array([action, 1], dtype=int)
            obs, reward, done, *_ = local_env.step(action)
            total_reward += reward
        return total_reward

    import optuna
    from optuna_integration import TensorBoardCallback
    tb_callback = TensorBoardCallback(
        "optuna_tensorboard_logs/ppo", metric_name="reward")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_with_env, n_trials=n_trials,
                   n_jobs=-1, show_progress_bar=True, callbacks=[tb_callback])
    print('Best trial:', study.best_trial.params)
    return study.best_trial.params


def run_agent_training(auto_tune=False, n_trials=20):
    """
    For backward compatibility: runs manual or auto-tune PPO agent training.
    """
    if auto_tune:
        best_params = tune_ppo(n_trials=n_trials)
        print('Best hyperparameters:', best_params)
        return best_params
    else:
        train_data_path = os.path.join(os.path.dirname(
            __file__), '../../data/processed/AAPL_alpha_vantage_train.csv')
        df = pd.read_csv(train_data_path)
        from src.drl_environment.trading_env import TradingEnv
        env = TradingEnv(df)
        agent = PPOAgent(env)
        agent.train(total_timesteps=10000)
        agent.save('ppo_agent_manual.pth')
        print('Manual training complete.')
