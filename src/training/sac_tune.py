"""
Automated SAC hyperparameter tuning using Optuna.
Switch between auto and manual mode with a flag.
"""
import os
import optuna
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from src.drl_environment.trading_env import TradingEnv
from src.models.agents.sac_agent import SACAgent


def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    gamma = trial.suggest_uniform('gamma', 0.90, 0.999)
    tau = trial.suggest_uniform('tau', 0.005, 0.05)
    ent_coef = trial.suggest_loguniform('ent_coef', 1e-8, 1e-2)
    total_timesteps = trial.suggest_int('total_timesteps', 10000, 200000)

    train_data_path = os.path.join(os.path.dirname(
        __file__), '../../data/processed/AAPL_alpha_vantage_train.csv')
    df = pd.read_csv(train_data_path)
    env = TradingEnv(df)

    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        ent_coef=ent_coef,
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps)

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


def tune_sac(env=None, n_trials=20):
    """
    Run Optuna hyperparameter tuning for SACAgent. If env is None, loads default training env.
    Returns best hyperparameters as a dict.
    """
    def objective_with_env(trial):
        local_env = env
        if local_env is None:
            train_data_path = os.path.join(os.path.dirname(
                __file__), '../../data/processed/AAPL_alpha_vantage_train.csv')
            df = pd.read_csv(train_data_path)
            from src.drl_environment.trading_env import TradingEnv
            local_env = TradingEnv(df)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        batch_size = trial.suggest_categorical(
            'batch_size', [8, 16, 32, 64])
        gamma = trial.suggest_uniform('gamma', 0.90, 0.999)
        tau = trial.suggest_uniform('tau', 0.005, 0.05)
        ent_coef = trial.suggest_loguniform('ent_coef', 1e-8, 1e-2)
        total_timesteps = trial.suggest_int('total_timesteps', 10000, 200000)
        from stable_baselines3 import SAC
        model = SAC(
            'MlpPolicy',
            local_env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            ent_coef=ent_coef,
            verbose=0,
        )
        model.learn(total_timesteps=total_timesteps)
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
        "optuna_tensorboard_logs", metric_name="reward")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_with_env, n_trials=n_trials,
                   n_jobs=-1, show_progress_bar=True, callbacks=[tb_callback])
    print('Best trial:', study.best_trial.params)
    return study.best_trial.params


def run_agent_training(auto_tune=False, n_trials=20):
    if auto_tune:
        best_params = tune_sac(n_trials=n_trials)
        print('Best hyperparameters:', best_params)
        return best_params
    else:
        train_data_path = os.path.join(os.path.dirname(
            __file__), '../../data/processed/AAPL_alpha_vantage_train.csv')
        df = pd.read_csv(train_data_path)
        from src.drl_environment.trading_env import TradingEnv
        env = TradingEnv(df)
        agent = SACAgent(env)
        agent.train(total_timesteps=10000)
        agent.save('sac_agent_manual.pth')
        print('Manual training complete.')
