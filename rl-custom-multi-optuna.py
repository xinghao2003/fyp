import gym_trading_env
import gymnasium as gym
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
import numpy as np
import random
import os
from datetime import datetime
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import logging
import shutil

# Import different reward functions
from reward import (
    reward_function_5_less_risk_averse as reward_func_1,
    reward_function_5_aggressive as reward_func_2,
    reward_function_5 as reward_func_3,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Generate unique timestamp-based ID for this run
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Please record this ID for tracking: {RUN_ID}")

# Set seeds for reproducibility
SEED = 42


def set_seeds(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


set_seeds(SEED)

# Custom preprocessing function


def create_preprocess_function(feature_config):
    """Create a preprocessing function based on feature configuration"""
    def preprocess(df: pd.DataFrame):
        # Create your features based on the configuration
        try:
            # Basic price features (always included for trading)
            df["feature_close"] = df["close"]

            # Optional features based on trial suggestions
            if feature_config.get('use_volume', True):
                df["feature_volume"] = df["volume"]

            if feature_config.get('use_high', True):
                df["feature_high"] = df["high"]

            if feature_config.get('use_low', True):
                df["feature_low"] = df["low"]

            if feature_config.get('use_open', True):
                df["feature_open"] = df["open"]

            # Technical indicators
            if feature_config.get('use_macd', True):
                # macd feature for trend detection
                df["feature_macd"] = df["macd"]

            if feature_config.get('use_rsi', False):
                if 'rsi' in df.columns:
                    df["feature_rsi"] = df["rsi"]

            if feature_config.get('use_sma', False):
                if 'close_10_sma' in df.columns:
                    df["feature_sma"] = df["close_10_sma"]

            if feature_config.get('use_ema', False):
                if 'close_10_ema' in df.columns:
                    df["feature_ema"] = df["close_10_ema"]

            if feature_config.get('use_adx', False):
                if 'adx' in df.columns:
                    df["feature_adx"] = df["adx"]

            if feature_config.get('use_bb_upper', False):
                if 'boll_ub' in df.columns:
                    df["feature_bb_upper"] = df["boll_ub"]

            if feature_config.get('use_bb_lower', False):
                if 'boll_lb' in df.columns:
                    df["feature_bb_lower"] = df["boll_lb"]

            if feature_config.get('use_bb_middle', False):
                if 'boll' in df.columns:
                    df["feature_bb_middle"] = df["boll"]

            if feature_config.get('use_stoch_k', False):
                if 'kdjk' in df.columns:
                    df["feature_stoch_k"] = df["kdjk"]

            if feature_config.get('use_stoch_d', False):
                if 'kdjd' in df.columns:
                    df["feature_stoch_d"] = df["kdjd"]

            if feature_config.get('use_stoch_j', False):
                if 'kdjj' in df.columns:
                    df["feature_stoch_j"] = df["kdjj"]

            if feature_config.get('use_atr', False):
                if 'atr' in df.columns:
                    df["feature_atr"] = df["atr"]

        except Exception as e:
            print(f"Error during preprocessing: {e}")
        return df

    return preprocess


def get_reward_function(trial):
    """Select reward function based on trial suggestion"""
    reward_choice = trial.suggest_categorical(
        'reward_function', ['risk_averse', 'aggressive', 'balanced'])

    reward_functions = {
        'risk_averse': reward_func_1,
        'aggressive': reward_func_2,
        'balanced': reward_func_3,
    }

    return reward_functions[reward_choice]


def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    train_env = None
    eval_env = None
    trial_dir = None

    try:
        # Suggest hyperparameters
        learning_rate = trial.suggest_float(
            'learning_rate', 1e-5, 1e-2, log=True)
        n_steps = trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096])
        batch_size = trial.suggest_categorical(
            'batch_size', [32, 64, 128, 256])
        n_epochs = trial.suggest_int('n_epochs', 5, 20)
        gamma = trial.suggest_float('gamma', 0.9, 0.9999)
        gae_lambda = trial.suggest_float('gae_lambda', 0.8, 0.99)
        clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
        ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-1, log=True)
        vf_coef = trial.suggest_float('vf_coef', 0.1, 1.0)

        # PPO-specific hyperparameters
        windows = trial.suggest_int('windows', 10, 30)
        trading_fees = trial.suggest_float('trading_fees', 0.0005, 0.002)
        borrow_interest_rate = trial.suggest_float(
            'borrow_interest_rate', 0.0001, 0.0005)

        # Feature selection hyperparameters
        # Note: 'close' price is always included as it's essential for trading
        # Other features are optional and will be optimized by Optuna
        feature_config = {
            # Basic OHLCV features (volume, high, low, open are optional)
            'use_volume': trial.suggest_categorical('use_volume', [True, False]),
            'use_high': trial.suggest_categorical('use_high', [True, False]),
            'use_low': trial.suggest_categorical('use_low', [True, False]),
            'use_open': trial.suggest_categorical('use_open', [True, False]),

            # Momentum Indicators
            # Relative Strength Index
            'use_rsi': trial.suggest_categorical('use_rsi', [True, False]),
            # Moving Average Convergence Divergence
            'use_macd': trial.suggest_categorical('use_macd', [True, False]),
            # Stochastic Oscillator %K
            'use_stoch_k': trial.suggest_categorical('use_stoch_k', [True, False]),
            # Stochastic Oscillator %D
            'use_stoch_d': trial.suggest_categorical('use_stoch_d', [True, False]),
            # Stochastic Oscillator %J
            'use_stoch_j': trial.suggest_categorical('use_stoch_j', [True, False]),

            # Trend Indicators
            # Simple Moving Average
            'use_sma': trial.suggest_categorical('use_sma', [True, False]),
            # Exponential Moving Average
            'use_ema': trial.suggest_categorical('use_ema', [True, False]),
            # Average Directional Index
            'use_adx': trial.suggest_categorical('use_adx', [True, False]),

            # Volatility Indicators
            # Bollinger Bands Upper
            'use_bb_upper': trial.suggest_categorical('use_bb_upper', [True, False]),
            # Bollinger Bands Lower
            'use_bb_lower': trial.suggest_categorical('use_bb_lower', [True, False]),
            # Bollinger Bands Middle
            'use_bb_middle': trial.suggest_categorical('use_bb_middle', [True, False]),
            # Average True Range
            'use_atr': trial.suggest_categorical('use_atr', [True, False]),
        }

        # Create preprocessing function with selected features
        preprocess_func = create_preprocess_function(feature_config)

        # Log selected features for this trial
        selected_features = [feature for feature, enabled in feature_config.items()
                             if enabled and not feature.endswith('_window')]
        logger.info(
            f"Trial {trial.number}: Selected features: {selected_features}")

        # Get reward function
        custom_reward_function = get_reward_function(trial)

        # Create trial-specific directory
        trial_dir = f"./optuna_trials/{RUN_ID}/trial_{trial.number}"
        os.makedirs(trial_dir, exist_ok=True)

        # Training environment
        train_env = gym.make('MultiDatasetTradingEnv',
                             dataset_dir='dataset/1d-2005/train/*.pkl',
                             reward_function=custom_reward_function,
                             preprocess=preprocess_func,
                             windows=windows,
                             positions=[-1, 0, 1],
                             trading_fees=trading_fees,
                             borrow_interest_rate=borrow_interest_rate,
                             )

        train_env.reset(seed=SEED + trial.number)

        # Create evaluation environment
        eval_env = gym.make('MultiDatasetTradingEnv',
                            dataset_dir='dataset/1d-2005/val/*.pkl',
                            reward_function=custom_reward_function,
                            preprocess=preprocess_func,
                            windows=windows,
                            positions=[-1, 0, 1],
                            trading_fees=trading_fees,
                            borrow_interest_rate=borrow_interest_rate,
                            )

        eval_env.reset(seed=SEED + trial.number)
        eval_env = Monitor(eval_env)

        # Create model with suggested hyperparameters
        model = RecurrentPPO("MlpLstmPolicy",
                             train_env,
                             learning_rate=learning_rate,
                             n_steps=n_steps,
                             batch_size=batch_size,
                             n_epochs=n_epochs,
                             gamma=gamma,
                             gae_lambda=gae_lambda,
                             clip_range=clip_range,
                             ent_coef=ent_coef,
                             vf_coef=vf_coef,
                             verbose=0,
                             seed=SEED + trial.number,
                             device="cpu",
                             )

        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=5, min_evals=3, verbose=0)

        eval_callback = EvalCallback(eval_env,
                                     best_model_save_path=trial_dir,
                                     log_path=trial_dir,
                                     eval_freq=50000,
                                     n_eval_episodes=3,
                                     deterministic=True,
                                     render=False,
                                     callback_after_eval=stop_callback,
                                     verbose=0)

        # Train the model
        model.learn(total_timesteps=1000000,  # Reduced for faster optimization
                    callback=[eval_callback])  # Remove pruning_callback for now

        # Get final evaluation score
        if len(eval_callback.evaluations_results) > 0:
            final_mean_reward = np.mean(eval_callback.evaluations_results[-1])
        else:
            final_mean_reward = -np.inf

        # Clean up environments
        if train_env:
            train_env.close()
        if eval_env:
            eval_env.close()

        # Clean up trial directory if not the best (with safe best_value access)
        try:
            current_best = trial.study.best_value
            if final_mean_reward < current_best:
                shutil.rmtree(trial_dir, ignore_errors=True)
        except (AttributeError, ValueError):
            # No best value exists yet or other database issue
            # Keep the trial directory for now
            pass

        return final_mean_reward

    except optuna.TrialPruned:
        # Clean up on pruning
        if train_env:
            train_env.close()
        if eval_env:
            eval_env.close()
        if trial_dir:
            shutil.rmtree(trial_dir, ignore_errors=True)
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        # Clean up on error
        if train_env:
            train_env.close()
        if eval_env:
            eval_env.close()
        if trial_dir:
            shutil.rmtree(trial_dir, ignore_errors=True)
        return -np.inf


def run_optuna_optimization():
    """Run Optuna hyperparameter optimization"""
    # Create necessary directories
    os.makedirs('optuna_studies', exist_ok=True)
    os.makedirs(f'optuna_trials/{RUN_ID}', exist_ok=True)

    # Create study with better error handling
    study_db_path = f'sqlite:///optuna_studies/{RUN_ID}_study.db'

    try:
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=SEED),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            study_name=f"ppo_trading_{RUN_ID}",
            storage=study_db_path,
            load_if_exists=True,
        )
    except Exception as e:
        logger.warning(
            f"Failed to create study with database: {e}. Creating in-memory study.")
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=SEED),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            study_name=f"ppo_trading_{RUN_ID}",
        )

    print(f"Starting Optuna optimization with study: {study.study_name}")

    # Optimize
    study.optimize(objective, n_trials=50, timeout=3600*12)  # 12 hours timeout

    # Print results
    print("\nOptimization completed!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value}")
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save study results
    study_path = f"./optuna_studies/{RUN_ID}_study.pkl"
    os.makedirs(os.path.dirname(study_path), exist_ok=True)
    with open(study_path, 'wb') as f:
        import pickle
        pickle.dump(study, f)

    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    best_trial = study.best_trial

    # Get best reward function
    temp_trial = optuna.trial.FixedTrial(best_trial.params)
    best_reward_function = get_reward_function(temp_trial)

    # Create best feature configuration
    best_feature_config = {
        'use_volume': best_trial.params.get('use_volume', True),
        'use_high': best_trial.params.get('use_high', True),
        'use_low': best_trial.params.get('use_low', True),
        'use_open': best_trial.params.get('use_open', True),
        # Momentum Indicators
        'use_rsi': best_trial.params.get('use_rsi', False),
        'use_macd': best_trial.params.get('use_macd', True),
        'use_stoch_k': best_trial.params.get('use_stoch_k', False),
        'use_stoch_d': best_trial.params.get('use_stoch_d', False),
        'use_stoch_j': best_trial.params.get('use_stoch_j', False),
        # Trend Indicators
        'use_sma': best_trial.params.get('use_sma', False),
        'use_ema': best_trial.params.get('use_ema', False),
        'use_adx': best_trial.params.get('use_adx', False),
        # Volatility Indicators
        'use_bb_upper': best_trial.params.get('use_bb_upper', False),
        'use_bb_lower': best_trial.params.get('use_bb_lower', False),
        'use_bb_middle': best_trial.params.get('use_bb_middle', False),
        'use_atr': best_trial.params.get('use_atr', False),
    }

    # Create best preprocessing function
    best_preprocess_func = create_preprocess_function(best_feature_config)

    # Create final environments
    final_train_env = gym.make('MultiDatasetTradingEnv',
                               dataset_dir='dataset/1d-2005/train/*.pkl',
                               reward_function=best_reward_function,
                               preprocess=best_preprocess_func,
                               windows=best_trial.params['windows'],
                               positions=[-1, 0, 1],
                               trading_fees=best_trial.params['trading_fees'],
                               borrow_interest_rate=best_trial.params['borrow_interest_rate'],
                               )

    final_eval_env = gym.make('MultiDatasetTradingEnv',
                              dataset_dir='dataset/1d-2005/val/*.pkl',
                              reward_function=best_reward_function,
                              preprocess=best_preprocess_func,
                              windows=best_trial.params['windows'],
                              positions=[-1, 0, 1],
                              trading_fees=best_trial.params['trading_fees'],
                              borrow_interest_rate=best_trial.params['borrow_interest_rate'],
                              )

    final_train_env.reset(seed=SEED)
    final_eval_env.reset(seed=SEED)
    final_eval_env = Monitor(final_eval_env)

    # Create final model
    final_model = RecurrentPPO("MlpLstmPolicy",
                               final_train_env,
                               learning_rate=best_trial.params['learning_rate'],
                               n_steps=best_trial.params['n_steps'],
                               batch_size=best_trial.params['batch_size'],
                               n_epochs=best_trial.params['n_epochs'],
                               gamma=best_trial.params['gamma'],
                               gae_lambda=best_trial.params['gae_lambda'],
                               clip_range=best_trial.params['clip_range'],
                               ent_coef=best_trial.params['ent_coef'],
                               vf_coef=best_trial.params['vf_coef'],
                               verbose=1,
                               tensorboard_log="./runs",
                               seed=SEED,
                               device="cpu",
                               )

    # Final training callbacks
    final_stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10, min_evals=5, verbose=1)

    final_eval_callback = EvalCallback(final_eval_env,
                                       best_model_save_path=f'./model/{RUN_ID}/',
                                       log_path=f'./eval_logs/{RUN_ID}/',
                                       eval_freq=100000,
                                       n_eval_episodes=5,
                                       deterministic=True,
                                       render=False,
                                       callback_after_eval=final_stop_callback,
                                       verbose=1)

    # Train final model
    print(f"Training final optimized model... [id: {RUN_ID}]")
    final_model.learn(total_timesteps=5000000,
                      tb_log_name=f"{RUN_ID}_optimized",
                      callback=final_eval_callback)

    final_model.save(f"./model/{RUN_ID}/final_optimized_model")

    # Save best parameters including feature configuration
    import json
    complete_config = {
        'hyperparameters': best_trial.params,
        'feature_config': best_feature_config,
        'run_id': RUN_ID,
        'study_name': study.study_name,
        'best_value': study.best_value,
        'best_trial_number': best_trial.number
    }

    with open(f"./model/{RUN_ID}/best_params.json", 'w') as f:
        json.dump(complete_config, f, indent=2)

    # Also save feature configuration separately for easy access
    with open(f"./model/{RUN_ID}/feature_config.json", 'w') as f:
        json.dump(best_feature_config, f, indent=2)

    print(f"Optimization complete! Best model saved with ID: {RUN_ID}")
    print(f"Best feature configuration:")
    for feature, enabled in best_feature_config.items():
        if enabled and not feature.endswith('_window'):
            print(f"  - {feature}: {enabled}")

    return study


if __name__ == "__main__":
    # Run Optuna optimization
    study = run_optuna_optimization()
