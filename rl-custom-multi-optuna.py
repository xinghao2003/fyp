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

        # Get reward function
        custom_reward_function = get_reward_function(trial)

        # Create trial-specific directory
        trial_dir = f"./optuna_trials/{RUN_ID}/trial_{trial.number}"
        os.makedirs(trial_dir, exist_ok=True)

        # Training environment
        train_env = gym.make('MultiDatasetTradingEnv',
                             dataset_dir='dataset/1d-2005/train/*.pkl',
                             reward_function=custom_reward_function,
                             preprocess=preprocess,
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
                            preprocess=preprocess,
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

        # Set up callbacks with pruning
        class OptunaPruningCallback:
            def __init__(self, trial, eval_freq):
                self.trial = trial
                self.eval_freq = eval_freq
                self.eval_count = 0

            def __call__(self, locals, globals):
                self.eval_count += 1
                if self.eval_count % (self.eval_freq // 1000) == 0:
                    # Report intermediate value for pruning
                    if len(locals['self'].evaluations_results) > 0:
                        mean_reward = np.mean(
                            locals['self'].evaluations_results[-1])
                        self.trial.report(mean_reward, self.eval_count)

                        # Prune trial if needed
                        if self.trial.should_prune():
                            raise optuna.TrialPruned()

        pruning_callback = OptunaPruningCallback(trial, eval_freq=50000)

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
                    callback=[eval_callback, pruning_callback])

        # Get final evaluation score
        if len(eval_callback.evaluations_results) > 0:
            final_mean_reward = np.mean(eval_callback.evaluations_results[-1])
        else:
            final_mean_reward = -np.inf

        # Clean up environments
        train_env.close()
        eval_env.close()

        # Clean up trial directory if not the best
        if final_mean_reward < trial.study.best_value if trial.study.best_value else True:
            shutil.rmtree(trial_dir, ignore_errors=True)

        return final_mean_reward

    except optuna.TrialPruned:
        # Clean up on pruning
        try:
            train_env.close()
            eval_env.close()
            shutil.rmtree(trial_dir, ignore_errors=True)
        except:
            pass
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return -np.inf


def run_optuna_optimization():
    """Run Optuna hyperparameter optimization"""
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        study_name=f"ppo_trading_{RUN_ID}"
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

    # Create final environments
    final_train_env = gym.make('MultiDatasetTradingEnv',
                               dataset_dir='dataset/1d-2005/train/*.pkl',
                               reward_function=best_reward_function,
                               preprocess=preprocess,
                               windows=best_trial.params['windows'],
                               positions=[-1, 0, 1],
                               trading_fees=best_trial.params['trading_fees'],
                               borrow_interest_rate=best_trial.params['borrow_interest_rate'],
                               )

    final_eval_env = gym.make('MultiDatasetTradingEnv',
                              dataset_dir='dataset/1d-2005/val/*.pkl',
                              reward_function=best_reward_function,
                              preprocess=preprocess,
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

    # Save best parameters
    import json
    with open(f"./model/{RUN_ID}/best_params.json", 'w') as f:
        json.dump(best_trial.params, f, indent=2)

    print(f"Optimization complete! Best model saved with ID: {RUN_ID}")
    return study


if __name__ == "__main__":
    # Run Optuna optimization
    study = run_optuna_optimization()
