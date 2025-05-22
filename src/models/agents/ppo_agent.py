"""
Implementation of PPO agent.
"""


import numpy as np
from stable_baselines3 import PPO


class PPOAgent:
    def __init__(self, env, policy='MlpPolicy', **kwargs):
        self.model = PPO(policy, env, **kwargs)

    def train(self, total_timesteps=10000, tensorboard_log_dir=None, csv_log_path=None):
        # If tensorboard_log_dir is provided, pass it to SB3
        if tensorboard_log_dir:
            self.model.tensorboard_log = tensorboard_log_dir
        callback = None
        if csv_log_path:
            from src.training.trainer_callbacks import CsvLoggingCallback
            callback = CsvLoggingCallback(csv_log_path)
        self.model.learn(
            total_timesteps=total_timesteps,
            tb_log_name="PPO",
            reset_num_timesteps=True,
            callback=callback
        )
        # Note: SB3 automatically logs to TensorBoard if tensorboard_log is set

    def predict(self, observation):
        action, _states = self.model.predict(observation, deterministic=True)
        # If action is a scalar, convert to [action_type, units=1] for backward compatibility
        if np.isscalar(action):
            return np.array([action, 1], dtype=int)
        # If action is a 1D array of length 2, return as is
        if isinstance(action, (np.ndarray, list, tuple)) and len(action) == 2:
            return np.array(action, dtype=int)
        # Otherwise, fallback to hold
        return np.array([0, 1], dtype=int)

    def save(self, path):
        self.model.save(path)

    @classmethod
    def load(cls, path, env=None):
        loaded_model = PPO.load(path, env=env)
        agent = cls(env=env)
        agent.model = loaded_model
        return agent
