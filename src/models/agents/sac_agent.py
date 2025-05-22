"""
Implementation of SAC agent.
"""

import numpy as np
from stable_baselines3 import SAC


class SACAgent:
    def __init__(self, env, policy='MlpPolicy', **kwargs):
        self.model = SAC(policy, env, **kwargs)

    def train(self, total_timesteps=10000, tensorboard_log_dir=None):
        if tensorboard_log_dir:
            self.model.tensorboard_log = tensorboard_log_dir
        self.model.learn(total_timesteps=total_timesteps,
                         tb_log_name="SAC", reset_num_timesteps=True)

    def predict(self, observation):
        action, _states = self.model.predict(observation, deterministic=True)
        if np.isscalar(action):
            return np.array([action, 1], dtype=int)
        if isinstance(action, (np.ndarray, list, tuple)) and len(action) == 2:
            return np.array(action, dtype=int)
        return np.array([0, 1], dtype=int)

    def save(self, path):
        self.model.save(path)

    @classmethod
    def load(cls, path, env=None):
        loaded_model = SAC.load(path, env=env)
        agent = cls(env=env)
        agent.model = loaded_model

        return agent
