"""
Custom callbacks for training (e.g., early stopping, model saving).
"""

from stable_baselines3.common.callbacks import BaseCallback
import csv
import os


class CsvLoggingCallback(BaseCallback):
    """
    Custom callback for logging training information to a CSV file.
    Logs episode, step, and reward information.
    """

    def __init__(self, csv_path, verbose=0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.episode_rewards = []
        self.episode_steps = 0
        self.episode_num = 0
        self.current_reward = 0

    def _on_training_start(self) -> None:
        # Write header if file does not exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['episode', 'step', 'reward'])

    def _on_step(self) -> bool:
        # Called after each env.step()
        reward = self.locals.get('rewards', [0])[0]
        done = self.locals.get('dones', [False])[0]
        self.episode_steps += 1
        self.current_reward += reward
        if done:
            self.episode_num += 1
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(
                    [self.episode_num, self.episode_steps, self.current_reward])
            self.episode_steps = 0
            self.current_reward = 0
        return True


pass
