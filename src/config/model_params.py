"""
Configuration for DRL models, hyperparameters.
"""

# Example PPO parameters
PPO_PARAMS = {
    "learning_rate": 0.0003,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

# Example SAC parameters
SAC_PARAMS = {
    "learning_rate": 0.0003,
    "buffer_size": 1000000,
    "learning_starts": 100,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    # "train_freq": (1, "episode"), # or (n, "step")
    "gradient_steps": 1,
}


# General training parameters
TRAINING_PARAMS = {
    "total_timesteps_ppo": 1e6,
    "total_timesteps_sac": 1e6,
}

pass
