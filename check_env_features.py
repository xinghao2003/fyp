import gym_trading_env
import gymnasium as gym
import pandas as pd
import numpy as np
from reward import reward_function_5 as custom_reward_function


def preprocess(df: pd.DataFrame):
    # Create your features
    try:
        df["feature_macd"] = df["macd"]
        print(f"Successfully added feature_macd from macd column")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        print(f"Available columns: {df.columns.tolist()}")
    return df


def inspect_environment():
    print("=" * 60)
    print("ENVIRONMENT FEATURE INSPECTION")
    print("=" * 60)

    # Create environment
    env = gym.make('MultiDatasetTradingEnv',
                   dataset_dir='dataset/1d-2005/*.pkl',
                   reward_function=custom_reward_function,
                   preprocess=preprocess,
                   verbose=2  # Increase verbosity to see dataset selection
                   )

    # Reset environment to initialize
    obs, info = env.reset(seed=42)

    print(f"\n1. DATASET INFO:")
    print(f"   Current dataset: {env.name}")
    print(f"   DataFrame shape: {env.df.shape}")

    print(f"\n2. COLUMN ANALYSIS:")
    print(f"   All columns: {env.df.columns.tolist()}")
    print(f"   Feature columns: {env._features_columns}")
    print(f"   Info columns: {env._info_columns}")

    print(f"\n3. MACD FEATURE CHECK:")
    if 'macd' in env.df.columns:
        print(f"   ✓ 'macd' column found in DataFrame")
        print(f"   MACD sample values: {env.df['macd'].head().tolist()}")
    else:
        print(f"   ✗ 'macd' column NOT found in DataFrame")

    if 'feature_macd' in env.df.columns:
        print(f"   ✓ 'feature_macd' column found in DataFrame")
        print(
            f"   Feature MACD sample values: {env.df['feature_macd'].head().tolist()}")
    else:
        print(f"   ✗ 'feature_macd' column NOT found in DataFrame")

    print(f"\n4. OBSERVATION SPACE:")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Number of features: {env._nb_features}")
    print(f"   Static features: {env._nb_static_features}")

    print(f"\n5. ACTUAL OBSERVATION:")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation sample: {obs}")

    print(f"\n6. FEATURE MAPPING:")
    for i, col in enumerate(env._features_columns):
        print(f"   Feature {i}: {col}")

    # Take a few steps to see if everything works
    print(f"\n7. ENVIRONMENT STEP TEST:")
    for step in range(3):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(
            f"   Step {step+1}: Action={action}, Reward={reward:.6f}, Done={done}, Truncated={truncated}")
        if done or truncated:
            break

    env.close()
    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    inspect_environment()
