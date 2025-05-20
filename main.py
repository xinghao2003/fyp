"""
Main script to run experiments, training, or evaluation pipelines.
This script will serve as the primary entry point to orchestrate different project workflows.
"""


import argparse
import os
from src.config import settings
from src.data_ingestion.collectors import AlphaVantageCollector
from src.data_ingestion.parsers import parse_alpha_vantage_csv
import pandas as pd
# from src.training import train_agent # Example import
# from src.evaluation import backtester # Example import


def run_data_pipeline():
    print("Running data ingestion and parsing pipeline...")
    # Set output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True)

    # Download AAPL data from Alpha Vantage
    try:
        api_key = getattr(settings, 'ALPHA_VANTAGE_API_KEY')
        if not api_key:
            raise ValueError(
                "ALPHA_VANTAGE_API_KEY is required but not set in settings")
        alpha_collector = AlphaVantageCollector(api_key=api_key)
        alpha_data = alpha_collector.fetch_historical_data(
            'AAPL', interval='1d', outputsize='full')
        alpha_path = os.path.join(output_dir, 'AAPL_alpha_vantage.csv')
        alpha_data.to_csv(alpha_path)
        print(f"AAPL Alpha Vantage data saved to {alpha_path}")

        # Parse the downloaded CSV using the parser
        parsed_df = parse_alpha_vantage_csv(alpha_path)
        print(f"Parsed DataFrame (head):\n{parsed_df.head()}")
        # Store the parsed data in the processed directory, and split into train/test
        processed_dir = os.path.join(
            os.path.dirname(__file__), 'data', 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        # Sort by date if possible
        if 'date' in parsed_df.columns:
            parsed_df = parsed_df.sort_values('date')
        # Split by datetime: 2020-2024 for train, 2025 for eval
        if 'date' in parsed_df.columns:
            print("Splitting data into train and eval sets based on date...")
            parsed_df['date'] = pd.to_datetime(parsed_df['date'])
            train_df = parsed_df[(parsed_df['date'] >= '2020-01-01')
                                 & (parsed_df['date'] < '2025-01-01')]
            eval_df = parsed_df[(parsed_df['date'] >= '2025-01-01')
                                & (parsed_df['date'] < '2026-01-01')]
        else:
            # fallback: use 80/20 split if no date column
            print(
                "No date column found. Splitting data into train and eval sets based on index...")
            split_idx = int(0.8 * len(parsed_df))
            train_df = parsed_df.iloc[:split_idx]
            eval_df = parsed_df.iloc[split_idx:]
        train_path = os.path.join(
            processed_dir, 'AAPL_alpha_vantage_train.csv')
        eval_path = os.path.join(processed_dir, 'AAPL_alpha_vantage_eval.csv')
        train_df.to_csv(train_path, index=False)
        eval_df.to_csv(eval_path, index=False)
        print(f"Train data saved to {train_path}")
        print(f"Eval data saved to {eval_path}")
    except Exception as e:
        print(f"Alpha Vantage download or parsing failed: {e}")


def run_training_pipeline(agent_type):
    print(f"Running training pipeline for {agent_type} agent...")
    # 1. Load training data
    train_data_path = os.path.join(os.path.dirname(
        __file__), 'data', 'processed', 'AAPL_alpha_vantage_train.csv')
    if not os.path.exists(train_data_path):
        print(
            f"Training data not found at {train_data_path}. Please run the data pipeline first.")
        return
    import pandas as pd
    data = pd.read_csv(train_data_path)

    # 2. Initialize trading environment
    from src.drl_environment.trading_env import TradingEnv
    env = TradingEnv(data)

    # 3. Select and initialize the correct DRL agent
    agent = None
    if agent_type == "ppo":
        from src.models.agents.ppo_agent import PPOAgent
        agent_class = PPOAgent
    elif agent_type == "sac":
        from src.models.agents.sac_agent import SACAgent
        agent_class = SACAgent
    elif agent_type == "dqn":
        from src.models.agents.dqn_agent import DQNAgent
        agent_class = DQNAgent
    else:
        print(f"Unknown agent type: {agent_type}")
        return

    # Check for autotune arguments from global scope (argparse args)
    import sys
    auto_tune = getattr(sys.modules['__main__'], 'auto_tune', False)
    n_trials = getattr(sys.modules['__main__'], 'n_trials', 20)

    if auto_tune:
        print(
            f"Auto-tuning enabled for {agent_type} agent with {n_trials} trials...")
        # Try to import the tuning function for the agent
        try:
            if agent_type == "ppo":
                from src.training.ppo_tune import tune_ppo
                tune_func = tune_ppo
            elif agent_type == "sac":
                from src.training.sac_tune import tune_sac
                tune_func = tune_sac
            elif agent_type == "dqn":
                from src.training.dqn_tune import tune_dqn
                tune_func = tune_dqn
            else:
                print(f"No tuning function found for agent type: {agent_type}")
                return
            best_params = tune_func(env, n_trials=n_trials)
            print(f"Best hyperparameters found: {best_params}")
            agent = agent_class(env, **best_params)
        except Exception as e:
            print(f"Auto-tuning failed: {e}")
            print("Falling back to default agent initialization.")
            agent = agent_class(env)
    else:
        agent = agent_class(env)

    # 4. Train the agent with TensorBoard logging
    print("Starting training with TensorBoard logging...")
    tensorboard_log_dir = os.path.join(
        os.path.dirname(__file__), 'runs', agent_type)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    # Path for CSV logging
    csv_log_path = os.path.join(os.path.dirname(__file__), 'training_log.csv')
    agent.train(tensorboard_log_dir=tensorboard_log_dir,
                csv_log_path=csv_log_path)
    print(f"Training log saved to {csv_log_path}")
    print(f"TensorBoard logs saved to {tensorboard_log_dir}")

    # 5. Save the trained model
    model_save_path = os.path.join(os.path.dirname(
        __file__), 'models', f'{agent_type}_agent_trained.pth')
    agent.save(model_save_path)
    print(f"Trained model saved to {model_save_path}")


def run_evaluation_pipeline(model_path):
    print(f"Running evaluation pipeline for model at {model_path}...")
    import pandas as pd
    import os
    import numpy as np
    # 1. Load evaluation data
    eval_data_path = os.path.join(os.path.dirname(
        __file__), 'data', 'processed', 'AAPL_alpha_vantage_eval.csv')
    if not os.path.exists(eval_data_path):
        print(
            f"Evaluation data not found at {eval_data_path}. Please run the data pipeline first.")
        return
    data = pd.read_csv(eval_data_path)

    # 2. Initialize trading environment
    from src.drl_environment.trading_env import TradingEnv
    env = TradingEnv(data)

    # 3. Detect agent type from model_path (simple heuristic)
    agent_type = None
    if 'ppo' in os.path.basename(model_path).lower():
        agent_type = 'ppo'
    elif 'sac' in os.path.basename(model_path).lower():
        agent_type = 'sac'
    elif 'dqn' in os.path.basename(model_path).lower():
        agent_type = 'dqn'
    else:
        print("Could not determine agent type from model path. Please use a filename containing 'ppo', 'sac', or 'dqn'.")
        return

    # 4. Load the trained agent
    agent = None
    if agent_type == 'ppo':
        from src.models.agents.ppo_agent import PPOAgent
        agent = PPOAgent.load(model_path, env=env)
    elif agent_type == 'sac':
        from src.models.agents.sac_agent import SACAgent
        agent = SACAgent.load(model_path, env=env)
    elif agent_type == 'dqn':
        from src.models.agents.dqn_agent import DQNAgent
        agent = DQNAgent.load(model_path, env=env)

    if agent is None:
        print(f"Failed to initialize agent of type {agent_type}.")
        return

    # 5. Run evaluation (simple loop, can be replaced with backtester)
    # Handle both old and new gym API formats for reset()
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, _ = reset_result  # New Gym API returns (obs, info)
    else:
        obs = reset_result     # Old Gym API returns just obs

    done = False
    total_reward = 0
    steps = 0

    # For visualization: record prices, actions, portfolio values
    prices = []
    actions = []
    portfolio_values = []

    while not done:
        action = agent.predict(obs)
        # Convert numpy array to int if needed
        if isinstance(action, np.ndarray):
            action = int(action.item()) if action.size == 1 else int(
                np.argmax(action))
        # Unpack the step result properly (handling potential 5-tuple return)
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, done, _, info = step_result
        else:
            obs, reward, done, info = step_result

        # Record for plotting
        prices.append(env._get_price())
        actions.append(action)
        portfolio_values.append(env.total_asset)

        total_reward += reward
        steps += 1

    print(
        f"Evaluation finished in {steps} steps. Total reward: {total_reward}")

    # Visualization
    try:
        from src.evaluation.plotting import plot_trading_evaluation
        plot_trading_evaluation(prices, actions, portfolio_values,
                                title="Evaluation: Price, Actions, Portfolio Value")
    except Exception as e:
        print(f"Plotting failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="DRL Trading Model Framework")
    parser.add_argument("--pipeline", type=str, choices=["data", "train", "evaluate"], required=True,
                        help="Pipeline to run: data, train, or evaluate")
    parser.add_argument("--agent_type", type=str,
                        choices=["ppo", "sac", "dqn"], help="Type of DRL agent for training")
    parser.add_argument("--model_path", type=str,
                        help="Path to the trained model for evaluation")
    parser.add_argument("--auto_tune", action="store_true",
                        help="Enable hyperparameter auto-tuning for training agents.")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of trials for hyperparameter tuning (default: 20)")

    args = parser.parse_args()

    # Make auto_tune and n_trials available globally for run_training_pipeline
    import sys
    sys.modules['__main__'].auto_tune = args.auto_tune
    sys.modules['__main__'].n_trials = args.n_trials

    if args.pipeline == "data":
        run_data_pipeline()
    elif args.pipeline == "train":
        if not args.agent_type:
            parser.error("--agent_type is required for the training pipeline.")
        run_training_pipeline(args.agent_type)
    elif args.pipeline == "evaluate":
        if not args.model_path:
            parser.error(
                "--model_path is required for the evaluation pipeline.")
        run_evaluation_pipeline(args.model_path)


if __name__ == "__main__":
    main()
