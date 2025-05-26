import logging
from math import log
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import optuna
from data_retreival import AlphaVantageDownloader, YahooFinanceDownloader
from data_processing import AlphaVantageDataProcessor, YahooFinanceDataProcessor
from environment import StocksEnv
from stable_baselines3 import PPO

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Step 1: Stock Data Download
    # try:
    #     logger.info("Starting pipeline step 1: Stock data download")
    #     stock_data = YahooFinanceDownloader.download_stock_data()
    #     if stock_data is not None and len(stock_data) > 0:
    #         logger.info("Pipeline step 1 completed successfully!")
    #     else:
    #         logger.warning("Pipeline step 1 did not return any data.")
    # except Exception as e:
    #     logger.error(f"Error in pipeline step 1: {str(e)}", exc_info=True)

    # Step 2: Data Processing
    # try:
    #     logger.info("Starting pipeline step 2: Data processing")
    #     processor = YahooFinanceDataProcessor(
    #         'stock_data/AAPL_Yahoo_Adj_2007-2024_Daily.csv')
    #     processed_data = processor.process_data()

    #     if processed_data:
    #         # Store processed data to disk
    #         processor.save_processed_data(
    #             processed_data, 'processed_data/AAPL_Yahoo_Adj_2007-2024_Daily.pkl')
    #         logger.info("Processed data saved successfully!")
    #         logger.info("Pipeline step 2 completed successfully!")
    #     else:
    #         logger.warning(
    #             "Pipeline step 2 did not return any processed data.")
    # except Exception as e:
    #     logger.error(f"Error in pipeline step 2: {str(e)}", exc_info=True)

    # Step 2.5: Load Processed Data
    try:
        logger.info("Starting pipeline step 2.5: Load processed data")
        processed_data = YahooFinanceDataProcessor.load_processed_data(
            'processed_data/AAPL_Yahoo_Adj_2007-2024_Daily.pkl')
        if processed_data:
            logger.info("Processed data loaded successfully!")
        else:
            logger.warning("No data loaded from the processed file.")
    except Exception as e:
        # Step 3: Prepare the environment using gym_anytrading
        logger.error(f"Error in pipeline step 2.5: {str(e)}", exc_info=True)
    try:
        logger.info("Starting pipeline step 3: Prepare the environment")

        # Set proper frame bounds - start after window_size to ensure enough historical data
        window_size = 10
        train_start = window_size
        train_end = len(processed_data['train']) - 1

        env = StocksEnv(df=processed_data['train'], window_size=window_size, frame_bound=(
            train_start, train_end))
        logger.info(
            f"Environment prepared successfully! Frame bound: ({train_start}, {train_end})")
    except Exception as e:
        logger.error(f"Error in pipeline step 3: {str(e)}", exc_info=True)

    # Section A: Basic Pipeline
    # Step 4: Implement an RL agent using Stable Baselines3
    # Implement PPO agent
    # try:
    #     logger.info("Starting pipeline step 4: Implement RL agent")

    #     model = PPO('MlpPolicy', env, verbose=1,
    #                 tensorboard_log="./runs/ppo_stock_trading", seed=42)
    #     logger.info("RL agent implemented successfully!")
    # except Exception as e:
    #     logger.error(f"Error in pipeline step 4: {str(e)}", exc_info=True)

    # Step 5: Train the RL agent
    # try:
    #     # Based on a rough estimate, the agent should achieve a optimal policy within 450,000 timesteps
    #     logger.info("Starting pipeline step 5: Train the RL agent")
    #     model.learn(total_timesteps=200000,
    #                 reset_num_timesteps=True, progress_bar=True)
    #     logger.info("RL agent trained successfully!")
    # except Exception as e:
    #     logger.error(f"Error in pipeline step 5: {str(e)}", exc_info=True)

    # Section B: Optuna Tuning
    # Step 4: Implement an RL agent using Stable Baselines3
    # Implement PPO agent with hyperparameter tuning
    try:
        logger.info(
            "Starting Section B Step 4: Implement RL agent with hyperparameter tuning")

        def objective(trial):
            """Objective function for Optuna hyperparameter optimization"""
            # Suggest hyperparameters
            learning_rate = trial.suggest_float(
                'learning_rate', 1e-5, 1e-2, log=True)
            n_steps = trial.suggest_categorical(
                'n_steps', [128, 256, 512, 1024, 2048])
            batch_size = trial.suggest_categorical(
                'batch_size', [32, 64, 128, 256])
            n_epochs = trial.suggest_int('n_epochs', 3, 30)
            gamma = trial.suggest_float('gamma', 0.9, 0.9999)
            gae_lambda = trial.suggest_float('gae_lambda', 0.8, 1.0)
            clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
            ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-1, log=True)

            # Create model with suggested hyperparameters
            model = PPO(
                'MlpPolicy',
                env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                verbose=1,
                tensorboard_log="./runs/ppo_stock_trading_optuna",
                seed=42
            )

            # Train the model for a short period for evaluation
            model.learn(total_timesteps=50000,
                        reset_num_timesteps=True, progress_bar=True)

            # Evaluate on validation environment
            val_start = window_size
            val_end = len(processed_data['validation']) - 1
            env_val = StocksEnv(df=processed_data['validation'], window_size=window_size,
                                frame_bound=(val_start, val_end))

            obs, info = env_val.reset()
            total_reward = 0
            done = False
            truncated = False

            while not done and not truncated:
                action, _states = model.predict(obs)
                obs, reward, done, truncated, info = env_val.step(action)
                total_reward += reward

            # Return the total profit as the objective to maximize
            return info.get('total_profit', total_reward)

        logger.info(
            "Objective function for hyperparameter tuning created successfully!")

    except Exception as e:
        logger.error(f"Error in Section B Step 4: {str(e)}", exc_info=True)

    # Step 5: Train the RL agent with optimal hyperparameters
    try:
        logger.info(
            "Starting Section B Step 5: Hyperparameter optimization and training")

        # Create Optuna study
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(
                                        seed=42),
                                    storage='sqlite:///ppo_optuna.db',)

        # Optimize hyperparameters
        logger.info("Starting hyperparameter optimization...")
        # 20 trials or 1 hour timeout
        study.optimize(objective, n_trials=20, show_progress_bar=True)

        logger.info("Hyperparameter optimization completed!")
        logger.info(f"Best trial value: {study.best_value}")
        logger.info(f"Best parameters: {study.best_params}")

        # Train final model with best hyperparameters
        logger.info("Training final model with optimal hyperparameters...")
        best_params = study.best_params

        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=best_params['learning_rate'],
            n_steps=best_params['n_steps'],
            batch_size=best_params['batch_size'],
            n_epochs=best_params['n_epochs'],
            gamma=best_params['gamma'],
            gae_lambda=best_params['gae_lambda'],
            clip_range=best_params['clip_range'],
            ent_coef=best_params['ent_coef'],
            verbose=1,
            tensorboard_log="./runs/ppo_stock_trading_optimized",
            seed=42
        )

        # Train the optimized model for full timesteps
        model.learn(total_timesteps=200000,
                    reset_num_timesteps=True, progress_bar=True)

        # Save the trained model
        model.save("models/ppo_optimized_stock_trading")
        logger.info("Optimized model saved successfully!")
        logger.info("Section B Step 5 completed successfully!")

    except Exception as e:
        logger.error(f"Error in Section B Step 5: {str(e)}", exc_info=True)

    # Step 6: Backtest the RL agent
    try:
        logger.info("Starting pipeline step 6: Backtest the RL agent")

        # Test the trained model on validation data
        window_size = 10
        val_start = window_size
        val_end = len(processed_data['validation']) - 1

        env_test = StocksEnv(df=processed_data['validation'], window_size=window_size, frame_bound=(
            val_start, val_end))
        obs, info = env_test.reset()  # New gymnasium API returns obs and info
        total_reward = 0
        done = False
        truncated = False

        # Store data for plotting
        actions = []
        rewards = []
        profits = []
        step_count = 0

        while not done and not truncated:
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = env_test.step(
                action)  # New gymnasium API returns 5 values
            total_reward += reward

            # Store data for plotting
            actions.append(action)
            rewards.append(reward)
            profits.append(info.get('total_profit', 0))
            step_count += 1

        logger.info(f"Reward from backtesting: {total_reward}")
        logger.info(
            f"Total profit from backtesting: {info.get('total_profit', 'N/A')}")

        logger.info("Backtesting completed successfully!")
    except Exception as e:
        logger.error(f"Error in pipeline step 6: {str(e)}", exc_info=True)

    # Step 7: Plot visualization of trading results
    try:
        logger.info("Starting pipeline step 7: Plot trading visualization")

        # Get stock price data for the validation period
        val_data = processed_data['validation'].iloc[val_start:val_end+1]
        dates = val_data.index[:len(actions)]
        prices = val_data['Close'].values[:len(actions)]

        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

        # Plot 1: Stock price with buy/sell actions
        ax1.plot(dates, prices, label='Stock Price', color='blue', linewidth=1)

        # Mark buy and sell actions
        buy_points = [i for i, action in enumerate(
            actions) if action == 1]  # Buy action (Actions.Buy)
        sell_points = [i for i, action in enumerate(
            actions) if action == 0]  # Sell action (Actions.Sell)

        if buy_points:
            ax1.scatter([dates[i] for i in buy_points], [prices[i] for i in buy_points],
                        color='green', marker='^', s=50, label='Buy', alpha=0.7)
        if sell_points:
            ax1.scatter([dates[i] for i in sell_points], [prices[i] for i in sell_points],
                        color='red', marker='v', s=50, label='Sell', alpha=0.7)

        ax1.set_title('Stock Price and Trading Actions')
        ax1.set_ylabel('Price (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Total profit over time
        ax2.plot(dates, profits, label='Total Profit',
                 color='purple', linewidth=2)
        ax2.set_title('Total Profit Over Time')
        ax2.set_ylabel('Profit (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Reward over time
        cumulative_rewards = np.cumsum(rewards)
        ax3.plot(dates, rewards, label='Step Reward',
                 color='orange', alpha=0.7)
        ax3.plot(dates, cumulative_rewards, label='Cumulative Reward',
                 color='darkred', linewidth=2)
        ax3.set_title('Rewards Over Time')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Reward')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('trading_results_visualization.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

        # Print summary statistics
        logger.info(
            f"Total trades executed: {len([a for a in actions if a in [0, 1]])}")
        logger.info(f"Buy actions: {len(buy_points)}")
        logger.info(f"Sell actions: {len(sell_points)}")
        logger.info(f"Final profit: {profits[-1]:.2f}x")
        logger.info(f"Final cumulative reward: {cumulative_rewards[-1]:.2f}")
        logger.info(
            "Trading visualization saved as 'trading_results_visualization.png'")
        logger.info("Pipeline step 7 completed successfully!")

    except Exception as e:
        logger.error(f"Error in pipeline step 7: {str(e)}", exc_info=True)
