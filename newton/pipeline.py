import logging
from math import log
from dotenv import load_dotenv
from data_retreival import AlphaVantageDownloader, YahooFinanceDownloader
from data_processing import AlphaVantageDataProcessor, YahooFinanceDataProcessor
from gym_anytrading.envs import StocksEnv
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

    # Step 4: Implement an RL agent using Stable Baselines3
    # Implement PPO agent
    try:
        logger.info("Starting pipeline step 4: Implement RL agent")

        model = PPO('MlpPolicy', env, verbose=1,
                    tensorboard_log="./runs/ppo_stock_trading", seed=42)
        logger.info("RL agent implemented successfully!")
    except Exception as e:
        logger.error(f"Error in pipeline step 4: {str(e)}", exc_info=True)

    # Step 5: Train the RL agent
    try:
        logger.info("Starting pipeline step 5: Train the RL agent")
        model.learn(total_timesteps=200000,
                    reset_num_timesteps=True, progress_bar=True)
        logger.info("RL agent trained successfully!")
    except Exception as e:
        # Step 6: Backtest the RL agent
        logger.error(f"Error in pipeline step 5: {str(e)}", exc_info=True)
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

        while not done and not truncated:
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = env_test.step(
                action)  # New gymnasium API returns 5 values
            total_reward += reward

        logger.info(f"Reward from backtesting: {total_reward}")
        logger.info(
            f"Total profit from backtesting: {info.get('total_profit', 'N/A')}")

        logger.info("Backtesting completed successfully!")
    except Exception as e:
        logger.error(f"Error in pipeline step 6: {str(e)}", exc_info=True)
