import gym_trading_env
import gymnasium as gym
from gym_trading_env.downloader import download
import datetime
import pandas as pd

# # Download BTC/USDT historical data from Binance and stores it to directory ./data/binance-BTCUSDT-1h.pkl
# download(exchange_names=["binance"],
#          symbols=["BTC/USDT"],
#          timeframe="1h",
#          dir="data",
#          since=datetime.datetime(year=2020, month=1, day=1),
#          )
# # Import your fresh data
# df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")

import pandas as pd
# Available in the github repo : examples/data/BTC_USD-Hourly.csv
url = "https://raw.githubusercontent.com/ClementPerroud/Gym-Trading-Env/main/examples/data/BTC_USD-Hourly.csv"
df = pd.read_csv(url, parse_dates=["date"], index_col="date")
df.sort_index(inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

env = gym.make("TradingEnv",
               name="BTCUSD",
               df=df,  # Your dataset with your custom features
               positions=[-1, 0, 1],  # -1 (=SHORT), 0(=OUT), +1 (=LONG)
               # 0.01% per stock buy / sell (Binance fees)
               trading_fees=0.01/100,
               # 0.0003% per timestep (one timestep = 1h here)
               borrow_interest_rate=0.0003/100,
               )

# Run an episode until it ends :
done, truncated = False, False
observation, info = env.reset()
while not done and not truncated:
    # Pick a position by its index in your position list (=[-1, 0, 1])....usually something like : position_index = your_policy(observation)
    # At every timestep, pick a random position index from your position list (=[-1, 0, 1])
    position_index = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(position_index)

# At the end of the episode you want to render
env.save_for_render(dir="render_logs")
