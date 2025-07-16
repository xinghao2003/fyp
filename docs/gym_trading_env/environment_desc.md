Title: Environment Quick Summary

URL Source: http://gym-trading-env.readthedocs.io/en/latest/environment_desc.html

Published Time: Fri, 21 Feb 2025 10:24:37 GMT

Markdown Content:
Toggle table of contents sidebar

[![Image 1: _images/render.gif](https://gym-trading-env.readthedocs.io/en/latest/_images/render.gif)](https://gym-trading-env.readthedocs.io/en/latest/_images/render.gif)
This environment is a [Gymnasium](https://gymnasium.farama.org/content/basic_usage/) environment designed for trading on a single pair.

Action Space`Discrete(number_of_positions)`
Observation Space`Box(-np.inf, +np.inf, shape=...)`
Import`gymnasium.make("TradingEnv", df=df)`

Important Parameters[#](http://gym-trading-env.readthedocs.io/en/latest/environment_desc.html#important-parameters "Permalink to this heading")
-----------------------------------------------------------------------------------------------------------------------------------------------

*   `df`_(required)_: A pandas.DataFrame with a `close` and DatetimeIndex as index. To perform a render, your DataFrame also needs to contain `open`, `low`, and `high`.

*   `positions`_(optional, default: [-1, 0, 1])_: The list of positions that your agent can take. Each position is represented by a number (as described in the _Action Space_ section).

[Documentation of all the parameters](https://gym-trading-env.readthedocs.io/en/latest/documentation.html#gym_trading_env.environments.TradingEnv)

Action Space[#](http://gym-trading-env.readthedocs.io/en/latest/environment_desc.html#action-space "Permalink to this heading")
-------------------------------------------------------------------------------------------------------------------------------

The action space is a list of **positions** given by the user. Every position is labeled from -inf to +inf and corresponds to the ratio of the portfolio valuation engaged in the position ( > 0 to bet on the rise, < 0 to bet on the decrease).

Example with BTC/USDT pair (%pv means _“Percent of the Portfolio Valuation”_)[#](http://gym-trading-env.readthedocs.io/en/latest/environment_desc.html#id1 "Permalink to this table")| Position examples | BTC (%pv) | USDT (%pv) | Borrowed BTC (%pv) | Borrowed USDT (%pv) |
| --- | --- | --- | --- | --- |
| **0** |  | 100 |  |  |
| **1** | 100 |  |  |  |
| **0.5** | 50 | 50 |  |  |
| **2** | 200 |  |  | 100 |
| **-1** |  | 200 | 100 |  |

If `position < 0`: the environment performs a SHORT (by borrowing USDT and buying BTC with it).

If `position > 1`: the environment uses MARGIN trading (by borrowing BTC and selling it to get USDT).

Observation Space[#](http://gym-trading-env.readthedocs.io/en/latest/environment_desc.html#observation-space "Permalink to this heading")
-----------------------------------------------------------------------------------------------------------------------------------------

The observation space is an np.array containing:

*   The row of your DataFrame columns containing `features` in their name, at a given step : the **static features**

*   The **dynamic features** (by default, the last position taken by the agent, and the current real position).

>>> df["feature_pct_change"] = df["close"].pct_change()
>>> df["feature_high"] = df["high"] / df["close"] - 1
>>> df["feature_low"] = df["low"] / df["close"] - 1
>>> df.dropna(inplace= True)
>>> env = gymnasium.make("TradingEnv", df = df, positions = [-1, 0, 1], initial_position= 1)
>>> observation, info = env.reset()
>>> observation
array([-2.2766300e-04, 1.0030895e+00, 9.9795288e-01, 1.0000000e+00], dtype=float32)

If the `windows` parameter is set to an integer W > 1, the observation is a stack of the last W states.

>>> env = gymnasium.make("TradingEnv", df = df, positions = [-1, 0, 1], initial_position= 1, windows = 3)
>>> observation, info = env.reset()
>>> observation
array([[-0.00231082, 1.0052915 , 0.9991996 , 1. ],
 [ 0.01005705, 1.0078559 , 0.98854125, 1. ],
 [-0.00408145, 1.0069852 , 0.99777853, 1. ]],
 dtype=float32)

Reward[#](http://gym-trading-env.readthedocs.io/en/latest/environment_desc.html#reward "Permalink to this heading")
-------------------------------------------------------------------------------------------------------------------

The reward is given by the formula  . It is highly recommended to [customize the reward function](https://gym-trading-env.readthedocs.io/en/latest/customization.html#custom-reward-function) to your needs.

Starting State[#](http://gym-trading-env.readthedocs.io/en/latest/environment_desc.html#starting-state "Permalink to this heading")
-----------------------------------------------------------------------------------------------------------------------------------

The environment explores the given DataFrame and starts at its beginning.

Episode Termination[#](http://gym-trading-env.readthedocs.io/en/latest/environment_desc.html#episode-termination "Permalink to this heading")
---------------------------------------------------------------------------------------------------------------------------------------------

The episode finishes if:

1 - The environment reaches the end of the DataFrame, `truncated` is returned as `True` 2 - The portfolio valuation reaches 0 (or bellow). `done` is returned as `True`. It can happen when taking margin positions (>1 or <0).

Arguments[#](http://gym-trading-env.readthedocs.io/en/latest/environment_desc.html#arguments "Permalink to this heading")
-------------------------------------------------------------------------------------------------------------------------

_class_ gym_trading_env.environments.TradingEnv(_df:~pandas.core.frame.DataFrame,positions:list=[0,1],dynamic\_feature\_functions=[<function dynamic\_feature\_last\_position\_taken>,<function dynamic\_feature\_real\_position>],reward\_function=<function basic\_reward\_function>,windows=None,trading\_fees=0,borrow\_interest\_rate=0,portfolio\_initial\_value=1000,initial\_position='random',max\_episode\_duration='max',verbose=1,name='Stock',render\_mode='logs'_)[#](http://gym-trading-env.readthedocs.io/en/latest/environment_desc.html#gym_trading_env.environments.TradingEnv "Permalink to this definition")
An easy trading environment for OpenAI gym. It is recommended to use it this way :

import gymnasium as gym
import gym_trading_env
env = gym.make('TradingEnv', ...)

Parameters
*   **df** (_pandas.DataFrame_) – The market DataFrame. It must contain ‘open’, ‘high’, ‘low’, ‘close’. Index must be DatetimeIndex. Your desired inputs need to contain ‘feature’ in their column name : this way, they will be returned as observation at each step.

*   **positions** (_optional - list_ _[_ _int_ _or_ _float_ _]_) – List of the positions allowed by the environment.

*   **dynamic_feature_functions** (_optional - list_) –

The list of the dynamic features functions. By default, two dynamic features are added :

    *   the last position taken by the agent.

    *   the real position of the portfolio (that varies according to the price fluctuations)

*   **reward_function** (_optional - function<History->float>_) – Take the History object of the environment and must return a float.

*   **windows** (_optional - None_ _or_ _int_) – Default is None. If it is set to an int: N, every step observation will return the past N observations. It is recommended for Recurrent Neural Network based Agents.

*   **trading_fees** (_optional - float_) – Transaction trading fees (buy and sell operations). eg: 0.01 corresponds to 1% fees

*   **borrow_interest_rate** (_optional - float_) – Borrow interest rate per step (only when position < 0 or position > 1). eg: 0.01 corresponds to 1% borrow interest rate per STEP ; if your know that your borrow interest rate is 0.05% per day and that your timestep is 1 hour, you need to divide it by 24 -> 0.05/100/24.

*   **portfolio_initial_value** (_float_ _or_ _int_) – Initial valuation of the portfolio.

*   **initial_position** (_optional - float_ _or_ _int_) – You can specify the initial position of the environment or set it to ‘random’. It must contained in the list parameter ‘positions’.

*   **max_episode_duration** (_optional - int_ _or_ _'max'_) – If a integer value is used, each episode will be truncated after reaching the desired max duration in steps (by returning truncated as True). When using a max duration, each episode will start at a random starting point.

*   **verbose** (_optional - int_) – If 0, no log is outputted. If 1, the env send episode result logs.

*   **name** (_optional - str_) – The name of the environment (eg. ‘BTC/USDT’)
