Title: Customization

URL Source: http://gym-trading-env.readthedocs.io/en/latest/customization.html

Markdown Content:
[Back to top](http://gym-trading-env.readthedocs.io/en/latest/customization.html#)

Toggle table of contents sidebar

Custom reward function[#](http://gym-trading-env.readthedocs.io/en/latest/customization.html#custom-reward-function "Permalink to this heading")
------------------------------------------------------------------------------------------------------------------------------------------------

Use the [History object](https://gym-trading-env.readthedocs.io/en/latest/history.html) to create your custom reward function. Below is an example with a really basic reward function  (this is the default reward function).

import gymnasium as gym
import numpy as np
def reward_function(history):
        return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])

env = gym.make("TradingEnv",
        ...
        reward_function = reward_function
        ...
    )

Custom logs[#](http://gym-trading-env.readthedocs.io/en/latest/customization.html#custom-logs "Permalink to this heading")
--------------------------------------------------------------------------------------------------------------------------

Use the [History object](https://gym-trading-env.readthedocs.io/en/latest/history.html) to add custom logs. If the `verbose` parameter of your trading environment is set to `1` or `2`, the environment displays a quick summary of your episode. By default Market Return and Portfolio Return are the displayed metrics.

Market Return :  25.30%   |   Portfolio Return : 45.24%

You can add custom metrics using the method `.add_metric(name, function)` after initializing your environment :

env = gym.make("TradingEnv",
       ...
   )
env.add_metric('Position Changes', lambda history : np.sum(np.diff(history['position']) != 0) )
env.add_metric('Episode Length', lambda history : len(history['position']) )
# Then, run your episode(s)

Market Return :  25.30%   |   Portfolio Return : 45.24%   |   Position Changes : 28417   |   Episode Lenght : 33087

The `.add_metric` method takes 2 parameters :

*   `name` : The displayed name of the metrics

*   `function` : The function that takes the [History object](https://gym-trading-env.readthedocs.io/en/latest/history.html) as parameters and returns a value (you might want to prefer string over other types to avoid error).

Note

If you want to use your metrics to feed a custom logger, to visualize data or to track performance, you can access to results with `env.get_metrics()`**at the end of an episode**. In this case, it returns :

{ "Market Return" :  "25.30%", "Portfolio Return" : "45.24%", "Position Changes" : 28417, "Episode Lenght" : 33087 }

Note

> If you want to use your metrics to feed a custom logger, to visualize data or to track performance, you can access to results with `env.get_metrics()`**at the end of an episode**. In this case, it returns :

{ "Market Return" :  "25.30%", "Portfolio Return" : "45.24%", "Position Changes" : 28417, "Episode Lenght" : 33087 }
