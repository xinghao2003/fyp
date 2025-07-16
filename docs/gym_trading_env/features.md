Title: Features

URL Source: http://gym-trading-env.readthedocs.io/en/latest/features.html

Markdown Content:
[Back to top](http://gym-trading-env.readthedocs.io/en/latest/features.html#)

Toggle table of contents sidebar

As seen previously in the tutorial. We can easily create features that will be returned as observation at each time step. This type of feature is called a **static feature** as it is computed once, at the very beggining of the DataFrame processing.

Hint

**But what if you want to use a feature that we can not pre-compute ?**

In this case, you will use a **dynamic feature** that will be computed at each step.

Create static features[#](http://gym-trading-env.readthedocs.io/en/latest/features.html#create-static-features "Permalink to this heading")
-------------------------------------------------------------------------------------------------------------------------------------------

# df is a DataFrame with columns : "open", "high", "low", "close", "Volume USD"

# Create the feature : ( close[t] - close[t-1] )/ close[t-1]
df["feature_close"] = df["close"].pct_change()

# Create the feature : open[t] / close[t]
df["feature_open"] = df["open"]/df["close"]

# Create the feature : high[t] / close[t]
df["feature_high"] = df["high"]/df["close"]

# Create the feature : low[t] / close[t]
df["feature_low"] = df["low"]/df["close"]

 # Create the feature : volume[t] / max(*volume[t-7*24:t+1])
df["feature_volume"] = df["Volume USD"] / df["Volume USD"].rolling(7*24).max()

df.dropna(inplace= True) # Clean again !
# Eatch step, the environment will return 5 static inputs : "feature_close", "feature_open", "feature_high", "feature_low", "feature_volume"

env = gym.make('TradingEnv',
  df = df,
  ....
)

Important

The environment will recognize as inputs every column that contains the keyword ‘**feature**’ in its name.

Create dynamic features[#](http://gym-trading-env.readthedocs.io/en/latest/features.html#create-dynamic-features "Permalink to this heading")
---------------------------------------------------------------------------------------------------------------------------------------------

A **dynamic feature** is computed at each step. Be careful, dynamic features are _much less efficient_ in terms of computing time than static features.

Important

The dynamic features presented below are the default dynamic features used by the environment !

def dynamic_feature_last_position_taken(history):
    return history['position', -1]

def dynamic_feature_real_position(history):
    return history['real_position', -1]

env = gym.make(
    "TradingEnv",
    df = df,
    dynamic_feature_functions = [dynamic_feature_last_position_taken, dynamic_feature_real_position],
    ...
)

At each step, the environment will compute and add these 2 features at the end of the _observation_.
