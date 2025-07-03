"""
Convert every *.csv file inside INPUT_DIR to a *.pkl file that
MultiDatasetTradingEnv can read.

• INPUT  (one per file)
    date,open,high,low,close,volume
    2023-06-26 09:30:00-04:00,186.83,…,9851788
    …

• OUTPUT (same name, .pkl extension)
    DatetimeIndex               (UTC-naive, ascending)
    open, high, low, close,
    volume,
    feature_close_ret,          # examples – add more if you like
    feature_range,
    feature_vol_norm
"""

import os
from pathlib import Path
import glob
import pandas as pd
import numpy as np

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
INPUT_DIR = "1d-2005"        # folder that contains Yahoo-Finance csv files

# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
for csv_path in glob.glob(f"{INPUT_DIR}/**/*.csv", recursive=True):

    # 1) ─────── read & clean ───────────────────────────────────────────────
    df = pd.read_csv(csv_path)

    # date → datetime, drop timezone (tz-naive is safer with gym env)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)

    # enforce dtypes
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df.set_index("date", drop=True, inplace=True)
    df.sort_index(inplace=True)
    df.drop_duplicates(inplace=True)

    # 2) ─────── add features & final cleanup ───────────────────────────────
    df.dropna(inplace=True)           # REQUIRED → no NaNs in the env

    # 3) ─────── save as pickle ─────────────────────────────────────────────
    symbol_name = Path(csv_path).stem   # e.g. 'AAPL_30m'
    pkl_path = f"{Path(csv_path).parent}/{symbol_name}.pkl"
    df.to_pickle(pkl_path)

    # 4) ─────── remove original CSV file ───────────────────────────────────
    os.remove(csv_path)

    print(f"✓ {symbol_name}  →  {pkl_path} (original CSV removed)")

print("All files converted.")
