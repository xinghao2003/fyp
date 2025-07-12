import pandas as pd
import numpy as np
import os
import glob


def preprocess_hybrid(df: pd.DataFrame) -> pd.DataFrame:
    """
    A comprehensive preprocessing function that applies robust, feature-specific, 
    and adaptive normalization on a per-stock basis.

    This function should be passed to the `MultiDatasetTradingEnv`.

    :param df: Input DataFrame for a single stock (must have a DatetimeIndex).
    :return: DataFrame with added normalized feature columns, prefixed with 'norm_'.
    """
    # Ensure DataFrame is sorted by date and make a copy
    df = df.sort_index(ascending=True).copy()

    # --- Parameters ---
    WINDOW = 252  # Rolling window size (~1 year of trading days)
    MIN_PERIODS = 30  # Minimum periods for reliable stats
    EPS = 1e-8    # Small epsilon to avoid division by zero

    # --- 1. Price-based Features & Moving Averages/Bands ---
    # Normalize relative to the closing price to get percentage deviation.
    # This is self-normalizing and works across all price scales.
    price_features = ['open', 'high', 'low']
    ma_band_features = ['close_10_sma',
                        'close_10_ema', 'boll_ub', 'boll_lb', 'boll']

    # We normalize relative to the *previous* close to prevent any lookahead.
    # For the current 'close' price, we normalize it as the return from the previous close.
    df['norm_close'] = df['close'].pct_change().fillna(0.0)

    prev_close = df['close'].shift(1)
    for feature in price_features + ma_band_features:
        if feature in df.columns:
            # Formula: (feature_value / previous_close) - 1
            df[f'norm_{feature}'] = (df[feature] / prev_close) - 1

    # --- 2. Volume ---
    # Use log transform for skewness, then apply a rolling z-score.
    if 'volume' in df.columns:
        log_volume = np.log1p(df['volume'])

        # Use rolling z-score on the log-transformed volume
        rolling_mean = log_volume.shift(1).rolling(
            window=WINDOW, min_periods=MIN_PERIODS).mean()
        rolling_std = log_volume.shift(1).rolling(
            window=WINDOW, min_periods=MIN_PERIODS).std()
        df['norm_volume'] = (
            log_volume - rolling_mean) / (rolling_std + EPS)

    # --- 3. Bounded Oscillators (0-100 scale) ---
    # Scale to [-1, 1] to preserve the fixed meaning of levels (e.g., overbought/oversold).
    if 'rsi' in df.columns:
        df['norm_rsi'] = (df['rsi'] - 50) / 50.0

    kdj_features = ['kdjk', 'kdjd', 'kdjj']
    for feature in kdj_features:
        if feature in df.columns:
            df[f'norm_{feature}'] = (df[feature] - 50) / 50.0

    # --- 4. Unbounded Oscillators & Momentum Indicators ---
    # These benefit most from a rolling z-score to show extremity relative to recent history.
    unbounded_features = ['macd', 'adx', 'atr']
    for feature in unbounded_features:
        if feature in df.columns:
            # Rolling z-score
            rolling_mean = df[feature].shift(1).rolling(
                window=WINDOW, min_periods=MIN_PERIODS).mean()
            rolling_std = df[feature].shift(1).rolling(
                window=WINDOW, min_periods=MIN_PERIODS).std()
            df[f'norm_{feature}'] = (
                df[feature] - rolling_mean) / (rolling_std + EPS)

    # --- Final Cleanup ---
    # Select only the newly created normalized columns for cleanup
    norm_cols = [col for col in df.columns if col.startswith('norm_')]

    # Handle NaNs in normalized columns only by forward-filling them.
    df[norm_cols] = df[norm_cols].fillna(method='ffill')

    # Replace any infinite values in normalized columns only
    df[norm_cols] = df[norm_cols].replace([np.inf, -np.inf], 0.0)

    return df


# --- Usage Example ---
if __name__ == '__main__':
    # Folder containing CSV files (modify this path as needed)
    folder_path = r"1d-2005"

    # Find all CSV files recursively in the folder and subfolders
    csv_pattern = os.path.join(folder_path, "**", "*.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)

    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        exit()

    print(f"Found {len(csv_files)} CSV files to process:")
    for file in csv_files:
        print(f"  {file}")

    processed_count = 0
    error_count = 0

    for csv_file_path in csv_files:
        try:
            print(f"\nProcessing: {os.path.basename(csv_file_path)}")

            # Read CSV file
            df = pd.read_csv(csv_file_path, index_col='date', parse_dates=True)

            # Check if normalized columns already exist
            existing_norm_cols = [
                col for col in df.columns if col.startswith('norm_')]
            if existing_norm_cols:
                print(
                    f"  Skipping - already has normalized columns: {existing_norm_cols}")
                continue

            # Process the data
            processed_df = preprocess_hybrid(df)

            # Get only the new normalized columns
            norm_features = [
                col for col in processed_df.columns if col.startswith('norm_')]

            if norm_features:
                # Save back to the same file
                processed_df.to_csv(csv_file_path)
                print(
                    f"  ✓ Added {len(norm_features)} normalized columns and saved")
                processed_count += 1
            else:
                print(f"  ⚠ No normalized columns were created")

        except Exception as e:
            print(
                f"  ✗ Error processing {os.path.basename(csv_file_path)}: {e}")
            error_count += 1

    print(f"\n=== Summary ===")
    print(f"Successfully processed: {processed_count} files")
    print(f"Errors: {error_count} files")
    print(f"Total files found: {len(csv_files)}")
