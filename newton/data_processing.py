import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle
import os


class YahooFinanceDataProcessor:
    def __init__(self, data):
        self.data = data  # Path object to CSV file
        self.price_scaler = StandardScaler()
        self.volume_scaler = StandardScaler()

    def process_data(self):
        # Read CSV data from path
        df = pd.read_csv(self.data)

        # Convert date column to datetime and set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Select and rename columns for gym_anytrading format (OHLCV)
        processed_data = df[['open', 'high', 'low', 'close', 'volume']].copy()
        processed_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        # Remove any rows with missing values
        processed_data.dropna(inplace=True)

        # Sort by date to ensure chronological order
        processed_data.sort_index(inplace=True)

        # Split data chronologically: 70% train, 10% validation, 20% test
        total_len = len(processed_data)
        train_len = int(0.7 * total_len)
        val_len = int(0.1 * total_len)

        train_data = processed_data[:train_len].copy()
        val_data = processed_data[train_len:train_len + val_len].copy()
        test_data = processed_data[train_len + val_len:].copy()

        # Fit scalers on training data only
        price_columns = ['Open', 'High', 'Low', 'Close']
        self.price_scaler.fit(train_data[price_columns])
        self.volume_scaler.fit(train_data[['Volume']])

        # Apply standardization to all splits using training set parameters
        train_data[price_columns] = self.price_scaler.transform(
            train_data[price_columns])
        train_data[['Volume']] = self.volume_scaler.transform(
            train_data[['Volume']])

        val_data[price_columns] = self.price_scaler.transform(
            val_data[price_columns])
        val_data[['Volume']] = self.volume_scaler.transform(
            val_data[['Volume']])

        test_data[price_columns] = self.price_scaler.transform(
            test_data[price_columns])
        test_data[['Volume']] = self.volume_scaler.transform(
            test_data[['Volume']])

        return {
            'train': train_data,
            'validation': val_data,
            'test': test_data,
            'scalers': {
                'price_scaler': self.price_scaler,
                'volume_scaler': self.volume_scaler
            }
        }

    def save_processed_data(self, processed_data, save_path):
        """
        Save processed data and scalers to disk using pickle

        Args:
            processed_data: The return value from process_data()
            save_path: Path where to save the data (should end with .pkl)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump(processed_data, f)

        print(f"Processed data saved to {save_path}")

    @staticmethod
    def load_processed_data(load_path):
        """
        Load previously saved processed data

        Args:
            load_path: Path to the saved data file

        Returns:
            Dictionary containing train, validation, test data and scalers
        """
        load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"No saved data found at {load_path}")

        with open(load_path, 'rb') as f:
            processed_data = pickle.load(f)

        print(f"Processed data loaded from {load_path}")
        return processed_data


class AlphaVantageDataProcessor:
    def __init__(self, data):
        self.data = data

    def process_data(self):
        # Implement data processing logic here
        processed_data = self.data  # Placeholder for actual processing logic
        return processed_data

    def save_processed_data(self, processed_data, save_path):
        """
        Save processed data to disk using pickle

        Args:
            processed_data: The return value from process_data()
            save_path: Path where to save the data (should end with .pkl)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump(processed_data, f)

        print(f"Processed data saved to {save_path}")

    @staticmethod
    def load_processed_data(load_path):
        """
        Load previously saved processed data

        Args:
            load_path: Path to the saved data file

        Returns:
            Processed data
        """
        load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"No saved data found at {load_path}")

        with open(load_path, 'rb') as f:
            processed_data = pickle.load(f)

        print(f"Processed data loaded from {load_path}")
        return processed_data
