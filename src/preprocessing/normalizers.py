"""
Data normalization and scaling techniques for trading data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from typing import Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class TradingDataNormalizer:
    """
    Comprehensive normalizer for trading data that supports multiple scaling methods
    and maintains market-agnostic features for generalization.
    """

    def __init__(self, method: str = 'minmax', feature_range: Tuple[float, float] = (0, 1)):
        """
        Initialize the normalizer.

        Args:
            method: Normalization method ('minmax', 'standard', 'robust', 'percentage_change')
            feature_range: Range for MinMaxScaler
        """
        self.method = method.lower()
        self.feature_range = feature_range
        self.scalers = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> 'TradingDataNormalizer':
        """
        Fit the normalizer on training data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            self
        """
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(
                f"DataFrame must contain columns: {required_cols}")

        if self.method == 'percentage_change':
            # For percentage change, we don't need to fit scalers
            self.fitted = True
            return self

        # Fit scalers for price columns
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df.columns:
                self.scalers[col] = self._create_scaler()
                self.scalers[col].fit(df[[col]])

        # Handle volume separately (often has very different scale)
        if 'Volume' in df.columns:
            self.scalers['Volume'] = self._create_scaler()
            self.scalers['Volume'].fit(df[['Volume']])

        self.fitted = True
        logger.info(f"Normalizer fitted with method: {self.method}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted normalizer.

        Args:
            df: DataFrame to normalize

        Returns:
            Normalized DataFrame
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")

        result = df.copy()

        if self.method == 'percentage_change':
            return self._percentage_change_transform(result)

        # Apply fitted scalers
        for col, scaler in self.scalers.items():
            if col in result.columns:
                result[col] = scaler.transform(result[[col]]).flatten()

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            df: DataFrame to fit and transform

        Returns:
            Normalized DataFrame
        """
        return self.fit(df).transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform normalized data back to original scale.

        Args:
            df: Normalized DataFrame

        Returns:
            DataFrame in original scale
        """
        if not self.fitted:
            raise ValueError(
                "Normalizer must be fitted before inverse transform")

        if self.method == 'percentage_change':
            raise NotImplementedError(
                "Inverse transform not available for percentage_change method")

        result = df.copy()
        for col, scaler in self.scalers.items():
            if col in result.columns:
                result[col] = scaler.inverse_transform(result[[col]]).flatten()

        return result

    def _create_scaler(self):
        """Create scaler based on method."""
        if self.method == 'minmax':
            return MinMaxScaler(feature_range=self.feature_range)
        elif self.method == 'standard':
            return StandardScaler()
        elif self.method == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    def _percentage_change_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform using percentage changes - highly effective for market-agnostic features.
        This method is particularly good for generalization across different assets.
        """
        result = df.copy()

        # Calculate percentage changes for price columns
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in result.columns:
                result[f'{col}_pct'] = result[col].pct_change().fillna(0)

        # For volume, use log transformation + percentage change
        if 'Volume' in result.columns:
            # Handle zero volumes
            volume_safe = result['Volume'].replace(0, 1)
            result['Volume_log_pct'] = np.log(
                volume_safe).pct_change().fillna(0)

        # Add relative price features (market-agnostic)
        if all(col in result.columns for col in ['High', 'Low', 'Close']):
            result['HL_ratio'] = (
                result['High'] - result['Low']) / result['Close']
            result['OC_ratio'] = (
                result['Open'] - result['Close']) / result['Close']

        # Keep original columns or replace them
        return result


def normalize_for_training(df: pd.DataFrame,
                           method: str = 'percentage_change',
                           fit_on_training: bool = True) -> Tuple[pd.DataFrame, TradingDataNormalizer]:
    """
    Convenience function to normalize trading data for training.

    Args:
        df: DataFrame with OHLCV data
        method: Normalization method
        fit_on_training: Whether to fit normalizer on this data

    Returns:
        Tuple of (normalized_df, fitted_normalizer)
    """
    normalizer = TradingDataNormalizer(method=method)

    if fit_on_training:
        normalized_df = normalizer.fit_transform(df)
    else:
        normalized_df = normalizer.transform(df)

    return normalized_df, normalizer


def create_market_agnostic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create market-agnostic features that work across different assets.
    These features are scale-invariant and focus on relative movements.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with additional market-agnostic features
    """
    result = df.copy()

    # Technical indicators that are market-agnostic
    if all(col in df.columns for col in ['High', 'Low', 'Close']):
        # Price position within high-low range
        result['price_position'] = (
            result['Close'] - result['Low']) / (result['High'] - result['Low'])
        result['price_position'] = result['price_position'].fillna(0.5)

        # High-Low spread relative to close
        result['hl_spread'] = (
            result['High'] - result['Low']) / result['Close']

    if 'Close' in df.columns:
        # Returns (most important market-agnostic feature)
        result['returns'] = result['Close'].pct_change().fillna(0)

        # Volatility (rolling standard deviation of returns)
        result['volatility'] = result['returns'].rolling(
            window=5, min_periods=1).std().fillna(0)

        # Price momentum
        result['momentum_5'] = result['Close'].pct_change(5).fillna(0)
        result['momentum_10'] = result['Close'].pct_change(10).fillna(0)

    if 'Volume' in df.columns and 'Close' in df.columns:
        # Volume-price trend
        result['volume_price_trend'] = (
            result['Volume'] * result['Close']).pct_change().fillna(0)

    return result
