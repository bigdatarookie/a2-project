"""
Module for calculating technical indicators for financial time series data.
"""

import logging
import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import List, Optional, Union

# Configure logger
logger = logging.getLogger(__name__)

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for a given DataFrame.

    Args:
        df: DataFrame with OHLCV data. Must contain 'High', 'Low', 'Close', 'Volume', 'date' columns.

    Returns:
        DataFrame with added technical indicators

    Raises:
        ValueError: If required columns are missing
    """
    required_columns = ['High', 'Low', 'Close', 'Volume', 'date']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    # Check if we have enough data
    if len(df) < 100:  # Minimum required for longest indicator
        logger.warning(f"Not enough data points ({len(df)}) for calculating all indicators")
        # Return the original DataFrame to avoid errors, with NaN indicator columns
        for indicator in ['rsi', 'macd_line', 'macd_signal', 'macd_histogram',
                         'sma_12', 'sma_26', 'sma_50', 'sma_100', 'atr',
                         'bb_lower', 'bb_middle', 'bb_upper', 'vwap',
                         'adx', 'adx_positive', 'adx_negative', 'stoch_k', 'stoch_d', 'obv']:
            df[indicator] = np.nan
        return df

    # Sort by 'date' to ensure chronological order
    df = df.sort_values('date')

    # Set 'date' as DatetimeIndex for indicator calculation
    df = df.set_index('date')

    try:
        # Calculate technical indicators with error checking

        # Relative Strength Index
        rsi = ta.rsi(df['Close'], length=14)
        if rsi is not None:
            df['rsi'] = rsi
        else:
            df['rsi'] = np.nan

        # Moving Average Convergence Divergence
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        if macd is not None and isinstance(macd, pd.DataFrame):
            df['macd_line'] = macd.get('MACD_12_26_9', np.nan)
            df['macd_signal'] = macd.get('MACDs_12_26_9', np.nan)
            df['macd_histogram'] = macd.get('MACDh_12_26_9', np.nan)
        else:
            df['macd_line'] = df['macd_signal'] = df['macd_histogram'] = np.nan

        # Simple Moving Averages
        df['sma_12'] = ta.sma(df['Close'], length=12)
        df['sma_26'] = ta.sma(df['Close'], length=26)
        df['sma_50'] = ta.sma(df['Close'], length=50)
        df['sma_100'] = ta.sma(df['Close'], length=100)

        # Average True Range
        df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

        # Bollinger Bands
        bb = ta.bbands(df['Close'], length=20, std=2)
        df['bb_lower'] = bb['BBL_20_2.0']
        df['bb_middle'] = bb['BBM_20_2.0']
        df['bb_upper'] = bb['BBU_20_2.0']

        # Volume Weighted Average Price
        df['vwap'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])

        # Average Directional Index
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        df['adx'] = adx['ADX_14']
        df['adx_positive'] = adx['DMP_14']
        df['adx_negative'] = adx['DMN_14']

        # Stochastic Oscillator
        stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3, smooth_k=3)
        df['stoch_k'] = stoch['STOCHk_14_3_3']
        df['stoch_d'] = stoch['STOCHd_14_3_3']

        # On-Balance Volume
        df['obv'] = ta.obv(df['Close'], df['Volume'])

        logger.info("Successfully calculated all technical indicators")

    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        # Add default NaN values for all indicators
        for indicator in ['rsi', 'macd_line', 'macd_signal', 'macd_histogram',
                         'sma_12', 'sma_26', 'sma_50', 'sma_100', 'atr',
                         'bb_lower', 'bb_middle', 'bb_upper', 'vwap',
                         'adx', 'adx_positive', 'adx_negative', 'stoch_k', 'stoch_d', 'obv']:
            if indicator not in df.columns:
                df[indicator] = np.nan

    # Reset index to get 'date' back as a column
    df = df.reset_index()

    return df
