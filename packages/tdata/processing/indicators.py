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
        df: DataFrame with OHLCV data. Must contain 'high', 'low', 'close', 'volume', 'date' columns.
          If 'tic' column is present, indicators will be calculated per ticker.
          Column names are case-insensitive.

    Returns:
        DataFrame with added technical indicators

    Raises:
        ValueError: If required columns are missing
    """
    # Make a copy of the dataframe to avoid modifying the original
    df = df.copy()

    # Convert column names to lowercase for case-insensitive handling
    df.columns = [col.lower() for col in df.columns]

    required_columns = ['high', 'low', 'close', 'volume', 'date']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    def calculate_single_df(data: pd.DataFrame) -> pd.DataFrame:
        """Helper function to calculate indicators for a single DataFrame"""
        try:
            # Sort by date and set index
            data = data.sort_values('date').set_index('date')
            
            # RSI
            rsi = ta.rsi(data['close'], length=14)
            data['rsi'] = rsi if rsi is not None else np.nan
            
            # MACD
            macd = ta.macd(data['close'], fast=12, slow=26, signal=9)
            if isinstance(macd, pd.DataFrame):
                data['macd_line'] = macd.get('MACD_12_26_9', np.nan)
                data['macd_signal'] = macd.get('MACDs_12_26_9', np.nan)
                data['macd_histogram'] = macd.get('MACDh_12_26_9', np.nan)
            else:
                data['macd_line'] = data['macd_signal'] = data['macd_histogram'] = np.nan
            
            # Moving Averages
            data['sma_12'] = ta.sma(data['close'], length=12)
            data['sma_26'] = ta.sma(data['close'], length=26)
            data['sma_50'] = ta.sma(data['close'], length=50)
            data['sma_100'] = ta.sma(data['close'], length=100)
            
            # ATR
            data['atr'] = ta.atr(data['high'], data['low'], data['close'], length=14)
            
            # Bollinger Bands
            bb = ta.bbands(data['close'], length=20, std=2)
            if isinstance(bb, pd.DataFrame):
                data['bb_lower'] = bb['BBL_20_2.0']
                data['bb_middle'] = bb['BBM_20_2.0']
                data['bb_upper'] = bb['BBU_20_2.0']
            else:
                data['bb_lower'] = data['bb_middle'] = data['bb_upper'] = np.nan
            
            # VWAP
            data['vwap'] = ta.vwap(data['high'], data['low'], data['close'], data['volume'])
            
            # ADX
            adx = ta.adx(data['high'], data['low'], data['close'], length=14)
            if isinstance(adx, pd.DataFrame):
                data['adx'] = adx['ADX_14']
                data['adx_positive'] = adx['DMP_14']
                data['adx_negative'] = adx['DMN_14']
            else:
                data['adx'] = data['adx_positive'] = data['adx_negative'] = np.nan
            
            # Stochastic
            stoch = ta.stoch(data['high'], data['low'], data['close'], k=14, d=3, smooth_k=3)
            if isinstance(stoch, pd.DataFrame):
                data['stoch_k'] = stoch['STOCHk_14_3_3']
                data['stoch_d'] = stoch['STOCHd_14_3_3']
            else:
                data['stoch_k'] = data['stoch_d'] = np.nan
            
            # OBV
            data['obv'] = ta.obv(data['close'], data['volume'])
            
            # Reset index to get date back as column
            return data.reset_index()
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            # Add NaN columns for all indicators
            for col in ['rsi', 'macd_line', 'macd_signal', 'macd_histogram',
                       'sma_12', 'sma_26', 'sma_50', 'sma_100', 'atr',
                       'bb_lower', 'bb_middle', 'bb_upper', 'vwap',
                       'adx', 'adx_positive', 'adx_negative', 
                       'stoch_k', 'stoch_d', 'obv']:
                data[col] = np.nan
            return data.reset_index() if isinstance(data.index, pd.DatetimeIndex) else data

    # Process data based on whether it contains multiple tickers
    if 'tic' in df.columns:
        # Multi-ticker case
        result_dfs = []
        for ticker in df['tic'].unique():
            ticker_data = df[df['tic'] == ticker].copy()
            if len(ticker_data) < 100:
                logger.warning(f"Not enough data points ({len(ticker_data)}) for ticker {ticker}")
                continue
            try:
                processed_df = calculate_single_df(ticker_data)
                result_dfs.append(processed_df)
                logger.info(f"Successfully processed ticker {ticker}")
            except Exception as e:
                logger.error(f"Error processing ticker {ticker}: {str(e)}")
                continue
        
        if result_dfs:
            logger.info(f"Successfully processed {len(result_dfs)} tickers")
            return pd.concat(result_dfs, ignore_index=True)
        else:
            logger.error("No tickers were processed successfully")
            return df
    else:
        # Single ticker case
        if len(df) < 100:
            logger.warning(f"Not enough data points ({len(df)}) for calculating indicators")
            for indicator in ['rsi', 'macd_line', 'macd_signal', 'macd_histogram',
                            'sma_12', 'sma_26', 'sma_50', 'sma_100', 'atr',
                            'bb_lower', 'bb_middle', 'bb_upper', 'vwap',
                            'adx', 'adx_positive', 'adx_negative', 'stoch_k', 'stoch_d', 'obv']:
                df[indicator] = np.nan
            return df
        
        return calculate_single_df(df)