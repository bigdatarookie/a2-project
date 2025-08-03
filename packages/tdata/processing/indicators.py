"""
Module for calculating technical indicators for financial time series data.
"""

import logging
import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import List, Optional, Union
import warnings

# Filter out the specific warning about timezone conversion
warnings.filterwarnings('ignore', message='Converting to PeriodArray/Index representation will drop timezone information.')

# Configure logger
logger = logging.getLogger(__name__)

def calculate_indicators(df: pd.DataFrame, drop_na_rows: bool = True, min_coverage: float = 0.9) -> pd.DataFrame:
    """
    Calculate technical indicators for a given DataFrame.

    Args:
        df: DataFrame with OHLCV data. Must contain 'high', 'low', 'close', 'volume' columns and date pd.DatetimeIndex.
          If 'tic' column is present, indicators will be calculated per ticker.
          Column names are case-insensitive.
        drop_na_rows: If True, removes rows where any indicator is NaN. This ensures all indicators
          are properly calculated for the remaining data. If False, keeps all rows, with NaN values
          for indicators that couldn't be calculated at certain points.
        min_coverage: Minimum required data coverage (0-1) for each ticker. Tickers with less coverage
          will be filtered out. Default is 0.9 (90% coverage).

    Returns:
        DataFrame with added technical indicators and filtered/trimmed data for consistent timeframes.

    Raises:
        ValueError: If required columns are missing
    """
    # Make a copy of the dataframe to avoid modifying the original
    df = df.copy()

    # Convert column names to lowercase for case-insensitive handling
    df.columns = [col.lower() for col in df.columns]
    
    # Convert numeric columns to float type
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    required_columns = ['high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    def calculate_single_df(data: pd.DataFrame) -> pd.DataFrame:
        """Helper function to calculate indicators for a single DataFrame"""
        try:
            # Ensure we have a proper DatetimeIndex and data is sorted
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            data = data.sort_index()
            
            # Create clean series with the DatetimeIndex and remove any NaN values
            data = data.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            close_series = pd.Series(data['close'].values, index=data.index, name='close')
            
            # RSI
            rsi = ta.rsi(close_series, length=14)
            data['rsi'] = rsi if rsi is not None else np.nan
            
            # MACD
            macd = ta.macd(close_series, fast=12, slow=26, signal=9)
            if isinstance(macd, pd.DataFrame):
                data['macd_line'] = macd.get('MACD_12_26_9', np.nan)
                data['macd_signal'] = macd.get('MACDs_12_26_9', np.nan)
                data['macd_histogram'] = macd.get('MACDh_12_26_9', np.nan)
            else:
                data['macd_line'] = data['macd_signal'] = data['macd_histogram'] = np.nan
            
            # Create clean series for each input
            high_series = pd.Series(data['high'].values, index=data.index)
            low_series = pd.Series(data['low'].values, index=data.index)
            volume_series = pd.Series(data['volume'].values, index=data.index)
            
            # Moving Averages
            data['sma_12'] = ta.sma(close_series, length=12)
            data['sma_26'] = ta.sma(close_series, length=26)
            data['sma_50'] = ta.sma(close_series, length=50)
            data['sma_100'] = ta.sma(close_series, length=100)
            
            # ATR
            data['atr'] = ta.atr(high_series, low_series, close_series, length=14)
            
            # Bollinger Bands
            bb = ta.bbands(close_series, length=20, std=2)
            if isinstance(bb, pd.DataFrame):
                data['bb_lower'] = bb['BBL_20_2.0']
                data['bb_middle'] = bb['BBM_20_2.0']
                data['bb_upper'] = bb['BBU_20_2.0']
            else:
                data['bb_lower'] = data['bb_middle'] = data['bb_upper'] = np.nan
            
            # VWAP (calculate manually to avoid pandas_ta datetime issues)
            try:
                # Calculate typical price
                typical_price = (high_series + low_series + close_series) / 3
                # Calculate VWAP
                cumulative_tp_vol = (typical_price * volume_series).cumsum()
                cumulative_vol = volume_series.cumsum()
                data['vwap'] = cumulative_tp_vol / cumulative_vol
            except Exception as e:
                logger.warning(f"VWAP calculation failed: {str(e)}")
                data['vwap'] = np.nan
            
            # ADX
            adx = ta.adx(high_series, low_series, close_series, length=14)
            if isinstance(adx, pd.DataFrame):
                data['adx'] = adx['ADX_14']
                data['adx_positive'] = adx['DMP_14']
                data['adx_negative'] = adx['DMN_14']
            else:
                data['adx'] = data['adx_positive'] = data['adx_negative'] = np.nan
            
            # Stochastic
            stoch = ta.stoch(high_series, low_series, close_series, k=14, d=3, smooth_k=3)
            if isinstance(stoch, pd.DataFrame):
                data['stoch_k'] = stoch['STOCHk_14_3_3']
                data['stoch_d'] = stoch['STOCHd_14_3_3']
            else:
                data['stoch_k'] = data['stoch_d'] = np.nan
            
            # OBV
            data['obv'] = ta.obv(close_series, volume_series)
            
            # Keep the DatetimeIndex
            return data
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            # Add NaN columns for all indicators
            for col in ['rsi', 'macd_line', 'macd_signal', 'macd_histogram',
                       'sma_12', 'sma_26', 'sma_50', 'sma_100', 'atr',
                       'bb_lower', 'bb_middle', 'bb_upper', 'vwap',
                       'adx', 'adx_positive', 'adx_negative', 
                       'stoch_k', 'stoch_d', 'obv']:
                data[col] = np.nan
            return data

    # Process data based on whether it contains multiple tickers
    if 'tic' in df.columns:
        # Multi-ticker case
        result_dfs = {}
        total_periods = len(df.index.unique())
        
        # Calculate coverage for each ticker
        ticker_coverage = {}
        for ticker in sorted(df['tic'].unique()):
            ticker_data = df[df['tic'] == ticker].copy()
            coverage = len(ticker_data) / total_periods
            ticker_coverage[ticker] = coverage
            
            # Log coverage statistics
            coverage_pct = coverage * 100
            status = "✓" if coverage >= min_coverage else "✗"
            logger.info(f"Ticker {ticker}: {coverage_pct:.1f}% coverage {status}")
        
        # Filter tickers meeting the coverage threshold
        valid_tickers = [t for t, c in ticker_coverage.items() if c >= min_coverage]
        logger.info(f"\nData Quality Statistics:")
        logger.info(f"Total timeframes: {total_periods}")
        logger.info(f"Original tickers: {len(ticker_coverage)}")
        logger.info(f"Tickers with {min_coverage*100}% coverage: {len(valid_tickers)}")
        
        # Process valid tickers
        min_length = float('inf')
        for ticker in valid_tickers:
            ticker_data = df[df['tic'] == ticker].copy()
            try:
                processed_df = calculate_single_df(ticker_data)
                result_dfs[ticker] = processed_df
                min_length = min(min_length, len(processed_df))
                logger.info(f"Successfully processed ticker {ticker}")
            except Exception as e:
                logger.error(f"Error processing ticker {ticker}: {str(e)}")
                continue
        
        if result_dfs:
            # Trim all DataFrames to the same length and concatenate
            result = pd.DataFrame()
            for ticker in valid_tickers:
                if ticker in result_dfs:
                    ticker_data = result_dfs[ticker].iloc[:min_length]
                    result = pd.concat([result, ticker_data])
            
            logger.info(f"\nFinal dataset statistics:")
            logger.info(f"Valid tickers: {', '.join(valid_tickers)}")
            logger.info(f"Dataset length: {min_length} periods")
            
            # Clean up NaN values if requested
            if drop_na_rows:
                indicator_columns = ['rsi', 'macd_line', 'macd_signal', 'macd_histogram',
                                  'sma_12', 'sma_26', 'sma_50', 'sma_100', 'atr',
                                  'bb_lower', 'bb_middle', 'bb_upper', 'vwap',
                                  'adx', 'adx_positive', 'adx_negative',
                                  'stoch_k', 'stoch_d', 'obv']
                result = result.dropna(subset=indicator_columns)
                logger.info(f"Removed rows with NaN indicators. Remaining rows: {len(result)}")
            
            return result
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