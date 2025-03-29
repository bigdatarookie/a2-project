"""
Module for processing financial data files and calculating technical indicators.
"""

import os
import logging
import pandas as pd
from typing import Optional, Union, List

from packages.tdata.config import DATA_DIR, CACHE_DIR
from packages.tdata.processing.indicators import calculate_indicators

# Configure logger
logger = logging.getLogger(__name__)

def process_file_with_indicators(
    input_file: str,
    output_file: Optional[str] = None,
    drop_na: bool = True,
    optimize_types: bool = True,
    force_recalculate: bool = False
) -> pd.DataFrame:
    """
    Process a CSV file with OHLCV data to calculate technical indicators.

    Args:
        input_file: Path to the input CSV file (absolute or relative to current directory)
        output_file: Optional name for output file (without extension)
        drop_na: Whether to drop rows with NaN values after calculation
        optimize_types: Whether to optimize data types to reduce memory usage
        force_recalculate: Whether to force recalculation of indicators even if cached result exists

    Returns:
        DataFrame with calculated indicators

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If required columns are missing
    """
    logger.info(f"Processing file: {input_file}")

    # Check if input_file exists
    if not os.path.isfile(input_file):
        # Try finding it in DATA_DIR
        data_dir_path = os.path.join(DATA_DIR, input_file)
        if os.path.isfile(data_dir_path):
            input_file = data_dir_path
        else:
            raise FileNotFoundError(f"Input file not found: {input_file}")

    # Extract just the filename without path for cache naming
    input_filename = os.path.basename(input_file)
    base_filename = os.path.splitext(input_filename)[0]

    # Define cache file path
    cache_file = os.path.join(CACHE_DIR, f"{base_filename}_processed.pkl")

    # Check if cached processed data exists and if we're not forcing recalculation
    if os.path.exists(cache_file) and not force_recalculate:
        logger.info(f"Loading processed data from cache: {cache_file}")
        processed_df = pd.read_pickle(cache_file)

        # Save to output file if specified
        if output_file:
            # Ensure data directory exists
            os.makedirs(DATA_DIR, exist_ok=True)

            # Define output path
            output_path = os.path.join(DATA_DIR, f"{output_file}.csv")

            # Save to CSV
            processed_df.to_csv(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")

        return processed_df

    # Read the CSV file with parsed dates
    df = pd.read_csv(input_file, parse_dates=["date"])
    logger.info(f"Loaded {len(df)} rows from {input_file}")

    # Remove timezone information from 'date' if present (to prevent VWAP warnings)
    if hasattr(df['date'].dt, 'tz') and df['date'].dt.tz is not None:
        df['date'] = df['date'].dt.tz_convert(None).dt.tz_localize(None)
        logger.info("Removed timezone information from date column")

    # Rename columns for indicator calculation (which expects capitalized names)
    column_remap = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    for old_col, new_col in column_remap.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)

    # Optimize data types to reduce memory usage
    if optimize_types:
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[numeric_cols] = df[numeric_cols].astype('float32')
        logger.info("Optimized numeric data types")

    # Sort by ticker and date
    df.sort_values(['tic', 'date'], inplace=True)

    # Remove duplicate entries based on ticker and date
    original_len = len(df)
    df = df.drop_duplicates(subset=['tic', 'date'], keep='first')
    if len(df) < original_len:
        logger.info(f"Removed {original_len - len(df)} duplicate entries")

    # Calculate indicators per ticker group
    logger.info("Calculating technical indicators for each ticker")
    result_dfs = []

    for ticker, group in df.groupby('tic'):
        try:
            processed_group = calculate_indicators(group)
            result_dfs.append(processed_group)
            logger.info(f"Processed ticker: {ticker} ({len(processed_group)} rows)")
        except Exception as e:
            logger.error(f"Error processing ticker {ticker}: {str(e)}")
            # Continue with other tickers even if one fails

    # Combine all processed ticker groups
    processed_df = pd.concat(result_dfs, ignore_index=True)

    # Drop rows with NaN values if requested
    if drop_na:
        before_len = len(processed_df)
        processed_df = processed_df.dropna()
        dropped = before_len - len(processed_df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with NaN values")

    # Save to cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    processed_df.to_pickle(cache_file)
    logger.info(f"Saved processed data to cache: {cache_file}")

    # Save to output file if specified
    if output_file:
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)

        # Define output path
        output_path = os.path.join(DATA_DIR, f"{output_file}.csv")

        # Save to CSV
        processed_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")

    return processed_df
