"""
High-level module for fetching market data from Tinkoff API.
Orchestrates the process of scraping tickers, fetching FIGIs, and retrieving OHLCV data.
"""

import logging
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union

from packages.tdata.config import TINKOFF_TOKEN, INTERVAL_MAPPING, DATA_DIR, INTERVAL_TO_ENUM, MAX_PERIODS
from packages.tdata.ingestion.ticker_scraper import extract_tickers
from packages.tdata.ingestion.tinkoff_api import get_figis, fetch_ohlcv_data

# Configure logger
logger = logging.getLogger(__name__)

# Define cache directory
CACHE_DIR = os.path.join(DATA_DIR, 'cache')

def ensure_data_dir() -> None:
    """Ensure the data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        logger.info(f"Created data directory: {DATA_DIR}")

    # Also ensure cache directory exists
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)
        logger.info(f"Created cache directory: {CACHE_DIR}")

def get_rts_data(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    interval: str,
    output_file: Optional[str] = None,
    token: str = TINKOFF_TOKEN,
    force_refresh: bool = False,
    batch_size_days: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch market data for all RTS Index components.

    Args:
        start_date: Start date (string in format "YYYY-MM-DD" or datetime)
        end_date: End date (string in format "YYYY-MM-DD" or datetime)
        interval: Time interval (e.g., "1min", "5min", "1hour", "1day")
        output_file: Optional filename to save data (without extension)
        token: Tinkoff API token
        force_refresh: Whether to force refresh data from API instead of using cache
        batch_size_days: Number of days to fetch in each API batch request (if None, will be determined based on interval)

    Returns:
        DataFrame with OHLCV data for all RTS components

    Raises:
        ValueError: If interval is invalid or dates are invalid
    """
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = datetime.fromisoformat(end_date)

    # Create cache directory if it doesn't exist
    ensure_data_dir()

    # Define cache file path
    cache_file = os.path.join(CACHE_DIR, f"rts_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{interval}.pkl")

    # Check if cached data exists and we're not forcing a refresh
    if os.path.exists(cache_file) and not force_refresh:
        logger.info(f"Loading data from cache: {cache_file}")
        return pd.read_pickle(cache_file)

    # Validate interval
    if interval not in INTERVAL_MAPPING:
        valid_intervals = ", ".join(INTERVAL_MAPPING.keys())
        raise ValueError(f"Invalid interval: {interval}. Valid intervals: {valid_intervals}")

    # Map friendly interval name to Tinkoff API format
    tinkoff_interval = interval  # We're now passing the friendly interval name directly

    logger.info(f"Fetching RTS data from {start_date} to {end_date} with interval {interval}")

    # Extract tickers from TradingView
    tickers = extract_tickers()
    logger.info(f"Extracted {len(tickers)} tickers from TradingView")

    # Get FIGIs for tickers
    ticker_figis = get_figis(tickers, token)
    logger.info(f"Mapped {len(ticker_figis)} tickers to FIGIs")

    # Determine appropriate batch size based on interval if not specified
    if batch_size_days is None:
        # Get enum name for the interval
        interval_enum_name = None
        for k, v in INTERVAL_TO_ENUM.items():
            if k == interval:
                interval_enum_name = str(v).split('.')[-1]
                break

        # Set batch size based on MAX_PERIODS configuration
        if interval_enum_name and interval_enum_name in MAX_PERIODS:
            batch_size_days = max(1, int(MAX_PERIODS[interval_enum_name] * 0.9))  # Use 90% of max to be safe
        else:
            # Default to conservative batch size if interval not found
            batch_size_days = 1

    logger.info(f"Using batch size of {batch_size_days} days for interval {interval}")

    # Implement batch processing by date ranges
    all_data_rows = []
    current_start = start_date

    while current_start < end_date:
        # Calculate the end date for this batch
        batch_end = min(current_start + timedelta(days=batch_size_days), end_date)

        logger.info(f"Fetching batch data from {current_start} to {batch_end}")

        # Fetch OHLCV data for this batch
        batch_data = fetch_ohlcv_data(ticker_figis, current_start, batch_end, tinkoff_interval, token)

        # Add batch data to all data
        all_data_rows.extend(batch_data)

        # Move to next batch
        current_start = batch_end + timedelta(microseconds=1)  # Add microsecond to avoid overlap

        logger.info(f"Fetched {len(batch_data)} data points for batch ending {batch_end}")

    # Convert all collected data to DataFrame
    df = pd.DataFrame(all_data_rows)

    # Check if we got any data
    if df.empty:
        logger.warning("No data was fetched from the API")
    else:
        # Save to cache
        logger.info(f"Saving data to cache: {cache_file}")
        df.to_pickle(cache_file)

        # Save to file if output_file is specified
        if output_file:
            file_path = os.path.join(DATA_DIR, f"{output_file}.csv")
            df.to_csv(file_path, index=False)
            logger.info(f"Saved data to {file_path}")

    return df
