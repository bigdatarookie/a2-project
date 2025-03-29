"""
High-level module for fetching market data from Tinkoff API.
Orchestrates the process of scraping tickers, fetching FIGIs, and retrieving OHLCV data.
"""

import logging
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

from tdata.config import TINKOFF_TOKEN, INTERVAL_MAPPING, DATA_DIR
from tdata.ingestion.ticker_scraper import extract_tickers
from tdata.ingestion.tinkoff_api import get_figis, fetch_ohlcv_data

# Configure logger
logger = logging.getLogger(__name__)

def ensure_data_dir() -> None:
    """Ensure the data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        logger.info(f"Created data directory: {DATA_DIR}")

def get_rts_data(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    interval: str,
    output_file: Optional[str] = None,
    token: str = TINKOFF_TOKEN
) -> pd.DataFrame:
    """
    Fetch market data for all RTS Index components.
    
    Args:
        start_date: Start date (string in format "YYYY-MM-DD" or datetime)
        end_date: End date (string in format "YYYY-MM-DD" or datetime)
        interval: Time interval (e.g., "1min", "5min", "1hour", "1day")
        output_file: Optional filename to save data (without extension)
        token: Tinkoff API token
    
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
    
    # Validate interval
    if interval not in INTERVAL_MAPPING:
        valid_intervals = ", ".join(INTERVAL_MAPPING.keys())
        raise ValueError(f"Invalid interval: {interval}. Valid intervals: {valid_intervals}")
    
    # Map friendly interval name to Tinkoff API format
    tinkoff_interval = INTERVAL_MAPPING[interval]
    
    logger.info(f"Fetching RTS data from {start_date} to {end_date} with interval {interval}")
    
    # Extract tickers from TradingView
    tickers = extract_tickers()
    logger.info(f"Extracted {len(tickers)} tickers from TradingView")
    
    # Get FIGIs for tickers
    ticker_figis = get_figis(tickers, token)
    logger.info(f"Mapped {len(ticker_figis)} tickers to FIGIs")
    
    # Fetch OHLCV data
    data_rows = fetch_ohlcv_data(ticker_figis, start_date, end_date, tinkoff_interval, token)
    
    # Convert to DataFrame
    df = pd.DataFrame(data_rows)
    
    # Save to file if output_file is specified
    if output_file:
        ensure_data_dir()
        file_path = f"{DATA_DIR}/{output_file}.csv"
        df.to_csv(file_path, index=False)
        logger.info(f"Saved data to {file_path}")
    
    return df 