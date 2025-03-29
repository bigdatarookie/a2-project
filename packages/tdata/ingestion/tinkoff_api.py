"""
Module for interacting with the Tinkoff Investments API.
Provides functions for fetching instrument data, candles, etc.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional
from tinkoff.invest.sandbox.client import SandboxClient as Client
from tinkoff.invest import SecurityTradingStatus, CandleInterval
from tinkoff.invest.services import InstrumentsService
from tinkoff.invest.utils import quotation_to_decimal
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from packages.tdata.config import TINKOFF_TOKEN, MAX_PERIODS, DEFAULT_RATE_LIMIT_PAUSE

# Configure logger
logger = logging.getLogger(__name__)

def get_figis(tickers: List[str], token: str = TINKOFF_TOKEN) -> List[Tuple[str, str]]:
    """
    Get FIGI identifiers for a list of ticker symbols.

    Args:
        tickers: List of ticker symbols
        token: Tinkoff API token

    Returns:
        List of tuples containing (ticker, figi) pairs

    Raises:
        ValueError: If token is not provided or is invalid
    """
    if not token:
        raise ValueError("Tinkoff API token is required")

    logger.info(f"Fetching FIGIs for {len(tickers)} tickers")
    result = []

    with Client(token) as client:
        instruments: InstrumentsService = client.instruments
        all_instruments = []

        # Fetch all instrument types from Tinkoff API
        for method in ["shares", "bonds", "etfs", "currencies", "futures"]:
            try:
                method_instruments = getattr(instruments, method)().instruments
                all_instruments.extend(method_instruments)
                logger.debug(f"Fetched {len(method_instruments)} {method}")
            except Exception as e:
                logger.error(f"Error fetching {method}: {str(e)}")

        logger.info(f"Total instruments fetched: {len(all_instruments)}")

        # Create a dictionary for quick lookup by ticker
        ticker_to_instrument = {
            item.ticker: item for item in all_instruments
        }

        # Find FIGIs for requested tickers
        for ticker in tickers:
            if ticker in ticker_to_instrument:
                instrument = ticker_to_instrument[ticker]
                result.append((ticker, instrument.figi))
                logger.debug(f"Found FIGI for {ticker}: {instrument.figi}")
            else:
                logger.warning(f"No FIGI found for ticker: {ticker}")

    logger.info(f"Successfully mapped {len(result)} tickers to FIGIs")
    return result

def fetch_ohlcv_data(
    ticker_figis: List[Tuple[str, str]],
    start_date: datetime,
    end_date: datetime,
    interval: str,
    token: str = TINKOFF_TOKEN
) -> Dict[str, Any]:
    """
    Fetch OHLCV (Open, High, Low, Close, Volume) data for multiple instruments.

    Args:
        ticker_figis: List of (ticker, figi) tuples
        start_date: Start date for data fetching
        end_date: End date for data fetching
        interval: Candle interval (e.g., "CANDLE_INTERVAL_1_MIN")
        token: Tinkoff API token

    Returns:
        DataFrame with OHLCV data for all requested instruments

    Raises:
        ValueError: If token is not provided or is invalid
    """
    if not token:
        raise ValueError("Tinkoff API token is required")

    logger.info(f"Fetching OHLCV data for {len(ticker_figis)} instruments from {start_date} to {end_date}")

    # If no ticker_figis provided, return empty list immediately
    if not ticker_figis:
        return []

    # Define the maximum period per request for this interval
    max_duration = timedelta(days=7)  # Increase from 1 day to 7 days if possible

    # List to store data for all instruments
    data_rows = []

    total_figis = len(ticker_figis)
    total_data_points = 0

    # Convert string interval to proper enum
    interval_map = {
        "1min": CandleInterval.CANDLE_INTERVAL_1_MIN,
        "5min": CandleInterval.CANDLE_INTERVAL_5_MIN,
        "15min": CandleInterval.CANDLE_INTERVAL_15_MIN,
        "1hour": CandleInterval.CANDLE_INTERVAL_HOUR,
        "2hour": CandleInterval.CANDLE_INTERVAL_2_HOUR,
        "4hour": CandleInterval.CANDLE_INTERVAL_4_HOUR,
        "1day": CandleInterval.CANDLE_INTERVAL_DAY,
        "1week": CandleInterval.CANDLE_INTERVAL_WEEK
    }

    if interval not in interval_map:
        valid_intervals = ", ".join(interval_map.keys())
        raise ValueError(f"Invalid interval: {interval}. Valid intervals: {valid_intervals}")

    interval_enum = interval_map[interval]

    with Client(token) as client:
        # Add smarter rate limit handling with exponential backoff
        def get_candles_with_retry(client, figi, from_date, to_date, interval_enum, max_retries=5):
            retries = 0
            backoff_time = DEFAULT_RATE_LIMIT_PAUSE

            while retries < max_retries:
                try:
                    return client.market_data.get_candles(
                        figi=figi,
                        from_=from_date,
                        to=to_date,
                        interval=interval_enum
                    )
                except Exception as e:
                    error_message = str(e)
                    if "RESOURCE_EXHAUSTED" in error_message:
                        # Extract reset time if available
                        reset_time = DEFAULT_RATE_LIMIT_PAUSE
                        if hasattr(e, 'metadata') and hasattr(e.metadata, 'ratelimit_reset'):
                            reset_time = int(e.metadata.ratelimit_reset)

                        # Add jitter to avoid all clients hitting at once
                        wait_time = reset_time + random.uniform(0, 2)
                        logger.warning(f"Rate limit exceeded. Waiting for {wait_time:.1f} seconds...")
                        time.sleep(wait_time)
                        retries += 1
                        # Exponential backoff
                        backoff_time *= 1.5
                    else:
                        # For other errors, raise immediately
                        raise

            # If we exhausted retries
            raise Exception(f"Max retries exceeded when fetching candles for {figi}")

        # Process tickers in parallel (with care)
        # This is a bit risky with rate limits, but can be done carefully
        def process_ticker(ticker_figi):
            ticker, figi = ticker_figi
            logger.info(f"Processing {ticker} ({figi})...")
            data_points = []
            current_start = start_date

            while current_start < end_date:
                current_end = min(current_start + max_duration, end_date)
                try:
                    candles = get_candles_with_retry(client, figi, current_start, current_end, interval_enum)
                    # Process candles...
                    for candle in candles.candles:
                        row = {
                            "figi": figi,
                            "date": candle.time,
                            "open": float(candle.open.units + candle.open.nano / 1e9),
                            "high": float(candle.high.units + candle.high.nano / 1e9),
                            "low": float(candle.low.units + candle.low.nano / 1e9),
                            "close": float(candle.close.units + candle.close.nano / 1e9),
                            "volume": candle.volume,
                            "tic": ticker,
                            "day": candle.time.weekday(),
                        }
                        data_points.append(row)
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {str(e)}")

                current_start = current_end

            return data_points

        # Use ThreadPoolExecutor with limited workers to avoid hitting rate limits too hard
        all_data = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            for ticker_figi in ticker_figis[:10]:  # Process in chunks of 10
                futures[executor.submit(process_ticker, ticker_figi)] = ticker_figi

            for future in as_completed(futures):
                ticker_data = future.result()
                all_data.extend(ticker_data)

    logger.info(f"Total data points fetched: {len(all_data)}")
    return all_data
