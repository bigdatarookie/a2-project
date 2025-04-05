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
        interval: Candle interval (e.g., "1min", "5min", "1hour", "1day")
        token: Tinkoff API token

    Returns:
        List of dictionaries with OHLCV data for all requested instruments

    Raises:
        ValueError: If token is not provided or is invalid
    """
    if not token:
        raise ValueError("Tinkoff API token is required")

    logger.info(f"Fetching OHLCV data for {len(ticker_figis)} instruments from {start_date} to {end_date}")

    # If no ticker_figis provided, return empty list immediately
    if not ticker_figis:
        return []

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

    # Get the maximum period allowed for this interval from MAX_PERIODS
    # Use the string name of the enum as the key
    enum_name = str(interval_enum).split('.')[-1]

    # Use a safe default of 1 day if the interval is not found in MAX_PERIODS
    max_days = MAX_PERIODS.get(enum_name, 1)

    # Define the maximum period per request for this interval
    max_duration = timedelta(days=max_days)

    logger.info(f"Using maximum period of {max_days} days for interval {interval}")

    # List to store data for all instruments
    all_data = []

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

                    # Check for period exceeded error (code 30014)
                    if '30014' in error_message or 'maximum request period' in error_message.lower():
                        # If we hit this error, reduce the time range and try again
                        current_days = (to_date - from_date).days
                        new_days = max(1, current_days // 2)  # Reduce by half but ensure at least 1 day

                        logger.warning(f"Maximum period exceeded for {interval}. Reducing from {current_days} to {new_days} days")

                        # Update MAX_PERIODS for future reference
                        MAX_PERIODS[enum_name] = min(MAX_PERIODS.get(enum_name, float('inf')), new_days)

                        # If we can't reduce further, raise the error
                        if current_days <= 1:
                            raise

                        # Return with a signal to retry with smaller window
                        return {"retry_with_smaller_window": True, "new_days": new_days}

                    elif "RESOURCE_EXHAUSTED" in error_message:
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
        def process_ticker(ticker_figi):
            nonlocal max_duration  # Explicitly indicate we're using the outer scope variable
            ticker, figi = ticker_figi
            logger.info(f"Processing {ticker} ({figi})...")
            data_points = []
            current_start = start_date

            # Store the duration locally to allow per-ticker adjustments
            ticker_max_duration = max_duration

            while current_start < end_date:
                # Calculate end date for this batch
                current_end = min(current_start + ticker_max_duration, end_date)

                try:
                    result = get_candles_with_retry(client, figi, current_start, current_end, interval_enum)

                    # Check if we need to retry with a smaller window
                    if isinstance(result, dict) and result.get("retry_with_smaller_window"):
                        # Adjust max_duration based on the smaller window
                        new_days = result.get("new_days", 1)
                        ticker_max_duration = timedelta(days=new_days)
                        # Also update the shared max_duration to help other tickers avoid the same error
                        max_duration = min(max_duration, ticker_max_duration)
                        logger.info(f"Retrying with smaller window of {new_days} days")
                        continue

                    # Process candles...
                    for candle in result.candles:
                        row = {
                            "figi": figi,
                            "date": candle.time,
                            "open": float(quotation_to_decimal(candle.open)),
                            "high": float(quotation_to_decimal(candle.high)),
                            "low": float(quotation_to_decimal(candle.low)),
                            "close": float(quotation_to_decimal(candle.close)),
                            "volume": candle.volume,
                            "tic": ticker,
                            "day": candle.time.weekday(),
                        }
                        data_points.append(row)
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {str(e)}")

                # Move to next time period
                current_start = current_end

            return data_points

        # Use ThreadPoolExecutor with limited workers to avoid hitting rate limits too hard
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            for ticker_figi in ticker_figis:
                futures[executor.submit(process_ticker, ticker_figi)] = ticker_figi

            for future in as_completed(futures):
                ticker_data = future.result()
                all_data.extend(ticker_data)

    logger.info(f"Total data points fetched: {len(all_data)}")
    return all_data
