"""
Module for interacting with the Tinkoff Investments API.
Provides functions for fetching instrument data, candles, etc.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional
from tinkoff.invest.sandbox.client import SandboxClient as Client
from tinkoff.invest import SecurityTradingStatus
from tinkoff.invest.services import InstrumentsService
from tinkoff.invest.utils import quotation_to_decimal

from tdata.config import TINKOFF_TOKEN, MAX_PERIODS, DEFAULT_RATE_LIMIT_PAUSE

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

    # Define the maximum period per request for this interval
    max_duration = timedelta(days=MAX_PERIODS.get(interval, 1))

    # List to store data for all instruments
    data_rows = []

    total_figis = len(ticker_figis)
    total_data_points = 0

    with Client(token) as client:
        for figi_index, (ticker, figi) in enumerate(ticker_figis, start=1):
            logger.info(f"Processing {ticker} ({figi}) [{figi_index}/{total_figis}]...")
            current_start = start_date
            figi_data_points = 0

            while current_start < end_date:
                # Calculate the end of the current chunk
                current_end = min(current_start + max_duration, end_date)
                try:
                    # Fetch candles for the current chunk
                    candles = client.market_data.get_candles(
                        figi=figi,
                        from_=current_start,
                        to=current_end,
                        interval=interval,
                    )

                    # Process each candle
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
                        data_rows.append(row)
                        figi_data_points += 1

                    # Move to the next time chunk
                    current_start = current_end

                except Exception as e:
                    error_message = str(e)
                    logger.error(f"Error fetching data for {ticker} ({figi}): {error_message}")

                    # Handle rate limiting
                    if "RESOURCE_EXHAUSTED" in error_message:
                        # Try to extract reset time from metadata if available
                        wait_time = DEFAULT_RATE_LIMIT_PAUSE
                        if hasattr(e, 'args') and len(e.args) > 0:
                            metadata = getattr(e.args[-1], 'ratelimit_reset', None)
                            if metadata:
                                try:
                                    wait_time = int(metadata)
                                except (ValueError, TypeError):
                                    pass

                        logger.warning(f"Rate limit exceeded. Waiting for {wait_time} seconds...")
                        time.sleep(wait_time)
                        # Don't advance the start date, retry same chunk
                    else:
                        # For other errors, skip to next chunk to try to continue
                        logger.warning(f"Skipping to next time chunk due to error")
                        current_start = current_end

            logger.info(f"Completed {ticker}: {figi_data_points} data points")
            total_data_points += figi_data_points

    logger.info(f"Total data points fetched: {total_data_points}")
    return data_rows
