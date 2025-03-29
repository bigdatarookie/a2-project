"""
Module for scraping ticker symbols from web sources like TradingView.
"""

import requests
import logging
import time
from bs4 import BeautifulSoup
from typing import List, Optional

from tdata.config import MAX_RETRIES, BACKOFF_FACTOR, REQUEST_TIMEOUT

# Configure logger
logger = logging.getLogger(__name__)

def extract_tickers(url: str = "https://www.tradingview.com/symbols/RUS-RTSI/components/") -> List[str]:
    """
    Extracts stock tickers from TradingView for RTS index components.

    Args:
        url: URL of the TradingView components page for RTS index

    Returns:
        List of ticker symbols

    Raises:
        requests.exceptions.RequestException: If all retries fail
    """
    logger.info(f"Extracting tickers from {url}")

    for attempt in range(MAX_RETRIES):
        try:
            # Send a GET request to the webpage with timeout
            logger.debug(f"Attempt {attempt + 1} to fetch tickers")
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()  # Check for request errors

            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all rows matching the pattern for RTS components
            rows = soup.find_all('tr', class_="row-RdUXZpkv listRow")

            # Extract tickers from the 'data-rowkey' attribute
            tickers = [row['data-rowkey'].split(':')[1] for row in rows if 'data-rowkey' in row.attrs]

            logger.info(f"Successfully extracted {len(tickers)} tickers")
            return tickers

        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:  # Retry if there are attempts left
                sleep_time = BACKOFF_FACTOR * (2 ** attempt)
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logger.error("All retries failed. Exiting.")
                raise
