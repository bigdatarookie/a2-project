"""
Tdata ingestion package for fetching market data from Tinkoff API.
"""

from packages.tdata.ingestion.data_fetcher import get_rts_data
from packages.tdata.ingestion.ticker_scraper import extract_tickers
from packages.tdata.ingestion.tinkoff_api import get_figis, fetch_ohlcv_data

__all__ = ['get_rts_data', 'extract_tickers', 'get_figis', 'fetch_ohlcv_data']
