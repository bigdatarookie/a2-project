"""
Tests for the tinkoff_api module.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from packages.tdata.ingestion.tinkoff_api import get_figis, fetch_ohlcv_data

def test_get_figis(mock_tinkoff_client):
    """Test that get_figis correctly maps tickers to FIGIs."""
    # Call the function with test ticker symbols
    result = get_figis(['AAPL', 'MSFT', 'UNKNOWN'], token='test_token')

    # Verify the results
    assert len(result) == 2  # Only 2 of the 3 tickers should be found

    # Check that the tickers are correctly mapped to FIGIs
    tickers = [item[0] for item in result]
    figis = [item[1] for item in result]

    assert 'AAPL' in tickers
    assert 'MSFT' in tickers
    assert 'BBG000B9XRY4' in figis
    assert 'BBG000BPH459' in figis

    # Check that the mock client was called with the correct token
    mock_tinkoff_client.assert_called_once_with('test_token')

def test_fetch_ohlcv_data(mock_tinkoff_client):
    """Test that fetch_ohlcv_data fetches and processes data correctly."""
    # Setup test data
    ticker_figis = [('AAPL', 'BBG000B9XRY4'), ('MSFT', 'BBG000BPH459')]
    start_date = datetime.now() - timedelta(days=1)
    end_date = datetime.now()
    interval = "1hour"  # Use the friendly name

    # Call the function with the mock client
    result = fetch_ohlcv_data(ticker_figis, start_date, end_date, interval, token='test_token')

    # Print summary instead of entire result
    print(f"Result contains {len(result)} data points")

    # Check that result is not empty
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_fetch_ohlcv_data_empty_list():
    """Test that fetch_ohlcv_data handles empty ticker list properly."""
    # Setup empty test data
    ticker_figis = []
    start_date = datetime.now() - timedelta(days=1)
    end_date = datetime.now()
    interval = "1hour"  # Use the friendly name that matches your config

    # Call the function with empty data
    result = fetch_ohlcv_data(ticker_figis, start_date, end_date, interval, token='test_token')

    # Check that result is an empty dataframe
    assert isinstance(result, pd.DataFrame)
    assert result.empty

def test_fetch_ohlcv_data_invalid_token():
    """Test that fetch_ohlcv_data raises ValueError for an invalid token."""
    with pytest.raises(ValueError):
        fetch_ohlcv_data([('AAPL', 'BBG000B9XRY4')], datetime.now(), datetime.now(), 'CANDLE_INTERVAL_HOUR', token=None)
