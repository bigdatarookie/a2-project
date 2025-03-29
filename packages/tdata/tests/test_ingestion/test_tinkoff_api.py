"""
Tests for the tinkoff_api module.
"""

import pytest
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
    """Test that fetch_ohlcv_data correctly fetches and processes OHLCV data."""
    # Prepare test data
    ticker_figis = [('AAPL', 'BBG000B9XRY4'), ('MSFT', 'BBG000BPH459')]
    start_date = datetime.now() - timedelta(days=1)
    end_date = datetime.now()
    interval = 'CANDLE_INTERVAL_HOUR'
    
    # Call the function
    result = fetch_ohlcv_data(ticker_figis, start_date, end_date, interval, token='test_token')
    
    # Verify the results
    assert isinstance(result, list)
    assert len(result) > 0  # Should have returned some data
    
    # Check the structure of the returned data
    for row in result:
        assert 'figi' in row
        assert 'date' in row
        assert 'open' in row
        assert 'high' in row
        assert 'low' in row
        assert 'close' in row
        assert 'volume' in row
        assert 'tic' in row
        assert 'day' in row
        
        # Verify the data types
        assert isinstance(row['figi'], str)
        assert isinstance(row['date'], datetime)
        assert isinstance(row['open'], float)
        assert isinstance(row['high'], float)
        assert isinstance(row['low'], float)
        assert isinstance(row['close'], float)
        assert isinstance(row['volume'], int)
        assert isinstance(row['tic'], str)
        assert isinstance(row['day'], int)
        
        # Verify the values are within expected ranges
        assert row['day'] >= 0 and row['day'] <= 6  # Day of week (0-6)
        assert row['open'] > 0
        assert row['high'] >= row['open']
        assert row['low'] <= row['open']
        assert row['volume'] >= 0

def test_fetch_ohlcv_data_empty_list(mock_tinkoff_client):
    """Test that fetch_ohlcv_data handles an empty list of ticker_figis."""
    # Call with empty list
    result = fetch_ohlcv_data([], datetime.now(), datetime.now(), 'CANDLE_INTERVAL_HOUR', token='test_token')
    
    # Should return an empty list, not raise an error
    assert isinstance(result, list)
    assert len(result) == 0

def test_fetch_ohlcv_data_invalid_token():
    """Test that fetch_ohlcv_data raises ValueError for an invalid token."""
    with pytest.raises(ValueError):
        fetch_ohlcv_data([('AAPL', 'BBG000B9XRY4')], datetime.now(), datetime.now(), 'CANDLE_INTERVAL_HOUR', token=None) 