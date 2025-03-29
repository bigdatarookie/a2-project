"""
Pytest fixtures for testing the tdata package.
"""

import pytest
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

@pytest.fixture
def sample_ohlcv_data():
    """
    Create a sample OHLCV DataFrame for testing.
    
    Returns:
        DataFrame with sample OHLCV data for a single ticker
    """
    # Create a date range for the past 100 days
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=100)
    dates = [start_date + timedelta(days=i) for i in range(100)]
    
    # Create sample price data with some patterns
    base_price = 100
    trend = np.linspace(0, 20, 100)  # Upward trend
    noise = np.random.normal(0, 2, 100)  # Random noise
    cycle = 10 * np.sin(np.linspace(0, 4*np.pi, 100))  # Cyclical pattern
    
    close_prices = base_price + trend + noise + cycle
    open_prices = close_prices - np.random.normal(0, 1, 100)
    high_prices = np.maximum(close_prices, open_prices) + abs(np.random.normal(0, 1, 100))
    low_prices = np.minimum(close_prices, open_prices) - abs(np.random.normal(0, 1, 100))
    volumes = np.random.randint(1000, 10000, 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes,
        'tic': ['AAPL'] * 100
    })
    
    return df

@pytest.fixture
def sample_multi_ticker_data():
    """
    Create a sample OHLCV DataFrame with multiple tickers.
    
    Returns:
        DataFrame with sample OHLCV data for multiple tickers
    """
    tickers = ['AAPL', 'MSFT', 'GOOG']
    dfs = []
    
    for ticker in tickers:
        # Create a date range for the past 100 days
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=100)
        dates = [start_date + timedelta(days=i) for i in range(100)]
        
        # Create sample price data with some patterns
        base_price = 100 + (50 if ticker == 'AAPL' else 0) + (100 if ticker == 'GOOG' else 0)
        trend = np.linspace(0, 20, 100)  # Upward trend
        noise = np.random.normal(0, 2, 100)  # Random noise
        cycle = 10 * np.sin(np.linspace(0, 4*np.pi, 100))  # Cyclical pattern
        
        close_prices = base_price + trend + noise + cycle
        open_prices = close_prices - np.random.normal(0, 1, 100)
        high_prices = np.maximum(close_prices, open_prices) + abs(np.random.normal(0, 1, 100))
        low_prices = np.minimum(close_prices, open_prices) - abs(np.random.normal(0, 1, 100))
        volumes = np.random.randint(1000, 10000, 100)
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volumes,
            'tic': [ticker] * 100
        })
        
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

@pytest.fixture
def mock_tinkoff_client():
    """
    Create a mock for the Tinkoff client with sample responses.
    
    Returns:
        Mock client that can be used in tests
    """
    # Create a MagicMock for the Client
    with patch('packages.tdata.ingestion.tinkoff_api.Client') as mock_client:
        # Set up the client instance
        instance = mock_client.return_value.__enter__.return_value
        
        # Mock the instruments service
        instruments = MagicMock()
        instance.instruments = instruments
        
        # Mock the market_data service
        market_data = MagicMock()
        instance.market_data = market_data
        
        # Sample Candle
        class MockCandle:
            def __init__(self, time, open_val, high_val, low_val, close_val, volume):
                self.time = time
                
                # Mock quotation format
                self.open = MagicMock()
                self.open.units = int(open_val)
                self.open.nano = int((open_val - int(open_val)) * 1e9)
                
                self.high = MagicMock()
                self.high.units = int(high_val)
                self.high.nano = int((high_val - int(high_val)) * 1e9)
                
                self.low = MagicMock()
                self.low.units = int(low_val)
                self.low.nano = int((low_val - int(low_val)) * 1e9)
                
                self.close = MagicMock()
                self.close.units = int(close_val)
                self.close.nano = int((close_val - int(close_val)) * 1e9)
                
                self.volume = volume
        
        # Generate sample candles
        candles = []
        now = datetime.now()
        for i in range(10):
            candle_time = now - timedelta(hours=i)
            candle = MockCandle(
                time=candle_time,
                open_val=100.5 + i,
                high_val=105.5 + i,
                low_val=95.5 + i,
                close_val=102.5 + i,
                volume=1000 * (i + 1)
            )
            candles.append(candle)
        
        # Mock the get_candles response
        candles_response = MagicMock()
        candles_response.candles = candles
        market_data.get_candles.return_value = candles_response
        
        # Sample instrument
        class MockInstrument:
            def __init__(self, name, ticker, figi, class_code, uid, min_price_increment, lot):
                self.name = name
                self.ticker = ticker
                self.figi = figi
                self.class_code = class_code
                self.uid = uid
                
                # Mock quotation format
                self.min_price_increment = MagicMock()
                self.min_price_increment.units = int(min_price_increment)
                self.min_price_increment.nano = int((min_price_increment - int(min_price_increment)) * 1e9)
                
                self.lot = lot
                
                # Other attributes used in get_figis
                self.trading_status = 1  # SECURITY_TRADING_STATUS_NORMAL
                self.api_trade_available_flag = True
                self.currency = "USD"
                self.exchange = "MOEX"
                self.buy_available_flag = True
                self.sell_available_flag = True
                self.short_enabled_flag = True
                self.klong = MagicMock()
                self.klong.units = 2
                self.klong.nano = 0
                self.kshort = MagicMock()
                self.kshort.units = 2
                self.kshort.nano = 0
        
        # Create sample instruments
        sample_shares = [
            MockInstrument("Apple Inc.", "AAPL", "BBG000B9XRY4", "TQBR", "apple_uid", 0.01, 1),
            MockInstrument("Microsoft Corporation", "MSFT", "BBG000BPH459", "TQBR", "msft_uid", 0.01, 1),
            MockInstrument("Lukoil", "LKOH", "BBG004731354", "TQBR", "lkoh_uid", 0.5, 1)
        ]
        
        # Mock the instruments methods
        shares_response = MagicMock()
        shares_response.instruments = sample_shares
        instruments.shares.return_value = shares_response
        
        # Mock other instrument types to return empty lists
        for method in ["bonds", "etfs", "currencies", "futures"]:
            method_response = MagicMock()
            method_response.instruments = []
            getattr(instruments, method).return_value = method_response
        
        yield mock_client 