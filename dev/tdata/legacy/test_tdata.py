import pytest
import pandas as pd

from datetime import timedelta
from decimal import Decimal
from unittest.mock import patch, Mock
from tdata import get_figis, extract_tickers, fetch_ohlcv_data

@patch('tdata.requests.get')
def test_extract_tickers_success(mock_get):
    # Mock HTML content returned by the GET request
    mock_html = '''
    <html>
        <body>
            <table>
                <tr class="row-RdUXZpkv listRow" data-rowkey="MOEX:GAZP"></tr>
                <tr class="row-RdUXZpkv listRow" data-rowkey="MOEX:SBER"></tr>
                <tr class="row-RdUXZpkv listRow" data-rowkey="MOEX:LKOH"></tr>
            </table>
        </body>
    </html>
    '''

    # Configure the mock response object
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = mock_html
    mock_get.return_value = mock_response

    # Call the function
    tickers = extract_tickers()

    # Assert the expected results
    assert tickers == ['GAZP', 'SBER', 'LKOH']

@patch('tdata.requests.get')
def test_extract_tickers_empty(mock_get):
    # Mock HTML content with no rows
    mock_html = '''
    <html>
        <body>
            <table></table>
        </body>
    </html>
    '''

    # Configure the mock response object
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = mock_html
    mock_get.return_value = mock_response

    # Call the function
    tickers = extract_tickers()

    # Assert the expected results
    assert tickers == []

@patch('tdata.requests.get')
def test_extract_tickers_invalid_html(mock_get):
    # Mock invalid HTML content
    mock_html = "<html><body><p>Invalid content</p></body></html>"

    # Configure the mock response object
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = mock_html
    mock_get.return_value = mock_response

    # Call the function
    tickers = extract_tickers()

    # Assert the expected results
    assert tickers == []

@patch('tdata.requests.get')
def test_extract_tickers_request_failure(mock_get):
    # Configure the mock to raise an HTTP error
    mock_get.side_effect = Exception("Request failed")

    # Assert that an exception is raised
    with pytest.raises(Exception):
        extract_tickers()
@patch('tdata.Client')
def test_get_figis_success(mock_client):
    # Mock the instruments data returned by the Client
    mock_instruments = Mock()
    mock_instruments.shares.return_value.instruments = [
        Mock(
            name="Share1",
            ticker="TICK1",
            class_code="CLASS1",
            figi="FIGI1",
            uid="UID1",
            min_price_increment=Mock(nano=500),
            lot=10,
            trading_status=0,
            api_trade_available_flag=True,
            currency="USD",
            exchange="EXCHANGE1",
            buy_available_flag=True,
            sell_available_flag=True,
            short_enabled_flag=True,
            klong=Mock(nano=1000),
            kshort=Mock(nano=1000),
        )
    ]
    mock_instruments.bonds.return_value.instruments = []
    mock_instruments.etfs.return_value.instruments = []
    mock_instruments.currencies.return_value.instruments = []
    mock_instruments.futures.return_value.instruments = []

    mock_client.return_value.__enter__.return_value.instruments = mock_instruments

    input_tickers = ["TICK1"]
    TOKEN = "test_token"

    with patch('tdata.quotation_to_decimal', side_effect=lambda x: Decimal(f"0.{x.nano}")):
        result = get_figis(input_tickers, TOKEN)

    # Assert the expected result
    assert result == [("TICK1", "FIGI1")]

@patch('tdata.Client')
def test_get_figis_ticker_not_found(mock_client):
    # Mock the instruments data returned by the Client
    mock_instruments = Mock()
    mock_instruments.shares.return_value.instruments = []
    mock_instruments.bonds.return_value.instruments = []
    mock_instruments.etfs.return_value.instruments = []
    mock_instruments.currencies.return_value.instruments = []
    mock_instruments.futures.return_value.instruments = []

    mock_client.return_value.__enter__.return_value.instruments = mock_instruments

    input_tickers = ["UNKNOWN_TICKER"]
    TOKEN = "test_token"

    with patch('tdata.quotation_to_decimal', side_effect=lambda x: Decimal(f"0.{x.nano}")):
        with pytest.raises(Exception):
            get_figis(input_tickers, TOKEN)

@patch('tdata.Client')
def test_get_figis_empty_input(mock_client):
    # Mock the instruments data returned by the Client
    mock_instruments = Mock()
    mock_instruments.shares.return_value.instruments = []
    mock_instruments.bonds.return_value.instruments = []
    mock_instruments.etfs.return_value.instruments = []
    mock_instruments.currencies.return_value.instruments = []
    mock_instruments.futures.return_value.instruments = []

    mock_client.return_value.__enter__.return_value.instruments = mock_instruments

    input_tickers = []
    TOKEN = "test_token"

    with patch('tdata.quotation_to_decimal', side_effect=lambda x: Decimal(f"0.{x.nano}")):
        result = get_figis(input_tickers, TOKEN)

    # Assert the expected result for empty input
    assert result == []

@patch('tdata.Client')
def test_fetch_ohlcv_data_success(mock_client):
    # Mock candle data
    mock_candles = Mock()
    mock_candles.candles = [
        Mock(
            time=pd.Timestamp("2023-12-01T12:00:00Z"),
            open=Mock(units=100, nano=500000000),
            high=Mock(units=110, nano=0),
            low=Mock(units=95, nano=0),
            close=Mock(units=105, nano=500000000),
            volume=1000,
        )
    ]

    # Mock the market data service
    mock_market_data = Mock()
    mock_market_data.get_candles.return_value = mock_candles

    # Mock the Client object
    mock_client_instance = Mock()
    mock_client_instance.market_data = mock_market_data
    mock_client.return_value.__enter__.return_value = mock_client_instance

    # Input data
    ticker_figis = [("TICK1", "FIGI1")]
    start_date = pd.Timestamp("2023-12-01T00:00:00Z")
    end_date = pd.Timestamp("2023-12-02T00:00:00Z")
    interval = "CANDLE_INTERVAL_15_MIN"
    TOKEN = "test_token"

    # Call the function
    result = fetch_ohlcv_data(ticker_figis, start_date, end_date, interval, TOKEN)

    # Expected DataFrame
    expected_data = [
        {
            "figi": "FIGI1",
            "date": pd.Timestamp("2023-12-01T12:00:00Z"),
            "open": 100.5,
            "high": 110.0,
            "low": 95.0,
            "close": 105.5,
            "volume": 1000,
            "tic": "TICK1",
            "day": 4,  # Assuming 2023-12-01 is a Friday
        }
    ]
    expected_df = pd.DataFrame(expected_data)

    # Convert 'date' column to datetime in both DataFrames
    result["date"] = pd.to_datetime(result["date"])
    expected_df["date"] = pd.to_datetime(expected_df["date"])

    # Assert the result
    pd.testing.assert_frame_equal(result, expected_df)


@patch('tdata.Client')
def test_fetch_ohlcv_data_rate_limit(mock_client):
    # Mock the market data service to raise a rate limit exception
    mock_market_data = Mock()
    mock_market_data.get_candles.side_effect = [
        Exception("Failed to fetch data for FIGI BBG000R607Y3 (Ticker: PLZL): (<StatusCode.RESOURCE_EXHAUSTED: (8, 'resource exhausted')>, '', Metadata(tracking_id=None, ratelimit_limit='600, 600;w=60', ratelimit_remaining=0, ratelimit_reset=3, message=None))"),
        Mock(
            candles=[
                Mock(
                    time=pd.Timestamp("2023-12-01T12:00:00Z"),
                    open=Mock(units=100, nano=0),
                    high=Mock(units=110, nano=0),
                    low=Mock(units=95, nano=0),
                    close=Mock(units=105, nano=0),
                    volume=1000,
                )
            ]
        ),
    ]

    # Mock the Client object
    mock_client_instance = Mock()
    mock_client_instance.market_data = mock_market_data
    mock_client.return_value.__enter__.return_value = mock_client_instance

    # Input data
    ticker_figis = [("TICK1", "FIGI1")]
    start_date = pd.Timestamp("2023-12-01T00:00:00Z")
    end_date = pd.Timestamp("2023-12-02T00:00:00Z")
    interval = "CANDLE_INTERVAL_15_MIN"
    TOKEN = "test_token"

    # Call the function
    result = fetch_ohlcv_data(ticker_figis, start_date, end_date, interval, TOKEN)

    # Assert that one retry was made due to rate limit
    assert mock_market_data.get_candles.call_count == 2

    # Assert the data contains rows from the second attempt
    assert len(result) == 1
