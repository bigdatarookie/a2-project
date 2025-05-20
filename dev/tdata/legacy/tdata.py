import requests
import pandas as pd
import logging
import time

from datetime import timedelta
from bs4 import BeautifulSoup
from tinkoff.invest.sandbox.client import SandboxClient as Client
from tinkoff.invest import SecurityTradingStatus
from tinkoff.invest.services import InstrumentsService
from tinkoff.invest.utils import quotation_to_decimal

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def extract_tickers(url="https://www.tradingview.com/symbols/RUS-RTSI/components/"):
    retries = 3  # Number of retries
    backoff_factor = 2  # Exponential backoff factor (seconds)

    for attempt in range(retries):
        try:
            # Send a GET request to the webpage
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Check for request errors

            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all rows matching the pattern
            rows = soup.find_all('tr', class_="row-RdUXZpkv listRow")

            # Extract tickers from the 'data-rowkey' attribute
            tickers = [row['data-rowkey'].split(':')[1] for row in rows if 'data-rowkey' in row.attrs]

            return tickers

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:  # Retry if there are attempts left
                sleep_time = backoff_factor * (2 ** attempt)
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("All retries failed. Exiting.")
                raise

def get_figis(input_tickers, TOKEN):
    """Gets list of ticker-figi indexes pairs for given list of tickers"""
    result = []
    
    with Client(TOKEN) as client:
        instruments: InstrumentsService = client.instruments
        tickers = []
        for method in ["shares", "bonds", "etfs", "currencies", "futures"]:
            for item in getattr(instruments, method)().instruments:
                tickers.append(
                    {
                        "name": item.name,
                        "ticker": item.ticker,
                        "class_code": item.class_code,
                        "figi": item.figi,
                        "uid": item.uid,
                        "type": method,
                        "min_price_increment": quotation_to_decimal(
                            item.min_price_increment
                        ),
                        "scale": 9 - len(str(item.min_price_increment.nano)) + 1,
                        "lot": item.lot,
                        "trading_status": str(
                            SecurityTradingStatus(item.trading_status).name
                        ),
                        "api_trade_available_flag": item.api_trade_available_flag,
                        "currency": item.currency,
                        "exchange": item.exchange,
                        "buy_available_flag": item.buy_available_flag,
                        "sell_available_flag": item.sell_available_flag,
                        "short_enabled_flag": item.short_enabled_flag,
                        "klong": quotation_to_decimal(item.klong),
                        "kshort": quotation_to_decimal(item.kshort),
                    }
                )

        tickers_df = pd.DataFrame(tickers)

        for ticker in input_tickers :
            ticker_df = tickers_df[tickers_df["ticker"] == ticker]
            
            if ticker_df.empty:
                logger.error("There is no such ticker: %s", ticker)
                return
    
            figi = ticker_df["figi"].iloc[0]
            
            result.append((ticker, figi))
    return result

def fetch_ohlcv_data(ticker_figis, start_date, end_date, interval, TOKEN):
    """
    Fetch OHLCV data for multiple FIGIs with progress logging.
    """
    # List to store rows of data
    data_rows = []

    # Define the maximum period per request for each interval
    max_periods = {
        "CANDLE_INTERVAL_1_MIN": timedelta(days=1),
        "CANDLE_INTERVAL_5_MIN": timedelta(days=7),
        "CANDLE_INTERVAL_15_MIN": timedelta(days=21),
        "CANDLE_INTERVAL_HOUR": timedelta(days=90),
        "CANDLE_INTERVAL_2_HOUR": timedelta(days=90),
        "CANDLE_INTERVAL_4_HOUR": timedelta(days=90),
        "CANDLE_INTERVAL_DAY": timedelta(days=365),
        "CANDLE_INTERVAL_WEEK": timedelta(days=365 * 5),
    }

    # Get the maximum allowable period for the given interval
    max_duration = max_periods.get(interval, timedelta(days=1))  # Default to 1 day if interval is unknown

    total_figis = len(ticker_figis)  # Total number of FIGIs to process
    total_data_points = 0  # Total data points fetched

    with Client(TOKEN) as client:
        for figi_index, (ticker, figi) in enumerate(ticker_figis, start=1):  # Enumerate to track progress
            print(f"Processing {ticker} ({figi}) [{figi_index}/{total_figis}]...")
            current_start = start_date
            figi_data_points = 0  # Data points fetched for this FIGI

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

                    # Append each candle as a row with ticker and figi
                    for candle in candles.candles:
                        row = {
                            "figi": figi,
                            "date": candle.time,
                            "open": candle.open.units + candle.open.nano / 1e9,
                            "high": candle.high.units + candle.high.nano / 1e9,
                            "low": candle.low.units + candle.low.nano / 1e9,
                            "close": candle.close.units + candle.close.nano / 1e9,
                            "volume": candle.volume,
                            "tic": ticker,
                            "day": candle.time.weekday(),
                        }
                        data_rows.append(row)
                        figi_data_points += 1

                    # Move to the next time chunk only if the current batch is successful
                    current_start = current_end

                except Exception as e:
                    print(f"Failed to fetch data for FIGI {figi} (Ticker: {ticker}): {e}")

                    # Check if the error is RESOURCE_EXHAUSTED
                    if "RESOURCE_EXHAUSTED" in str(e):
                        # Extract reset time from the metadata if available
                        metadata = getattr(e.args[-1], 'ratelimit_reset', None)
                        reset_time = int(metadata) if metadata else 60  # Default to 60 seconds if not available
                        print(f"Rate limit exceeded. Fetched {figi_data_points} rows so far for {ticker}.")
                        print(f"Sleeping for {reset_time} seconds...")
                        time.sleep(reset_time)
                    else:
                        # Log and continue for other errors
                        # Skip this batch as it's not recoverable
                        break

            # Log progress for the current FIGI
            print(f"Completed {ticker} ({figi}). Total rows fetched for this FIGI: {figi_data_points}.")
            total_data_points += figi_data_points

    # Convert the list of rows into a DataFrame
    df = pd.DataFrame(data_rows)

    # Final log output
    print(f"All rows fetched: {total_data_points} rows in total.")
    return df

def get_rts_data(start_date, end_date, interval, TOKEN):
    tickers = extract_tickers()
    ticker_figis = get_figis(tickers, TOKEN)
    result = fetch_ohlcv_data(ticker_figis, start_date, end_date, interval, TOKEN)
    return result