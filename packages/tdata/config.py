"""
Configuration module for Tdata package.
Contains settings for API access, data storage, and processing parameters.
"""
import os
from dotenv import load_dotenv
from tinkoff.invest import CandleInterval

# Load environment variables
load_dotenv()

# Data Storage Configuration
DATA_DIR = os.getenv('DATA_DIR', 'data')

# Create cache directory if it doesn't exist
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# API Configuration
TINKOFF_TOKEN = os.getenv('TINKOFF_TOKEN')

# Data Storage Configuration
DATA_DIR = os.getenv('DATA_DIR', 'data')

# API Request Configuration
MAX_RETRIES = 3
BACKOFF_FACTOR = 2
REQUEST_TIMEOUT = 10  # seconds

# Rate Limiting
DEFAULT_RATE_LIMIT_PAUSE = 60  # seconds to wait when rate limit is hit

# Intervals mapping from string to Tinkoff API format
INTERVAL_MAPPING = {
    "1min": "1min",  # These are the friendly names used as keys in fetch_ohlcv_data
    "5min": "5min",
    "15min": "15min",
    "1hour": "1hour",
    "2hour": "2hour",
    "4hour": "4hour",
    "1day": "1day",
    "1week": "1week"
}

# If you still need the enum mapping, create a separate constant
INTERVAL_TO_ENUM = {
    "1min": CandleInterval.CANDLE_INTERVAL_1_MIN,
    "5min": CandleInterval.CANDLE_INTERVAL_5_MIN,
    "15min": CandleInterval.CANDLE_INTERVAL_15_MIN,
    "1hour": CandleInterval.CANDLE_INTERVAL_HOUR,
    "2hour": CandleInterval.CANDLE_INTERVAL_2_HOUR,
    "4hour": CandleInterval.CANDLE_INTERVAL_4_HOUR,
    "1day": CandleInterval.CANDLE_INTERVAL_DAY,
    "1week": CandleInterval.CANDLE_INTERVAL_WEEK
}

# Maximum time periods per request for each interval
# (to avoid exceeding API limits)
MAX_PERIODS = {
    'CANDLE_INTERVAL_1_MIN': 1,      # days
    'CANDLE_INTERVAL_5_MIN': 7,      # days
    'CANDLE_INTERVAL_15_MIN': 21,    # days
    'CANDLE_INTERVAL_HOUR': 90,      # days
    'CANDLE_INTERVAL_2_HOUR': 90,    # days
    'CANDLE_INTERVAL_4_HOUR': 90,    # days
    'CANDLE_INTERVAL_DAY': 365,      # days
    'CANDLE_INTERVAL_WEEK': 365 * 5, # days
}
