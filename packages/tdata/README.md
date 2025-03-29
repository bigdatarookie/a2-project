# Tdata

Financial Market Data Collection and Processing for Trading Applications

## Overview

Tdata is a Python package for gathering financial market data from the Tinkoff Investments API and processing it with technical indicators for time series analysis, machine learning, and trading strategies.

## Features

- Scrape ticker symbols from TradingView
- Fetch OHLCV (Open, High, Low, Close, Volume) data from Tinkoff API
- Calculate technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Process data files with indicators
- Command-line interface for data fetching and processing

## Installation

```bash
# Install from source
pip install -e .
```

## Usage

### Python API

```python
from tdata.ingestion import get_rts_data
from tdata.processing import process_file_with_indicators
from tdata.utils import setup_logging
import logging

# Setup logging
setup_logging(log_level=logging.INFO)

# Fetch market data
data = get_rts_data(
    start_date='2023-01-01',
    end_date='2023-01-31',
    interval='1day',
    output_file='rts_daily_jan'
)

# Process data with indicators
processed_data = process_file_with_indicators(
    input_file='rts_daily_jan.csv',
    output_file='rts_daily_jan_indicators'
)
```

### Command-line Interface

```bash
# Fetch data
tdata fetch --start-date 2023-01-01 --end-date 2023-01-31 --interval 1day --output rts_daily_jan

# Process data
tdata process --input rts_daily_jan.csv --output rts_daily_jan_indicators
```

## Requirements

- Python 3.8+
- Tinkoff Investments API Token

## Configuration

Set the Tinkoff API token as an environment variable or in a .env file:

```
TINKOFF_TOKEN=your_token_here
```

## Technical Indicators

The package calculates the following technical indicators:

- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Simple Moving Averages (12, 26, 50, 100 periods)
- Average True Range (ATR)
- Bollinger Bands
- Volume Weighted Average Price (VWAP)
- Average Directional Index (ADX)
- Stochastic Oscillator
- On-Balance Volume (OBV)

## License

[MIT License](LICENSE) 