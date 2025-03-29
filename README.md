# A2 Project

This project provides a Docker-based environment for working with Jupyter notebooks and data analysis.

## Prerequisites

- Docker
- Docker Compose

### Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/a2-project.git
   cd a2-project
   ```

2. Create a `.env` file with your configuration:
   ```
   JUPYTER_TOKEN=your_jupyter_token
   TINKOFF_TOKEN=your_tinkoff_api_token
   ```

3. Build and start the container:
   ```bash
   docker-compose up --build
   ```

4. Access Jupyter Lab at `http://localhost:8888` using your Jupyter token.

### Automated Testing

When the container starts, it automatically runs the test suite for the `tdata` package. Test results are displayed in the terminal. If all tests pass, Jupyter Lab will start.

- **Test Structure**: Tests are organized in the `tdata/tests/` directory, mirroring the package structure.
- **Test Coverage**: The test runner calculates and displays code coverage statistics.

## Manual Usage

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
python -m tdata fetch --start-date 2023-01-01 --end-date 2023-01-31 --interval 1day --output rts_daily_jan

# Process data
python -m tdata process --input rts_daily_jan.csv --output rts_daily_jan_indicators
```

## Adding New Tests

To add new tests for the `tdata` package:

1. Create test files in the appropriate subdirectory of `tdata/tests/`.
2. Run the tests locally:
   ```bash
   docker-compose up
   ```
   or
   ```bash
   python run_tests.py
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

## Project Structure

```
.
├── docker/
│   └── Dockerfile
├── data/              # Data directory (mounted from a1-data volume)
├── requirements.txt   # Python dependencies
├── docker-compose.yml # Docker Compose configuration
└── README.md         # This file
```

## Development

- The `a1-data` volume is mounted to `/app/data` in the container
- All Python dependencies are listed in `requirements.txt`

## Git Workflow

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

3. Push your changes:
   ```bash
   git push origin feature/your-feature-name
   ```

## Docker Commands

- Start the container:
  ```bash
  docker-compose up
  ```

- Stop the container:
  ```bash
  docker-compose down
  ```

- Rebuild the container:
  ```bash
  docker-compose up --build
  ```

- View logs:
  ```bash
  docker-compose logs -f
  ```

## Data Management

The project uses a Docker volume named `a1-data` to persist data. This volume is mounted to `/app/data` inside the container.

To manage the volume:
- Create: `docker volume create a1-data`
- Remove: `docker volume rm a1-data`
- List: `docker volume ls` 