"""
Command-line interface for the Tdata package.
"""

import argparse
import logging
import sys
from datetime import datetime
from typing import Optional, Dict, Any, List

from tdata.ingestion import get_rts_data
from tdata.processing import process_file_with_indicators
from tdata.utils import setup_logging

def fetch_data(args: argparse.Namespace) -> None:
    """Handle the 'fetch' command."""
    # Setup logging
    setup_logging(
        log_level=logging.DEBUG if args.verbose else logging.INFO,
        log_file=args.log_file
    )

    # Parse dates
    try:
        start_date = datetime.fromisoformat(args.start_date)
        end_date = datetime.fromisoformat(args.end_date)
    except ValueError as e:
        print(f"Error parsing dates: {str(e)}")
        print("Dates should be in ISO format: YYYY-MM-DD")
        sys.exit(1)

    # Fetch data
    try:
        get_rts_data(
            start_date=start_date,
            end_date=end_date,
            interval=args.interval,
            output_file=args.output,
            token=args.token
        )
        print(f"Data fetched successfully and saved to {args.output}.csv")
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        sys.exit(1)

def process_data(args: argparse.Namespace) -> None:
    """Handle the 'process' command."""
    # Setup logging
    setup_logging(
        log_level=logging.DEBUG if args.verbose else logging.INFO,
        log_file=args.log_file
    )

    # Process data
    try:
        process_file_with_indicators(
            input_file=args.input,
            output_file=args.output,
            drop_na=not args.keep_na,
            optimize_types=not args.no_optimize
        )
        print(f"Data processed successfully and saved to {args.output}.csv")
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        sys.exit(1)

def main() -> None:
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Tdata - Financial Market Data Collection and Processing"
    )

    # Common arguments
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path"
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="Command to execute",
        required=True
    )

    # Fetch command
    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Fetch market data from Tinkoff API"
    )
    fetch_parser.add_argument(
        "--start-date", "-s",
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    fetch_parser.add_argument(
        "--end-date", "-e",
        required=True,
        help="End date (YYYY-MM-DD)"
    )
    fetch_parser.add_argument(
        "--interval", "-i",
        default="1day",
        choices=["1min", "5min", "15min", "1hour", "2hour", "4hour", "1day", "1week"],
        help="Candle interval"
    )
    fetch_parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output file name (without extension)"
    )
    fetch_parser.add_argument(
        "--token", "-t",
        help="Tinkoff API token (overrides environment variable)"
    )
    fetch_parser.set_defaults(func=fetch_data)

    # Process command
    process_parser = subparsers.add_parser(
        "process",
        help="Process data and calculate technical indicators"
    )
    process_parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input CSV file path"
    )
    process_parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output file name (without extension)"
    )
    process_parser.add_argument(
        "--keep-na",
        action="store_true",
        help="Keep rows with NaN values"
    )
    process_parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Don't optimize data types"
    )
    process_parser.set_defaults(func=process_data)

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    args.func(args)

if __name__ == "__main__":
    main()
