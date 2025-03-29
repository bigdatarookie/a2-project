"""
Logging configuration for the Tdata package.
"""

import os
import logging
import sys
from typing import Optional

def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_to_console: bool = True
) -> None:
    """
    Set up logging configuration for the tdata package.

    Args:
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Path to log file (if None, no file logging will be done)
        log_to_console: Whether to log to console
    """
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Add console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if log_file is specified
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Set tdata package logger level
    logging.getLogger('packages.tdata').setLevel(log_level)
    logging.getLogger('tinkoff.invest.logging').setLevel(logging.WARNING)
