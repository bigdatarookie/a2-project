"""
Tdata processing package for calculating technical indicators and processing financial data.
"""

from tdata.processing.indicators import calculate_indicators
from tdata.processing.data_processor import process_file_with_indicators

__all__ = ['calculate_indicators', 'process_file_with_indicators'] 