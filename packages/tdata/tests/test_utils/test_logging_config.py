"""
Tests for the logging_config module.
"""

import pytest
import logging
import os
import tempfile
from packages.tdata.utils.logging_config import setup_logging

def test_setup_logging_console_only():
    """Test that setup_logging correctly configures console logging."""
    # Get the root logger
    logger = logging.getLogger()
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Configure logging with console only
    setup_logging(log_level=logging.INFO, log_to_console=True)
    
    # Verify that the logger has a console handler
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    
    # Verify the log level
    assert logger.level == logging.INFO
    assert logging.getLogger('packages.tdata').level == logging.INFO

def test_setup_logging_file_only():
    """Test that setup_logging correctly configures file logging."""
    # Get the root logger
    logger = logging.getLogger()
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create a temporary log file
    with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as tmp:
        log_file = tmp.name
    
    try:
        # Configure logging with file only
        setup_logging(log_level=logging.DEBUG, log_file=log_file, log_to_console=False)
        
        # Verify that the logger has a file handler
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.FileHandler)
        
        # Verify the log level
        assert logger.level == logging.DEBUG
        assert logging.getLogger('packages.tdata').level == logging.DEBUG
        
        # Write a log message
        test_logger = logging.getLogger('packages.tdata.test')
        test_logger.debug('Test debug message')
        test_logger.info('Test info message')
        
        # Force handlers to flush
        for handler in logger.handlers:
            handler.flush()
        
        # Verify that the log messages are written to the file
        with open(log_file, 'r') as f:
            log_content = f.read()
            assert 'Test debug message' in log_content
            assert 'Test info message' in log_content
    
    finally:
        # Clean up
        if os.path.exists(log_file):
            os.unlink(log_file)

def test_setup_logging_both():
    """Test that setup_logging correctly configures both console and file logging."""
    # Get the root logger
    logger = logging.getLogger()
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create a temporary log file
    with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as tmp:
        log_file = tmp.name
    
    try:
        # Configure logging with both console and file
        setup_logging(log_level=logging.WARNING, log_file=log_file, log_to_console=True)
        
        # Verify that the logger has both handlers
        assert len(logger.handlers) == 2
        
        # Check that we have one StreamHandler and one FileHandler
        handler_types = [type(h) for h in logger.handlers]
        assert logging.StreamHandler in handler_types
        assert logging.FileHandler in handler_types
        
        # Verify the log level
        assert logger.level == logging.WARNING
        assert logging.getLogger('packages.tdata').level == logging.WARNING
    
    finally:
        # Clean up
        if os.path.exists(log_file):
            os.unlink(log_file)

def test_setup_logging_custom_level():
    """Test that setup_logging correctly configures different log levels."""
    # Get the root logger
    logger = logging.getLogger()
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Configure logging with a custom level
    setup_logging(log_level=logging.ERROR)
    
    # Verify the log level
    assert logger.level == logging.ERROR
    assert logging.getLogger('packages.tdata').level == logging.ERROR 