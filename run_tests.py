#!/usr/bin/env python
"""
Test runner script for the tdata package.
This script is executed when the container starts.
"""

import os
import sys
import pytest
import logging
from packages.tdata.utils import setup_logging

def main():
    """Run all tests for the tdata package."""
    # Setup logging
    setup_logging(log_level=logging.INFO)
    
    print("=" * 80)
    print("Running tests for tdata package")
    print("=" * 80)
    
    # Get the absolute path to the tdata package
    tdata_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'packages', 'tdata'))
    
    # Run tests
    exit_code = pytest.main([
        '-xvs',                      # Verbose output, exit on first failure
        '--no-header',               # No pytest header
        '--tb=native',               # Python traceback format
        f'{tdata_path}/tests/',      # Test directory
        '--cov=packages.tdata',      # Coverage report for tdata package
        '--cov-report=term',         # Output coverage report to terminal
    ])
    
    if exit_code == 0:
        print("\n" + "=" * 80)
        print("All tests passed!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("Some tests failed!")
        print("=" * 80)
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
