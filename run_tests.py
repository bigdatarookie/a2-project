#!/usr/bin/env python
"""
Test runner script for the tdata package.
This script is executed when the container starts.
"""

import os
import sys
import pytest
import logging
import subprocess
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

    # Handle pytest exit codes:
    # 0: All tests passed
    # 1: Some tests failed
    # 2: Test execution was interrupted
    # 3: Internal error
    # 4: pytest command line usage error
    # 5: No tests collected
    if exit_code == 0:
        print("\n" + "=" * 80)
        print("All tests passed!")
        print("=" * 80)
        return 0
    else:
        error_messages = {
            1: "Some tests failed!",
            2: "Test execution was interrupted!",
            3: "Internal error in pytest!",
            4: "Pytest command line usage error!",
            5: "No tests were collected!"
        }
        message = error_messages.get(exit_code, f"Unknown error (code {exit_code})!")

        print("\n" + "=" * 80)
        print(f"{message} Stopping container.")
        print("=" * 80)

        # Force stop the container using subprocess
        try:
            # Get container ID using Docker command
            container_id = os.environ.get('HOSTNAME')
            if container_id:
                print(f"Forcing container to stop...")
                # Exit with non-zero code to ensure the container doesn't restart
                os._exit(1)
        except Exception as e:
            print(f"Error stopping container: {e}")

        # Exit with the same code that pytest returned if we're still running
        sys.exit(exit_code)

if __name__ == "__main__":
    sys.exit(main())
