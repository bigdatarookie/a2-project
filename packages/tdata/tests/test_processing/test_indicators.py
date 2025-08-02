"""
Tests for the indicators module.
"""

import pytest
import pandas as pd
import numpy as np
from packages.tdata.processing.indicators import calculate_indicators

def test_calculate_indicators_unsorted_data(sample_ohlcv_data):
    """Test that calculate_indicators correctly handles unsorted data."""
    # Shuffle the sample data
    shuffled_df = sample_ohlcv_data.sample(frac=1).reset_index(drop=True)

    # Calculate indicators on shuffled data
    result = calculate_indicators(shuffled_df)

    # Calculate indicators on properly sorted data
    expected = calculate_indicators(sample_ohlcv_data)

    # Results should be identical after sorting by date
    result_sorted = result.sort_values('date').reset_index(drop=True)
    expected_sorted = expected.sort_values('date').reset_index(drop=True)

    # Compare the results (allowing for small numerical differences)
    pd.testing.assert_frame_equal(
        result_sorted,
        expected_sorted,
        check_exact=False,
        rtol=1e-5
    )
