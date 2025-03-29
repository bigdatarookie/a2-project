"""
Tests for the indicators module.
"""

import pytest
import pandas as pd
import numpy as np
from packages.tdata.processing.indicators import calculate_indicators

def test_calculate_indicators(sample_ohlcv_data):
    """Test that calculate_indicators correctly calculates all indicators."""
    # Call the function with sample data
    result = calculate_indicators(sample_ohlcv_data)

    # Verify that all expected indicators are present
    expected_indicators = [
        'rsi', 'macd_line', 'macd_signal', 'macd_histogram',
        'sma_12', 'sma_26', 'sma_50', 'sma_100', 'atr',
        'bb_lower', 'bb_middle', 'bb_upper', 'vwap',
        'adx', 'adx_positive', 'adx_negative', 'stoch_k', 'stoch_d', 'obv'
    ]

    for indicator in expected_indicators:
        assert indicator in result.columns

    # Verify that the data types are correct
    for indicator in expected_indicators:
        assert pd.api.types.is_numeric_dtype(result[indicator])

    # Verify that 'date' is included in the result
    assert 'date' in result.columns

    # Verify specific indicators

    # RSI should be between 0 and 100
    assert result['rsi'].min() >= 0
    assert result['rsi'].max() <= 100

    # Check SMA calculations
    for i in range(20, len(result)):
        expected_sma12 = sample_ohlcv_data['Close'].iloc[i-12:i].mean()
        assert abs(result['sma_12'].iloc[i] - expected_sma12) < 1e-6

    # Bollinger Bands: Middle band should be SMA(20), upper and lower should be 2 std dev away
    for i in range(20, len(result)):
        window = sample_ohlcv_data['Close'].iloc[i-20:i]
        expected_middle = window.mean()
        expected_std = window.std()

        assert abs(result['bb_middle'].iloc[i] - expected_middle) < 1e-6
        assert abs(result['bb_upper'].iloc[i] - (expected_middle + 2 * expected_std)) < 1e-6
        assert abs(result['bb_lower'].iloc[i] - (expected_middle - 2 * expected_std)) < 1e-6

def test_calculate_indicators_missing_columns():
    """Test that calculate_indicators raises ValueError for missing columns."""
    # Create a DataFrame missing required columns
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=100),
        'Close': np.random.random(100) * 100,  # Missing High, Low, Volume
    })

    # Should raise ValueError because required columns are missing
    with pytest.raises(ValueError):
        calculate_indicators(df)

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
