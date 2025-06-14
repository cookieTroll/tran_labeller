import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal
from datetime import datetime

from src.data_transformer import parse_dates  # adjust import path as needed


@pytest.fixture
def sample_formats():
    return [
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y",
        "%d.%m.%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d"
    ]


def test_basic_date_parsing(sample_formats):
    """Test parsing of dates in the primary format"""
    input_dates = pd.Series([
        "01/12/2023 15:30:00",
        "02/12/2023 09:45:00"
    ])
    expected = pd.Series([
        datetime(2023, 12, 1, 15, 30),
        datetime(2023, 12, 2, 9, 45)
    ])

    result = parse_dates(input_dates, formats=sample_formats)
    assert_series_equal(result, expected)


def test_multiple_formats(sample_formats):
    """Test parsing dates in different formats"""
    input_dates = pd.Series([
        "01/12/2023 15:30:00",  # first format
        "02/12/2023",  # second format
        "03.12.2023",  # third format
        "2023-12-04"  # fifth format
    ])

    expected = pd.Series([
        datetime(2023, 12, 1, 15, 30),
        datetime(2023, 12, 2),
        datetime(2023, 12, 3),
        datetime(2023, 12, 4)
    ])

    with pytest.raises(ValueError):
        parse_dates(input_dates, sample_formats, errors='raise')


def test_whitespace_handling(sample_formats):
    """Test handling of whitespace in date strings"""
    input_dates = pd.Series([
        "  01/12/2023 15:30:00  ",
    ])

    expected = pd.Series([
        datetime(2023, 12, 1, 15, 30),

    ])

    result = parse_dates(input_dates, formats=sample_formats)
    assert_series_equal(result, expected)


def test_error_handling():
    """Test error handling for invalid dates"""
    input_dates = pd.Series(["invalid_date", "01/13/2023"])  # invalid month
    formats = ["%d/%m/%Y"]

    # Test 'raise' mode
    with pytest.raises(ValueError):
        parse_dates(input_dates, formats, errors='raise')

    # Test 'coerce' mode
    result = parse_dates(input_dates, formats, errors='coerce')
    assert result[0] is pd.NaT
    assert result[1] == datetime(2023, 1, 13)


def test_empty_series(sample_formats):
    """Test handling of empty series"""
    input_dates = pd.Series([], dtype=str)
    result = parse_dates(input_dates, formats=sample_formats)
    assert len(result) == 0


def test_null_values(sample_formats):
    """Test handling of null values"""
    input_dates = pd.Series([
        "01/12/2023",
        None,
        np.nan,
        "02/12/2023"
    ])

    expected = pd.Series([
        datetime(2023, 12, 1),
        pd.NaT,
        pd.NaT,
        datetime(2023, 12, 2)
    ])

    result = parse_dates(input_dates, formats=sample_formats)
    assert_series_equal(result, expected)


@pytest.mark.parametrize("errors", ['raise', 'coerce', 'ignore'])
def test_errors_parameter(errors):
    """Test different error handling modes"""
    input_dates = pd.Series(["01/12/2023", "invalid"])
    formats = ["%d/%m/%Y"]

    if errors == 'raise':
        with pytest.raises(ValueError):
            parse_dates(input_dates, formats, errors=errors)
    else:
        result = parse_dates(input_dates, formats, errors=errors)
        if errors == 'coerce':
            assert pd.isna(result[1])
        elif errors == 'ignore':
            assert result[1] == "invalid"


def test_empty_formats_list():
    """Test behavior when formats list is empty"""
    input_dates = pd.Series(["01/12/2023"])

    with pytest.raises(ValueError):
        parse_dates(input_dates, formats=[])


def test_performance_order(sample_formats):
    """Test that formats are tried in order"""
    input_dates = pd.Series(["01/12/2023 15:30:00"])

    # Reverse the formats list to make the matching format last
    reversed_formats = sample_formats[::-1]

    # Both should give same result, but reversed should be slower
    result1 = parse_dates(input_dates, formats=sample_formats)
    result2 = parse_dates(input_dates, formats=reversed_formats)

    assert_series_equal(result1, result2)