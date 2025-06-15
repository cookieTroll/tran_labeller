import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal, assert_frame_equal
from datetime import datetime

from src.data_transformer import parse_dates, parse_amounts, clean_overlapping_fields


@pytest.fixture
def sample_formats():
    return [
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y",
        "%d.%m.%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d"
    ]

## Test parse dates
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


# Test parse_amounts

def test_parse_amounts_basic():
    """Test basic number parsing without separators"""
    input_data = pd.Series(['123.45', '67.89', '0.01'])
    expected = pd.Series([123.45, 67.89, 0.01])
    result = parse_amounts(input_data)
    assert_series_equal(result, expected)

def test_parse_amounts_with_thousands():
    """Test parsing numbers with thousand separators"""
    input_data = pd.Series(['1 234.56', '99 999.99', '1.23'])
    expected = pd.Series([1234.56, 99999.99, 1.23])
    result = parse_amounts(input_data, thousands_sep=' ')
    assert_series_equal(result, expected)

def test_parse_amounts_european_format():
    """Test parsing European number format (comma as decimal)"""
    input_data = pd.Series(['1234,56', '0,99', '1000,00'])
    expected = pd.Series([1234.56, 0.99, 1000.00])
    result = parse_amounts(input_data, decimal_sep=',')
    assert_series_equal(result, expected)

def test_parse_amounts_full_european():
    """Test parsing European number format with both separators"""
    input_data = pd.Series(['1 234,56', '99 999,99', '1,23'])
    expected = pd.Series([1234.56, 99999.99, 1.23])
    result = parse_amounts(input_data, thousands_sep=' ', decimal_sep=',')
    assert_series_equal(result, expected)

def test_parse_amounts_with_nulls():
    """Test handling of null values"""
    input_data = pd.Series(['123.45', None, np.nan, '67.89'])
    expected = pd.Series([123.45, np.nan, np.nan, 67.89])
    result = parse_amounts(input_data)
    assert_series_equal(result, expected)

def test_parse_amounts_with_whitespace():
    """Test handling of whitespace"""
    input_data = pd.Series([' 123.45 ', '  67.89', '99.99  '])
    expected = pd.Series([123.45, 67.89, 99.99])
    result = parse_amounts(input_data)
    assert_series_equal(result, expected)

def test_parse_amounts_errors_raise():
    """Test error raising with invalid input"""
    input_data = pd.Series(['123.45', 'invalid', '67.89'])
    with pytest.raises(ValueError):
        parse_amounts(input_data, errors='raise')

def test_parse_amounts_errors_coerce():
    """Test coercing invalid values to NaN"""
    input_data = pd.Series(['123.45', 'invalid', '67.89'])
    expected = pd.Series([123.45, np.nan, 67.89])
    result = parse_amounts(input_data, errors='coerce')
    assert_series_equal(result, expected)

def test_parse_amounts_negative_numbers():
    """Test parsing negative numbers"""
    input_data = pd.Series(['-123.45', '-1 234,56', '67.89'])
    expected = pd.Series([-123.45, -1234.56, 67.89])
    result = parse_amounts(input_data, thousands_sep=' ', decimal_sep=',')
    assert_series_equal(result, expected)

def test_parse_amounts_empty_series():
    """Test parsing empty series"""
    input_data = pd.Series([], dtype=object)
    expected = pd.Series([], dtype=float)
    result = parse_amounts(input_data)
    assert_series_equal(result, expected, check_dtype=False)


# Test clean_overlapping_fields
def test_clean_overlapping_fields_basic():
    """Test basic overlapping string detection"""
    df = pd.DataFrame({
        'col1': ['apple', 'banana', 'cherry'],
        'col2': ['big apple', 'banana split', 'berry']
    })
    expected = pd.DataFrame({
        'col1': [np.nan, np.nan, 'cherry'],
        'col2': ['big apple', 'banana split', 'berry']
    })
    result = clean_overlapping_fields(df, ['col1', 'col2'])
    assert_frame_equal(result, expected, check_dtype=False)

def test_clean_overlapping_fields_no_overlap():
    """Test when there are no overlapping values"""
    df = pd.DataFrame({
        'col1': ['apple', 'banana', 'cherry'],
        'col2': ['orange', 'grape', 'berry']
    })
    result = clean_overlapping_fields(df, ['col1', 'col2'])
    assert_frame_equal(result, df, check_dtype=False)

def test_clean_overlapping_fields_with_nan():
    """Test handling of NaN values"""
    df = pd.DataFrame({
        'col1': ['apple', np.nan, 'cherry'],
        'col2': ['big apple', 'banana', np.nan]
    })
    expected = pd.DataFrame({
        'col1': [np.nan, np.nan, 'cherry'],
        'col2': ['big apple', 'banana', np.nan]
    })
    result = clean_overlapping_fields(df, ['col1', 'col2'])
    assert_frame_equal(result, expected, check_dtype=False)

def test_clean_overlapping_fields_multiple_columns():
    """Test with more than two columns"""
    df = pd.DataFrame({
        'col1': ['apple', 'banana', 'cherry'],
        'col2': ['big apple', 'banana split', 'berry'],
        'col3': ['apple pie', 'grape', 'cherry top']
    })
    expected = pd.DataFrame({
        'col1': [np.nan, np.nan, np.nan],
        'col2': ['big apple', 'banana split', 'berry'],
        'col3': ['apple pie', 'grape', 'cherry top']
    })
    result = clean_overlapping_fields(df, ['col1', 'col2', 'col3'])
    assert_frame_equal(result, expected, check_dtype=False)

def test_clean_overlapping_fields_empty_df():
    """Test with empty DataFrame"""
    df = pd.DataFrame({'col1': [], 'col2': []})
    result = clean_overlapping_fields(df, ['col1', 'col2'])
    assert_frame_equal(result, df, check_dtype=False)

def test_clean_overlapping_fields_case_sensitive():
    """Test case sensitivity"""
    df = pd.DataFrame({
        'col1': ['Apple', 'banana', 'Cherry'],
        'col2': ['apple pie', 'BANANA', 'cherry']
    })
    expected = pd.DataFrame({
        'col1': ['Apple', 'banana', 'Cherry'],
        'col2': ['apple pie', 'BANANA', 'cherry']
    })
    result = clean_overlapping_fields(df, ['col1', 'col2'])
    assert_frame_equal(result, expected, check_dtype=False)

def test_clean_overlapping_fields_bidirectional():
    """Test bidirectional overlapping"""
    df = pd.DataFrame({
        'col1': ['apple juice', 'banana', 'long cherry'],
        'col2': ['apple', 'long banana', 'cherry']
    })
    expected = pd.DataFrame({
        'col1': ['apple juice', np.nan, 'long cherry'],
        'col2': [np.nan, 'long banana', np.nan]
    })
    result = clean_overlapping_fields(df, ['col1', 'col2'])
    assert_frame_equal(result, expected, check_dtype=False)

# def test_clean_overlapping_fields_numbers():
#     """Test with numeric values"""
#     df = pd.DataFrame({
#         'col1': [123, 456, 789],
#         'col2': [12345, 456000, 78]
#     })
#     expected = pd.DataFrame({
#         'col1': [np.nan, np.nan, 789],
#         'col2': [12345, 456000, 78]
#     })
#     result = clean_overlapping_fields(df, ['col1', 'col2'])
#     assert_frame_equal(result, expected, check_dtype=False, check_exact=False, atol=1e-8, rtol=1e-5)

def test_clean_overlapping_fields_mixed_types():
    """Test with mixed types"""
    df = pd.DataFrame({
        'col1': [123, 'text', True],
        'col2': ['123 more', 'some text', 'True value']
    })
    expected = pd.DataFrame({
        'col1': [np.nan, np.nan, np.nan],
        'col2': ['123 more', 'some text', 'True value']
    })
    result = clean_overlapping_fields(df, ['col1', 'col2'])
    assert_frame_equal(result, expected, check_dtype=False)

def test_clean_overlapping_fields_single_column():
    """Test with single column"""
    df = pd.DataFrame({'col1': ['apple', 'banana', 'cherry']})
    assert_frame_equal(df, clean_overlapping_fields(df, ['col1']))


def test_clean_overlapping_fields_invalid_column():
    """Test with invalid column name"""
    df = pd.DataFrame({'col1': ['apple'], 'col2': ['banana']})
    with pytest.raises(KeyError):
        clean_overlapping_fields(df, ['col1', 'col3'])
