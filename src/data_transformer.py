""" This module contains functions to transform data."""
import pandas as pd
import logging

def parse_dates(series: pd.Series, formats: list[str],
                errors: str = 'raise') -> pd.Series:
    """
    Parse dates from a pandas Series using multiple formats.

    Args:
        series: Input series containing date strings
        formats: List of date formats to try
        errors: How to handle parsing errors ('raise', 'coerce', or 'ignore')

    Returns:
        Series with parsed datetime values

    Raises:
        ValueError: If errors='raise' and no format matches
    """
    # Remove any leading/trailing whitespace
    clean_series = series.str.strip()

    # Try each format
    for fmt in formats:
        try:
            return pd.to_datetime(clean_series, format=fmt, errors='raise')
        except ValueError:
            continue

    # If we get here, none of the formats worked
    if errors == 'raise':
        # Find an example of problematic value for debugging
        sample_value = clean_series[
            clean_series.apply(lambda x: isinstance(x, str) and not pd.isnull(x))
        ].iloc[0]
        logging.error(f"Could not parse date: '{sample_value}' "
                      f"with any of the formats: {formats}")
        raise ValueError(f"No matching date format for value: {sample_value}")

    # Try pandas' flexible parser as a last resort
    return pd.to_datetime(clean_series, errors=errors)