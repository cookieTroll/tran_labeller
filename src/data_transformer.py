"""This module contains functions to transform data."""

import logging
from typing import Any

import numpy as np
import pandas as pd


DATE_FORMATS = [
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
]


def parse_dates(
    series: pd.Series, formats: list[str], errors: str = "raise"
) -> pd.Series:
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
            return pd.to_datetime(clean_series, format=fmt, errors="raise")
        except ValueError:
            continue

    # If we get here, none of the formats worked
    if errors == "raise":
        # Find an example of problematic value for debugging
        sample_value = clean_series[
            clean_series.apply(lambda x: isinstance(x, str) and not pd.isnull(x))
        ].iloc[0]
        logging.error(
            f"Could not parse date: '{sample_value}' "
            f"with any of the formats: {formats}"
        )
        raise ValueError(f"No matching date format for value: {sample_value}")

    # Try pandas' flexible parser as a last resort
    return pd.to_datetime(clean_series, errors=errors)


def parse_amounts(
    series: pd.Series,
    thousands_sep: str | None = None,
    decimal_sep: str = ".",
    errors: str = "raise",
) -> pd.Series:
    """
    Parse numeric amounts from a pandas Series with specified separators.

    Args:
        series: Input series containing amount strings
        thousands_sep: Thousand separator character (' ' or ',' or None)
        decimal_sep: Decimal separator character ('.' or ',')
        errors: How to handle parsing errors ('raise', 'coerce', or 'ignore')

    Returns:
        Series with parsed float values

    Raises:
        ValueError: If errors='raise' and parsing fails
    """
    # Remove any leading/trailing whitespace
    clean_series = series.str.strip()

    try:
        # Remove thousands separator if specified
        if thousands_sep:
            clean_series = clean_series.str.replace(thousands_sep, "", regex=False)

        # Replace decimal separator with standard period if different
        if decimal_sep != ".":
            clean_series = clean_series.str.replace(decimal_sep, ".", regex=False)

        return pd.to_numeric(clean_series, errors=errors)

    except Exception as e:
        if errors == "raise":
            sample_value = clean_series[
                clean_series.apply(lambda x: isinstance(x, str) and not pd.isnull(x))
            ].iloc[0]
            logging.error(f"Could not parse amount: '{sample_value}'")
            raise ValueError(
                f"Could not parse number format for value: {sample_value}"
            ) from e

        return pd.to_numeric(clean_series, errors=errors)


def clean_overlapping_fields(df: pd.DataFrame, column_ids: list[str]) -> pd.DataFrame:
    """
    Clean overlapping fields by removing redundant information.
    If the same text appears in multiple columns, keep it only in one place
    according to these rules:
    1. For exact matches, keep value in the first column
    2. If one string contains another, keep the longer string

    Args:
        df: DataFrame to process
        column_ids: List of column names to check for overlaps

    Returns:
        DataFrame with cleaned overlapping fields

    Example:
        >>> df = pd.DataFrame({
        ...     'col1': ['apple', 'banana', 'cherry'],
        ...     'col2': ['big apple', 'banana', 'berry']
        ... })
        >>> clean_overlapping_fields(df, ['col1', 'col2'])
           col1       col2
        0  NaN   big apple
        1  NaN     banana
        2  cherry    berry
    """
    data = df.copy()

    # Convert all relevant columns to string type
    for col in column_ids:
        data[col] = data[col].astype(str)

    # Process each pair of columns
    for i, col1 in enumerate(column_ids):
        for col2 in column_ids[i + 1 :]:
            # Create mask for non-null pairs
            valid_pairs = data[col1].notna() & data[col2].notna()
            if not valid_pairs.any():
                continue

            # Get series of valid pairs
            s1 = data.loc[valid_pairs, col1]
            s2 = data.loc[valid_pairs, col2]

            # Find exact matches
            exact_matches = s1 == s2
            data.loc[valid_pairs & exact_matches, col2] = np.nan

            # Find contained strings
            contains1 = s2.str.contains(s1, regex=False)
            contains2 = s1.str.contains(s2, regex=False)

            # Update based on containment
            data.loc[valid_pairs & contains1, col1] = np.nan
            data.loc[valid_pairs & contains2, col2] = np.nan

    return data


def combine_fields(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """
    Combine field groups into single columns based on configuration.

    Args:
        df: Input DataFrame
        cols: list of columns to combine

    Returns:
        DataFrame with combined fields
    """
    df = df.copy()
    df = clean_overlapping_fields(df, cols)

    # Combine non-null values with separator
    out = df[cols].apply(
        lambda x: "__".join(str(val) for val in x if pd.notna(val)), axis=1
    )
    out = out.replace("", np.nan)
    return out


def parse_data(
    data: pd.DataFrame, config: dict[str, Any], errors="coerce"
) -> pd.DataFrame:
    """
    Parses and transforms financial data from one format to another based on provided
    configuration.

    The function processes a DataFrame by performing transformations such as renaming
    columns, parsing amounts and dates, and combining multiple fields into new columns.
    It makes use of a configuration dictionary to map input and output formats and applies
    data manipulation accordingly. The resulting DataFrame contains only the specified
    columns in the desired format.

    Parameters:
    data: pd.DataFrame
        The input DataFrame containing the raw financial data to be parsed and formatted.
    config: dict[str, Any]
        A dictionary specifying the input and output data format mappings, including
        column names, date formats, and any field combining rules.
    errors: str, optional
        Defines how to handle parsing errors; default is "coerce", which forces invalid
        data into NaN values.

    Returns:
    pd.DataFrame
        The transformed DataFrame containing only the specified columns formatted
        according to the output configuration.
    """
    data_raw = data.copy(deep=True)
    i_f = config["input_format"]
    o_f = config["output_format"]

    keep_cols = [
        o_f["transaction_type"],
        o_f["payment_category"],
        o_f["amount"],
        o_f["date"]["col"],
        o_f["counterparty"],
        o_f["message"],
        "month",
    ]

    # tran_type
    data_raw[o_f["transaction_type"]] = data_raw[i_f["transaction_type"]]

    # payment_category
    if (
        not i_f.get("payment_category", None)
        or i_f["payment_category"] not in data_raw.columns
    ):
        data_raw[o_f["payment_category"]] = None
    else:
        data_raw[o_f["payment_category"]] = data_raw[i_f["payment_category"]]

    # amount
    data_raw[o_f["amount"]] = parse_amounts(
        data_raw[i_f["amount"]["col"]],
        i_f["amount"]["format"]["thousands_separator"],
        i_f["amount"]["format"]["decimal_separator"],
        errors,
    )

    # date
    data_raw[o_f["date"]["col"]] = parse_dates(
        data_raw[i_f["date"]["col"]], i_f["date"].get("format", DATE_FORMATS), errors
    )
    data_raw["month"] = data_raw[o_f["date"]["col"]].dt.strftime("%Y%m")
    data_raw[o_f["date"]["col"]] = data_raw[o_f["date"]["col"]].dt.strftime("%Y-%m-%d")

    # concat columns - protistrana & message
    data_raw[o_f["counterparty"]] = combine_fields(data_raw, i_f["counterparty_ids"])
    data_raw[o_f["message"]] = combine_fields(data_raw, i_f["message"])

    return data_raw[keep_cols]
