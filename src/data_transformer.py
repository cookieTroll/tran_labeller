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
    Clean overlapping values in specified columns. If value from one column
    is contained within another column's value, replace it with NaN.

    Args:
        df: Input DataFrame
        column_ids: List of column names to check for overlaps

    Returns:
        DataFrame with cleaned overlapping values
    """
    df = df.copy()

    # Convert all values to strings and handle NaN values
    for col in column_ids:
        df[col] = df[col].astype(str).replace("nan", np.nan)

    # Check each pair of columns for overlaps
    for i, col1 in enumerate(column_ids):
        for col2 in column_ids[i + 1 :]:
            mask1 = df[col1].notna() & df[col2].notna()
            if not mask1.any():
                continue

            # Create masks for overlapping values in both directions
            contains_mask1 = mask1 & df.apply(
                lambda x: str(x[col1]) in str(x[col2]), axis=1
            )
            contains_mask2 = mask1 & df.apply(
                lambda x: str(x[col2]) in str(x[col1]), axis=1
            )

            # Apply masks
            df.loc[contains_mask1, col1] = np.nan
            df.loc[contains_mask2, col2] = np.nan

    return df


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
    """The function takes loaded data, parses format and transforms them to a more convenient format."""
    data_raw = data.copy(deep=True)
    i_f = config["input_format"]
    o_f = config["output_format"]

    keep_cols = [
        o_f["transaction_type"],
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
        data_raw[o_f["payment_catyegory"]] = None

    # amount
    data_raw[o_f["amount"]] = parse_amounts(
        data_raw[i_f["amount"]["col"]],
        i_f["amount"]["format"]["thousands_sep"],
        i_f["amount"]["format"]["decimal_sep"],
        errors,
    )

    # date
    data_raw[o_f["date"]["col"]] = parse_dates(
        data_raw[i_f["date"]["col"]], i_f["date"].get("format", DATE_FORMATS), errors
    )
    data_raw[o_f["date"]["col"]] = data_raw[o_f["date"]["col"]].dt.strftime("%Y-%m-%d")
    data_raw["month"] = data_raw[o_f["date"]["col"]].dt.strftime("%Y%m")

    # concat columns - protistrana & message
    data_raw[o_f["counterparty"]] = combine_fields(data_raw, i_f["counterparty_ids"])
    data_raw[o_f["message"]] = combine_fields(data_raw, i_f["message"])

    return data_raw[keep_cols]
