"""Script contains functions to validate configuration files."""

import logging
from typing import Any, List, Literal, Optional, Union

import pandas as pd
from openpyxl.pivot.fields import Boolean
from pydantic import BaseModel, ValidationError


# fields within config
class AmountFormat(BaseModel):
    decimal_separator: str
    thousands_separator: str


class FormatConfig(BaseModel):
    """Base class for format configurations."""

    col: str
    format: Union[str, List[str], None] = None


class AmountConfig(FormatConfig):
    format: AmountFormat | None = None


class DateConfig(FormatConfig):
    format: List[str] | None = None


class ParsingConfig(BaseModel):
    delimiter: str


# structure of output config
class OutputFormat(BaseModel):
    display_groups: str
    counterparty: str
    message: str
    amount: str
    date: DateConfig
    transaction_type: str
    payment_category: Optional[str]
    categories_excl_from_agg: Optional[list[str]] = None
    generated_suggestions: str
    sort_by: List[str]


# structure of input config
class InputFormat(BaseModel):
    payment_category: str
    date: DateConfig
    counterparty_ids: List[str]
    message: List[str]
    amount: AmountConfig
    transaction_type: str
    inbound_keyword: str


class Input(BaseModel):
    input_format: InputFormat
    parsing: Optional[ParsingConfig] = None


# structure of high-level config which provides files to assemble full config
class BankConfig(BaseModel):
    file_name_input: str
    file_name_output: str
    fix_file_name: str

    input_config: str
    output_config: str
    keywords_config: str


# Full config used to drive the pipeline
class FullConfigSchema(BaseModel):
    file_name_input: str
    file_name_output: str
    fix_file_name: str
    input_config: Input
    output_config: OutputFormat
    keywords_config: Optional[dict[str, List[str]]] = None


def validate_config(
    config_dict: dict[str, Any], cf_type=Literal["Full", "High-level"]
) -> bool | BaseModel | None:
    """
    Validate the configuration dictionary against the schema.

    Args:
        config_dict: Dictionary containing the configuration
        cf_type: Literal['Full', 'High-level'] - Full config or high-level config.

    Returns:
        boolean: True if the configuration is valid, False otherwise.

    Raises:
        ValidationError: If the configuration is invalid
    """
    try:
        if cf_type == "Full":
            return FullConfigSchema(**config_dict)
        else:
            return BankConfig(**config_dict)

    except ValidationError as e:
        logging.critical("Bank configuration validation failed:")
        for error in e.errors():
            logging.critical(f"- {error['loc']}: {error['msg']}")
        raise e


def validate_data(data: pd.DataFrame, input_format_config: InputFormat) -> bool:
    """Validates that all columns in cols are present in the data."""

    i_f = input_format_config
    cols = (
        [
            i_f["payment_category"],
            i_f["date"]["col"],
            i_f["amount"]["col"],
            i_f["transaction_type"],
        ]
        + i_f["counterparty_ids"]
        + i_f["message"]
    )
    check = set(cols).issubset(set(data.columns))
    if not check:
        logging.critical(f"Missing columns in data: {set(cols) - set(data.columns)}")
        raise ValueError(f"Missing columns in data: {set(cols) - set(data.columns)}")
    return check
