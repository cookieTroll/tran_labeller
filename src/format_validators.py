""" Script contains functions to validate configuration files."""
import logging
import pandas as pd
from typing import List, Optional, Any

from openpyxl.pivot.fields import Boolean
from pydantic import BaseModel, Field, ValidationError


class AmountFormat(BaseModel):
    decimal_separator: str
    thousands_separator: str


class AmountConfig(BaseModel):
    col: str
    format: Optional[AmountFormat] = None


class DateConfig(BaseModel):
    col: str
    format: Optional[str] = None


class OutputFormat(BaseModel):
    display_groups: str
    counterparty: str
    message: str
    amount: str
    date: str
    transaction_type: str
    payment_category: str
    generated_suggestions: str
    sort_by: List[str]


class InputFormat(BaseModel):
    payment_category: str
    date: DateConfig
    counterparty_ids: List[str]
    message: List[str]
    amount: AmountConfig
    transaction_type: str
    inbound_keyword: str


class ParsingConfig(BaseModel):
    delimiter: str



class ConfigSchema(BaseModel):
    file_name_input: str
    file_name_output: str
    fix_file_name: str
    parsing: Optional[ParsingConfig] = None
    input_format: InputFormat
    output_format: OutputFormat


def validate_bank_config(config_dict: dict[str, Any]) -> bool:
    """
    Validate the configuration dictionary against the schema.

    Args:
        config_dict: Dictionary containing the configuration

    Returns:
        ConfigSchema: Validated configuration object

    Raises:
        ValidationError: If the configuration is invalid
    """
    try:
        ConfigSchema(**config_dict)
        return True
    except ValidationError as e:
        logging.critical("Bank configuration validation failed:")
        for error in e.errors():
            logging.critical(f"- {error['loc']}: {error['msg']}")
        raise e


def validate_data(data:pd.DataFrame, cols:list[str])->bool:
    """Validates that all columns in cols are present in the data."""
    check = set(cols).issubset(set(data.columns))
    if not check:
        logging.critical(f"Missing columns in data: {set(cols) - set(data.columns)}")
        raise ValueError(f"Missing columns in data: {set(cols) - set(data.columns)}")
    return check