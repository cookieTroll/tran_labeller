"""These functions are used to load data from files"""

import json
import logging
import os
from typing import Any, Optional, Union

import pandas as pd
import yaml


def is_path(file_path: str) -> Optional[bool]:
    """Verifies if a file path exists."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    else:
        return True


def load_config(
    file_path: str, encoding="utf-8"
) -> Optional[Union[dict[str, Any], list[Any]]]:
    """Loads and parses a YAML configuration file."""
    if is_path(file_path):
        with open(file_path, "r", encoding=encoding) as file:
            cf = yaml.safe_load(file)
            logging.info(f"Loaded config from :{file_path}")
            logging.info(cf)
            return cf


def construct_config(config: dict[str, Any]) -> dict[str, Any]:
    """Constructs a dictionary based on a high-level config."""
    cf = {}
    try:
        for key, file_path in config.items():
            if "config" not in key:
                cf[key] = file_path
            else:
                if not isinstance(file_path, str):
                    raise ValueError(f"Invalid path for {key}: must be string")

                cf[key] = load_config(file_path)

        logging.info(
            "Successfully constructed configuration from %d files", len(config)
        )
        return cf

    except Exception as e:
        logging.error("Failed to construct configuration: %s", e)
        raise


def read_csv(
    file_path: str, delimiter: str = ",", encoding: Optional[str] = None
) -> pd.DataFrame:
    """Reads a CSV file and returns a Pandas DataFrame."""
    df = pd.read_csv(f"{file_path}", delimiter=delimiter, encoding=encoding)
    logging.info(f"Loaded CSV from :{file_path}")
    logging.info(df)
    return df


def read_write_json(
    file_path: str, mode: str, payload: dict[str, Any] = None, encoding=None
) -> Optional[dict[str, Any]]:
    """Reads or writes a JSON file based on mode (r/w)."""
    if encoding is None:
        encoding = "utf-8"

    with open(file_path, mode, encoding=encoding) as f:
        if mode == "w":
            json.dump(payload, f, ensure_ascii=False, indent=2)
            logging.info(f"Json saved to {file_path}")
        if mode == "r":
            logging.info(f"Loaded JSON from {file_path}. {payload}")
            return json.load(f)

        return None
