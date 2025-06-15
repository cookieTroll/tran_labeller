import os
from typing import Any, Union, Optional

import pandas as pd
import yaml
import json
import logging

def is_path(file_path:str)-> Optional[bool]:
  """Verifies if a file path exists."""
  if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")
  else:
    return True


def load_config(file_path:str, encoding='utf-8')->Optional[Union[dict[str, Any], list[Any]]]:
  """Loads and parses a YAML configuration file."""
  if is_path(file_path):
    with open(file_path, 'r', encoding=encoding) as file:
      cf = yaml.safe_load(file)
      logging.info(f'Loaded config from :{file_path}')
      logging.info(cf)
      return cf

def read_csv(file_path: str, delimiter:str=',', encoding:Optional[str]=None)->pd.DataFrame:
  """ Reads a CSV file and returns a Pandas DataFrame."""
  df = pd.read_csv(f'{file_path}', delimiter=delimiter, encoding=encoding)
  logging.info(f'Loaded CSV from :{file_path}')
  logging.info(df)
  return df

def read_write_json(file_path:str, mode:str, payload:dict[str, Any]=None, encoding=None)->Optional[dict[str, Any]]:
  if encoding is None:
    encoding='utf-8'

  with open(file_path, mode, encoding=encoding) as f:
    if mode == 'w':
      json.dump(payload, f, ensure_ascii=False, indent=2)
      logging.info(f"Json saved to {file_path}")
    if mode == 'r':
      logging.info(f"Loaded JSON from {file_path}. {payload}")
      return json.load(f)

    return None