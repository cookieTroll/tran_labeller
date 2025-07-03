from typing import Union

import pandas as pd


def generate_suggestions(row:pd.DataFrame, id_cols:list[str], keywords:dict[str, list[str]])->Union[str, list[str]]:
  """
    Simple keywords match suggestions. There are categories with specified keywords and there's a row of data with selected columns.
    Each id_col value is searched for all keywords and then list of all matching categories is returned.
  """

  suggested_keywords = set()
  for category, kw in keywords.items():
    for keyword in kw:
      for col in id_cols:
        if keyword.lower() in str(row[col]).lower():
          suggested_keywords.add(category)
  return suggested_keywords.pop() if len(suggested_keywords) == 1 else list(suggested_keywords)
