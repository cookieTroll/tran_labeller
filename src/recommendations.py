from typing import Union, Any

import pandas as pd
import logging


def generate_suggestions(
    row: pd.DataFrame, id_cols: list[str], keywords: dict[str, list[str]]
) -> Union[str, list[str]]:
    """
    Simple keywords match suggestions.

    Searches for keywords in specified columns and returns matching categories.
    Each id_col value is searched for all keywords and then list of all
    matching categories is returned.

    Args:
        row: Data row to analyze
        id_cols: Columns to search for keywords
        keywords: Mapping of categories to their respective keywords

    Returns:
        Single category string if exactly one match, otherwise list of matches

    Raises:
        ValueError: When id_cols or keywords are empty
        KeyError: When any of id_cols is missing from row
    """

    if not id_cols or not keywords:
        raise ValueError("id_cols and keywords cannot be empty")
    if not all(col in row for col in id_cols):
        raise KeyError(f"Missing columns: {set(id_cols) - set(row.keys())}")

    suggested_keywords = set()
    for category, kw in keywords.items():
        for keyword in kw:
            for col in id_cols:
                if keyword.lower() in str(row[col]).lower():
                    suggested_keywords.add(category)
    return (
        suggested_keywords.pop()
        if len(suggested_keywords) == 1
        else list(suggested_keywords)
    )


def obtain_index_to_fix(
    data: pd.DataFrame,
    config: dict[str, Any],
    incl_inbound: bool = False,
    incl_conflicts: bool = False,
    incl_missing_suggestions: bool = False,
    ind_list: list[int] = None,
) -> tuple[list[int], dict[str, str]]:
    """
    Heuristic function, which returns ids of records, which should be manually fixed. The options are (each by separate flag):
      - to include all incoming transactions,
      - to include those rows for which the category suggestion and original category misalign,
      - to include those rows which do not get any suggestion
      - manual list of ids
    """
    ind = []
    cat_col = config["output_format"]["category_col"]
    transaction_type, inbound_flag = (
        config["output_format"]["transaction_type"],
        config["input_format"]["inbound_keyword"],
    )
    suggestions = config["output_format"]["generated_suggestions"]
    date = config["output_format"]["date"]
    amount = config["output_format"]["amount"]

    if incl_inbound:
        ind += data[
            data[transaction_type].str.contains(inbound_flag, na=False)
        ].index.tolist()  # to-do: make more agnostic
    if incl_conflicts:
        ind += data[
            data.apply(lambda row: str(row[cat_col]) != str(row[suggestions]), axis=1)
        ].index.tolist()
    if incl_missing_suggestions:
        ind += data[
            (data[suggestions].isna()) | (data[suggestions] == "")
        ].index.tolist()
    if ind_list:
        ind += ind_list

    ind = list(set(ind))
    non_matching_data = data.loc[ind].to_dict("index")
    fix_dict = {
        k: f"""{{ '{cat_col}': '{v[cat_col]}'/'{v['suggested_categories']}'}},
            #original: {v[cat_col]}, suggested: {v['suggested_categories']},
            proti: {v['protistrana']}, zprava: {v['zpráva']}, datum: {v[date]} castka: {v[amount]}
      """
        for k, v in non_matching_data.items()
    }

    logging.info(
        f"Found {len(ind)} records to fix. "
        f"Filters used: inbound={incl_inbound}, "
        f"conflicts={incl_conflicts}, "
        f"missing={incl_missing_suggestions}, "
        f"manual_list={'yes' if ind_list else 'no'}"
    )
    logging.info(fix_dict)

    return ind, fix_dict
