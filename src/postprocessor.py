import ast

import pandas as pd


def update_categories(
    df: pd.DataFrame, category_col: str, update_dict: dict[str, str]
) -> pd.DataFrame:
    """
    Updates the categories of a DataFrame column based on a given update mapping.

    This function creates a copy of the provided DataFrame, iterates through the
    update dictionary, and updates the specified column's values for each index
    based on the changes defined in the dictionary. Each key in the update
    dictionary represents a corresponding index in the DataFrame, and each value
    represents a mapping of new values for the specified column. Values in the
    update dictionary are expected to be strings that can be evaluated as
    dictionaries.

    Parameters:
    df : pd.DataFrame
        The input pandas DataFrame that contains the data to be updated.
    category_col : str
        The name of the column in the DataFrame whose categories are to be updated.
    update_dict : dict[str, str]
        A dictionary defining the updates to be applied. Each key corresponds to
        a row index in the DataFrame, while the associated value is a string that
        can be evaluated as a dictionary containing updates for the column.

    Returns:
    pd.DataFrame
        A new pandas DataFrame with the specified updates applied to the
        specified column.

    Raises:
    ValueError
        If a value in the update dictionary cannot be evaluated to a dictionary.
        If the evaluated dictionary is not actually a dictionary or does not
        contain the specified column as a key.
    """
    data = df.copy()
    for idx, ch in update_dict.items():
        try:
            # First try to evaluate the string

            changes = ast.literal_eval(ch)
            # Check if the result is actually a dictionary
            if not isinstance(changes, dict):
                raise ValueError("At index %s, the result is not a dictionary", idx)
        except (SyntaxError, ValueError):
            raise ValueError("At index %s, the result is not a dictionary", idx)

        data.at[int(idx), category_col] = changes[category_col]
    return data


def generate_aggregate_data(
    data: pd.DataFrame,
    category_col: str,
    excl_categories: list[str],
    grouper: list[str],
    amount_col: str,
) -> pd.DataFrame:
    """Function provides basic metrics on a predefined split - sum, count and mean."""
    for col in [category_col, amount_col] + grouper:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")

    excl_categories_lcase = [itm.lower() for itm in excl_categories]

    df_agg = (
        data[~data[category_col].str.lower().isin(excl_categories_lcase)]
        .groupby(grouper)
        .agg({amount_col: ["sum", "mean", "count"]})
        .reset_index()
    )
    return df_agg


def save_output(
    output_file_name: str,
    raw_data: pd.DataFrame,
    parsed_data: pd.DataFrame,
    final_data: pd.DataFrame,
    aggregate_data: pd.DataFrame,
    logs,
):
    """
    Saves multiple data tables and logs into an Excel workbook with dedicated sheets for each.

    This function collects raw, processed, updated, and aggregate data, along with logs, and writes
    each to a separate sheet in a specified Excel file. It utilizes pandas for handling both the data
    and the Excel writing process.

    Arguments:
        output_file_name: str
            Name of the Excel file to save the output. Should end with .xlsx.
        raw_data: pd.DataFrame
            Raw dataset to be saved into a sheet titled 'Original Data'.
        parsed_data: pd.DataFrame
            Processed dataset to be saved into a sheet titled 'Processed Data'.
        final_data: pd.DataFrame
            Updated and categorized dataset to be saved into a sheet titled 'Updated Categories'.
        aggregate_data: pd.DataFrame
            Aggregated dataset reflecting monthly expenses to be saved into a sheet
            titled 'Monthly Expenses'.
        logs
            Log messages captured in a StringIO instance, transformed into a DataFrame and
            saved into a sheet titled 'Logs'.

    Raises:
        ValueError
            Raised if the `output_file_name` does not have a valid file extension.
    """
    log_contents = logs.getvalue()
    log_df = pd.DataFrame(log_contents.strip().split("\n"), columns=["Log"])

    # Create a new Excel workbook
    with pd.ExcelWriter(output_file_name) as writer:
        # Write each DataFrame to a different sheet
        raw_data.to_excel(writer, sheet_name="Original Data", index=False)
        parsed_data.to_excel(writer, sheet_name="Processed Data", index=False)
        final_data.to_excel(writer, sheet_name="Updated Categories", index=False)
        aggregate_data.to_excel(writer, sheet_name="Monthly Expenses", index=False)
        log_df.to_excel(writer, sheet_name="Logs", index=False)
