from io import StringIO

import numpy as np
import pandas as pd
import pytest

from postprocessor import (generate_aggregate_data, save_output,
                           update_categories)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "category": ["food", "transport", "entertainment"],
            "amount": [100.0, 200.0, 300.0],
            "date": ["2025-01-01", "2025-01-01", "2025-01-02"],
        }
    )


@pytest.fixture
def sample_logs():
    """Create a sample StringIO object with logs."""
    logs = StringIO()
    logs.write("Log entry 1\nLog entry 2\nLog entry 3")
    return logs


class TestUpdateCategories:
    def test_successful_update(self, sample_df):
        update_dict = {"0": "{'category': 'groceries'}", "2": "{'category': 'leisure'}"}

        result = update_categories(sample_df, "category", update_dict)

        assert result.at["0", "category"] == "groceries"
        assert result.at["2", "category"] == "leisure"
        assert result.at["1", "category"] == "transport"  # Unchanged
        assert not result.equals(sample_df)  # Ensure original df wasn't modified

    def test_invalid_update_dict_syntax(self, sample_df):
        update_dict = {
            "0": "{'category': 'groceries'",  # Missing closing brace
        }

        with pytest.raises(
            ValueError, match="At index 0, the result is not a dictionary"
        ):
            update_categories(sample_df, "category", update_dict)

    def test_non_dict_evaluation(self, sample_df):
        update_dict = {
            "0": "'just a string'",  # Evaluates to string, not dict
        }

        with pytest.raises(
            ValueError, match="At index 0, the result is not a dictionary"
        ):
            update_categories(sample_df, "category", update_dict)

    def test_empty_update_dict(self, sample_df):
        result = update_categories(sample_df, "category", {})
        pd.testing.assert_frame_equal(result, sample_df)


class TestGenerateAggregateData:
    def test_successful_aggregation(self, sample_df):
        result = generate_aggregate_data(
            data=sample_df,
            category_col="category",
            excl_categories=["transport"],
            grouper=["date"],
            amount_col="amount",
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # Two unique dates
        assert all(
            col in result.columns
            for col in [
                "date",
                ("amount", "sum"),
                ("amount", "mean"),
                ("amount", "count"),
            ]
        )

    def test_missing_column(self, sample_df):
        with pytest.raises(ValueError, match="Column 'nonexistent' not found in data"):
            generate_aggregate_data(
                data=sample_df,
                category_col="nonexistent",
                excl_categories=[],
                grouper=["date"],
                amount_col="amount",
            )

    def test_exclude_all_categories(self, sample_df):
        result = generate_aggregate_data(
            data=sample_df,
            category_col="category",
            excl_categories=["food", "transport", "entertainment"],
            grouper=["date"],
            amount_col="amount",
        )

        assert result.empty

    def test_case_insensitive_exclusion(self, sample_df):
        result = generate_aggregate_data(
            data=sample_df,
            category_col="category",
            excl_categories=["FOOD", "Transport"],
            grouper=["date"],
            amount_col="amount",
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestSaveOutput:
    @pytest.fixture
    def sample_dfs(self, sample_df):
        # Create variations of sample_df for different stages
        raw_data = sample_df.copy()
        parsed_data = sample_df.copy()
        parsed_data["amount"] = parsed_data["amount"] * 2
        final_data = parsed_data.copy()
        final_data["category"] = final_data["category"].str.upper()

        # Create aggregate data
        aggregate_data = pd.DataFrame(
            {
                "date": ["2025-01-01", "2025-01-02"],
                "amount_sum": [300.0, 300.0],
                "amount_mean": [150.0, 300.0],
                "amount_count": [2, 1],
            }
        )

        return raw_data, parsed_data, final_data, aggregate_data

    def test_successful_save(self, tmp_path, sample_dfs, sample_logs):
        output_file = tmp_path / "test_output.xlsx"
        raw_data, parsed_data, final_data, aggregate_data = sample_dfs

        save_output(
            str(output_file),
            raw_data,
            parsed_data,
            final_data,
            aggregate_data,
            sample_logs,
        )

        # Verify file exists
        assert output_file.exists()

        # Read back and verify contents
        with pd.ExcelFile(output_file) as xlsx:
            assert set(xlsx.sheet_names) == {
                "Original Data",
                "Processed Data",
                "Updated Categories",
                "Monthly Expenses",
                "Logs",
            }

            # Verify data in sheets
            pd.testing.assert_frame_equal(
                pd.read_excel(xlsx, "Original Data"), raw_data
            )
            pd.testing.assert_frame_equal(
                pd.read_excel(xlsx, "Processed Data"), parsed_data
            )
            pd.testing.assert_frame_equal(
                pd.read_excel(xlsx, "Updated Categories"), final_data
            )
            pd.testing.assert_frame_equal(
                pd.read_excel(xlsx, "Monthly Expenses"), aggregate_data
            )

            logs_df = pd.read_excel(xlsx, "Logs")
            assert len(logs_df) == 3  # Three log entries
            assert list(logs_df.columns) == ["Log"]

    def test_invalid_file_extension(self, tmp_path, sample_dfs, sample_logs):
        output_file = tmp_path / "test_output.txt"
        raw_data, parsed_data, final_data, aggregate_data = sample_dfs

        with pytest.raises(ValueError):
            save_output(
                str(output_file),
                raw_data,
                parsed_data,
                final_data,
                aggregate_data,
                sample_logs,
            )

    def test_empty_dataframes(self, tmp_path, sample_logs):
        output_file = tmp_path / "test_output.xlsx"
        empty_df = pd.DataFrame()

        save_output(
            str(output_file), empty_df, empty_df, empty_df, empty_df, sample_logs
        )

        assert output_file.exists()

    def test_empty_logs(self, tmp_path, sample_dfs):
        output_file = tmp_path / "test_output.xlsx"
        raw_data, parsed_data, final_data, aggregate_data = sample_dfs
        empty_logs = StringIO()

        save_output(
            str(output_file),
            raw_data,
            parsed_data,
            final_data,
            aggregate_data,
            empty_logs,
        )

        with pd.ExcelFile(output_file) as xlsx:
            logs_df = pd.read_excel(xlsx, "Logs")
            assert logs_df.empty
