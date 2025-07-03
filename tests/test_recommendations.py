import pytest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import numpy as np
from src.recommendations import generate_suggestions, obtain_index_to_fix


# Fixtures for common test data
@pytest.fixture
def sample_keywords():
    return {
        "groceries": ["food", "market", "grocery"],
        "utilities": ["electric", "water", "gas"],
        "entertainment": ["movie", "game", "netflix"],
    }


@pytest.fixture
def sample_config():
    return {
        "output_format": {
            "category_col": "category",
            "transaction_type": "type",
            "generated_suggestions": "suggested_categories",
            "date": "transaction_date",
            "amount": "transaction_amount",
        },
        "input_format": {"inbound_keyword": "incoming"},
    }


class TestGenerateSuggestions:
    def test_basic_matching(self, sample_keywords):
        """Test basic keyword matching functionality"""
        row = pd.Series(
            {"description": "Monthly Netflix subscription", "merchant": "NETFLIX.COM"}
        )
        result = generate_suggestions(
            row, id_cols=["description", "merchant"], keywords=sample_keywords
        )
        assert "entertainment" in result

    def test_multiple_category_matches(self, sample_keywords):
        """Test when text matches multiple categories"""
        row = pd.Series(
            {"description": "Food market game night", "merchant": "LOCAL MARKET"}
        )
        result = generate_suggestions(
            row, id_cols=["description", "merchant"], keywords=sample_keywords
        )
        assert isinstance(result, list)
        assert set(result) == {"groceries", "entertainment"}

    def test_case_insensitive_matching(self, sample_keywords):
        """Test case-insensitive matching"""
        row = pd.Series(
            {"description": "WATER Bill Payment", "merchant": "City utilities"}
        )
        result = generate_suggestions(
            row, id_cols=["description", "merchant"], keywords=sample_keywords
        )
        assert "utilities" in result

    def test_empty_input_validation(self, sample_keywords):
        """Test handling of empty inputs"""
        row = pd.Series({"description": "test"})

        with pytest.raises(ValueError):
            generate_suggestions(row, id_cols=[], keywords=sample_keywords)

        with pytest.raises(ValueError):
            generate_suggestions(row, id_cols=["description"], keywords={})

    def test_missing_columns(self, sample_keywords):
        """Test handling of missing columns"""
        row = pd.Series({"wrong_column": "test"})

        with pytest.raises(KeyError):
            generate_suggestions(row, id_cols=["description"], keywords=sample_keywords)

    def test_special_characters(self, sample_keywords):
        """Test handling of special characters and regex patterns"""
        row = pd.Series(
            {"description": "Food$ (market) [special]*", "merchant": "Special.Market"}
        )
        result = generate_suggestions(
            row, id_cols=["description", "merchant"], keywords=sample_keywords
        )
        assert "groceries" in result

    def test_null_values(self, sample_keywords):
        """Test handling of null values"""
        row = pd.Series({"description": np.nan, "merchant": "Food Market"})
        result = generate_suggestions(
            row, id_cols=["description", "merchant"], keywords=sample_keywords
        )
        assert "groceries" in result


class TestObtainIndexToFix:
    def test_inbound_transactions(self, sample_config):
        """Test identification of inbound transactions"""
        data = pd.DataFrame(
            {
                "type": ["incoming", "outgoing", "incoming"],
                "category": ["salary", "food", "refund"],
                "suggested_categories": ["salary", "groceries", "refund"],
                "transaction_date": ["2025-01-01"] * 3,
                "transaction_amount": [1000, -50, 100],
                "protistrana": ["EMPLOYER", "SHOP", "STORE"],
                "zpráva": ["Salary", "Food", "Refund"],
            }
        )

        indices, fix_dict = obtain_index_to_fix(data, sample_config, incl_inbound=True)
        assert set(indices) == {0, 2}

    def test_category_conflicts(self, sample_config):
        """Test identification of category conflicts"""
        data = pd.DataFrame(
            {
                "type": ["outgoing"] * 3,
                "category": ["food", "transport", "utilities"],
                "suggested_categories": ["groceries", "transport", "bills"],
                "transaction_date": ["2025-01-01"] * 3,
                "transaction_amount": [-50, -30, -100],
                "protistrana": ["SHOP", "TAXI", "UTILITY"],
                "zpráva": ["Food", "Ride", "Bills"],
            }
        )

        indices, fix_dict = obtain_index_to_fix(
            data, sample_config, incl_conflicts=True
        )
        assert set(indices) == {0, 2}

    def test_missing_suggestions(self, sample_config):
        """Test identification of missing suggestions"""
        data = pd.DataFrame(
            {
                "type": ["outgoing"] * 3,
                "category": ["food", "transport", "utilities"],
                "suggested_categories": ["groceries", "", np.nan],
                "transaction_date": ["2025-01-01"] * 3,
                "transaction_amount": [-50, -30, -100],
                "protistrana": ["SHOP", "TAXI", "UTILITY"],
                "zpráva": ["Food", "Ride", "Bills"],
            }
        )

        indices, fix_dict = obtain_index_to_fix(
            data, sample_config, incl_missing_suggestions=True
        )
        assert set(indices) == {1, 2}

    def test_manual_index_list(self, sample_config):
        """Test manual index list functionality"""
        data = pd.DataFrame(
            {
                "type": ["outgoing"] * 3,
                "category": ["food", "transport", "utilities"],
                "suggested_categories": ["groceries", "transport", "bills"],
                "transaction_date": ["2025-01-01"] * 3,
                "transaction_amount": [-50, -30, -100],
                "protistrana": ["SHOP", "TAXI", "UTILITY"],
                "zpráva": ["Food", "Ride", "Bills"],
            }
        )

        indices, fix_dict = obtain_index_to_fix(data, sample_config, ind_list=[0, 2])
        assert set(indices) == {0, 2}

    def test_empty_dataframe(self, sample_config):
        """Test handling of empty DataFrame"""
        data = pd.DataFrame(
            columns=[
                "type",
                "category",
                "suggested_categories",
                "transaction_date",
                "transaction_amount",
                "protistrana",
                "zpráva",
            ]
        )

        indices, fix_dict = obtain_index_to_fix(
            data,
            sample_config,
            incl_inbound=True,
            incl_conflicts=True,
            incl_missing_suggestions=True,
        )
        assert len(indices) == 0
        assert len(fix_dict) == 0

    def test_multiple_flags(self, sample_config):
        """Test combination of multiple flags"""
        data = pd.DataFrame(
            {
                "type": ["incoming", "outgoing", "outgoing"],
                "category": ["salary", "transport", "utilities"],
                "suggested_categories": ["salary", "", "bills"],
                "transaction_date": ["2025-01-01"] * 3,
                "transaction_amount": [1000, -30, -100],
                "protistrana": ["EMPLOYER", "TAXI", "UTILITY"],
                "zpráva": ["Salary", "Ride", "Bills"],
            }
        )

        indices, fix_dict = obtain_index_to_fix(
            data, sample_config, incl_inbound=True, incl_missing_suggestions=True
        )
        assert set(indices) == {0, 1}
