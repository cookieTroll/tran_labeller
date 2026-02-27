"""Tests for suggestion generation and processing functions."""

import logging
from unittest.mock import patch

import pandas as pd
import pytest

# Assuming the module is called suggestions.py
from src.recommender import generate_suggestions, obtain_index_to_fix


# Fixtures
@pytest.fixture
def sample_keywords():
    return {
        "groceries": ["food", "market", "grocery"],
        "transport": ["uber", "taxi", "bus"],
        "utilities": ["electric", "water", "gas"],
    }


@pytest.fixture
def sample_row():
    return pd.Series(
        {
            "description": "Grocery shopping at Food Market",
            "note": "Weekly groceries",
            "amount": 50.0,
        }
    )

@pytest.fixture
def sample_config():
    return {
        "output_format": {
            "payment_category": "category",
            "transaction_type": "type",
        },
        "output_config": {"transaction_type":"type",
                          "payment_category": "category",
                          "amount": "amount",
                          "generated_suggestions": "suggestions",
                          "date": {"col": "date"},
                          "counterparty": "counterparty",
                          "message": "message",
                          },
        "input_format": {"inbound_keyword": "incoming"},
        "input_config": {"input_format": {"inbound_keyword": "incoming"},}
    }


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "category": ["food", "transport", "unknown"],
            "type": ["outgoing", "incoming", "outgoing"],
            "suggestions": ["groceries", "transport", ""],
            "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "amount": [100, 200, 300],
            "counterparty": ["Market", "Uber", "Unknown"],
            "message": ["groceries", "ride", "misc"],
        }
    )


# Tests for generate_suggestions
def test_generate_suggestions_single_match(sample_row, sample_keywords):
    """Test when there's exactly one matching category."""
    result = generate_suggestions(
        sample_row, id_cols=["description", "note"], keywords=sample_keywords
    )
    assert result == "groceries"


def test_generate_suggestions_multiple_matches():
    """Test when multiple categories match."""
    row = pd.Series(
        {"description": "Uber ride to Food Market", "note": "Transport and groceries"}
    )
    result = generate_suggestions(
        row,
        id_cols=["description", "note"],
        keywords={"groceries": ["food"], "transport": ["uber"]},
    )
    assert isinstance(result, list)
    assert set(result) == {"groceries", "transport"}


def test_generate_suggestions_no_matches(sample_row):
    """Test when no categories match."""
    keywords = {"entertainment": ["movie", "theater"]}
    result = generate_suggestions(
        sample_row, id_cols=["description", "note"], keywords=keywords
    )
    assert isinstance(result, list)
    assert len(result) == 0


def test_generate_suggestions_empty_inputs(sample_row, sample_keywords):
    """Test error handling for empty inputs."""
    with pytest.raises(ValueError):
        generate_suggestions(sample_row, [], sample_keywords)

    with pytest.raises(ValueError):
        generate_suggestions(sample_row, ["description"], {})


def test_generate_suggestions_missing_column(sample_row):
    """Test error handling for missing columns."""
    with pytest.raises(KeyError):
        generate_suggestions(
            sample_row,
            id_cols=["nonexistent_column"],
            keywords={"category": ["keyword"]},
        )


# Tests for obtain_index_to_fix
def test_obtain_index_to_fix_inbound(sample_dataframe, sample_config):
    """Test filtering inbound transactions."""
    indices, fix_dict = obtain_index_to_fix(
        sample_dataframe, sample_config, incl_inbound=True
    )
    assert len(indices) == 1
    assert 1 in indices  # Index of the incoming transaction


def test_obtain_index_to_fix_conflicts(sample_dataframe, sample_config):
    """Test filtering conflicting categories."""
    indices, fix_dict = obtain_index_to_fix(
        sample_dataframe, sample_config, incl_conflicts=True
    )
    assert len(indices) == 2
    assert isinstance(fix_dict, dict)


def test_obtain_index_to_fix_missing_suggestions(sample_dataframe, sample_config):
    """Test filtering missing suggestions."""
    indices, fix_dict = obtain_index_to_fix(
        sample_dataframe, sample_config, incl_missing_suggestions=True
    )
    assert len(indices) == 1
    assert 2 in indices  # Index of row with empty suggestion


def test_obtain_index_to_fix_manual_list(sample_dataframe, sample_config):
    """Test manual list of indices."""
    manual_indices = [0, 2]
    indices, fix_dict = obtain_index_to_fix(
        sample_dataframe, sample_config, ind_list=manual_indices
    )
    assert set(indices) == set(manual_indices)
    assert len(fix_dict) == len(manual_indices)


@pytest.mark.parametrize(
    "inbound,conflicts,missing,expected_count",
    [
        (True, True, True, 3),  # All filters
        (True, False, False, 1),  # Only inbound
        (False, True, True, 2),  # Conflicts and missing
    ],
)
def test_obtain_index_to_fix_combined_filters(
    sample_dataframe, sample_config, inbound, conflicts, missing, expected_count
):
    """Test combinations of filters."""
    indices, _ = obtain_index_to_fix(
        sample_dataframe,
        sample_config,
        incl_inbound=inbound,
        incl_conflicts=conflicts,
        incl_missing_suggestions=missing,
    )
    assert len(indices) == expected_count


def test_obtain_index_to_fix_logging(sample_dataframe, sample_config):
    """Test logging functionality."""
    with patch.object(logging, "info") as mock_logging:
        obtain_index_to_fix(sample_dataframe, sample_config, incl_inbound=True)
        assert mock_logging.call_count == 2
