import logging
from enum import Enum
from io import StringIO
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.statement_processing_pipeline import (
    FULL_PIPELINE,
    PipelineStep,
    StatementProcessor,
)


@pytest.fixture
def sample_config():
    return {
        "file_name_input": "input.csv",
        "file_name_output": "output.xlsx",
        "fix_file_name": "fixes.json",
        "input_config": {
            "parsing": {"delimiter": ",", "encoding": "utf-8"},
            "input_format": {"required_columns": ["date", "amount", "description"]},
        },
        "output_config": {
            "payment_category": "category",
            "generated_suggestions": "suggestions",
            "counterparty": "counterparty",
            "message": "description",
            "amount": "amount",
            "sort_by": ["date"],
            "categories_excl_from_agg": ["transfer"],
        },
        "keywords_config": {
            "food": ["restaurant", "grocery"],
            "transport": ["uber", "taxi"],
        },
    }


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "amount": [100.0, 200.0],
            "description": ["Restaurant Payment", "Uber Ride"],
            "category": ["food", "transport"],
            "counterparty": ["Restaurant ABC", "Uber"],
        }
    )


@pytest.fixture
def logger():
    return logging.getLogger("test_logger")


class TestStatementProcessor:

    @pytest.fixture
    def processor(self, sample_config, logger):
        return StatementProcessor(sample_config, logger)

    def test_initialization(self, processor, sample_config):
        assert processor.config == sample_config
        assert processor.pipeline == FULL_PIPELINE
        assert processor.raw_data is None
        assert processor.parsed_data is None
        assert processor.final_data is None
        assert processor.aggregate_data is None

    def test_process_step_without_required_input(self, processor):
        with pytest.raises(ValueError, match="requires input data"):
            processor.process_step(PipelineStep.PARSE)

    def test_process_step_without_required_fixdict(self, processor, sample_data):
        with pytest.raises(ValueError, match="requires a fix dictionary"):
            processor.process_step(PipelineStep.UPDATE_CATEGORIES, sample_data)

    @patch("src.statement_processing_pipeline.read_csv")
    def test_load_data(self, mock_read_csv, processor, sample_data):
        mock_read_csv.return_value = sample_data
        data, _ = processor.process_step(PipelineStep.LOAD)
        assert isinstance(data, pd.DataFrame)
        assert processor.raw_data is not None
        assert len(data) == len(sample_data)

    @patch("src.statement_processing_pipeline.parse_data")
    def test_parse_data(self, mock_parse_data, processor, sample_data):
        mock_parse_data.return_value = sample_data
        data, _ = processor.process_step(PipelineStep.PARSE, sample_data)
        assert isinstance(data, pd.DataFrame)
        assert processor.parsed_data is None  # Should only be set in run_first_part

    def test_generate_suggestions(self, processor, sample_data):
        data, _ = processor.process_step(PipelineStep.GENERATE_SUGGESTIONS, sample_data)
        assert isinstance(data, pd.DataFrame)
        assert (
            processor.config["output_config"]["generated_suggestions"] in data.columns
        )

    @patch("src.statement_processing_pipeline.read_write_json")
    def test_save_conflicts(self, mock_read_write_json, processor, sample_data):
        mock_read_write_json.return_value = {}
        data, fix_dict = processor.process_step(
            PipelineStep.SAVE_CONFLICTS, sample_data
        )
        assert isinstance(data, pd.DataFrame)
        mock_read_write_json.assert_called_once()

    @patch("src.statement_processing_pipeline.read_write_json")
    def test_load_conflicts(self, mock_read_write_json, processor, sample_data):
        mock_read_write_json.return_value = {"1": {"category": "food"}}
        data, fix_dict = processor.process_step(
            PipelineStep.LOAD_CONFLICTS, sample_data
        )
        assert isinstance(fix_dict, dict)
        assert isinstance(data, pd.DataFrame)

    @patch("src.statement_processing_pipeline.update_categories")
    def test_update_categories(self, mock_update_categories, processor, sample_data):
        fix_dict = {"1": {"category": "food"}}
        mock_update_categories.return_value = sample_data
        data, _ = processor.process_step(
            PipelineStep.UPDATE_CATEGORIES, sample_data, fix_dict
        )
        assert isinstance(data, pd.DataFrame)
        assert processor.final_data is None  # Should only be set in run_second_part

    def test_run_first_part(self, processor, sample_data):
        with patch.multiple(
            processor,
            _load_data=Mock(return_value=(sample_data, None)),
            _parse_data=Mock(return_value=(sample_data, None)),
            _generate_suggestions=Mock(return_value=(sample_data, None)),
            _save_conflicts=Mock(return_value=(sample_data, None)),
        ):
            result = processor.run_first_part(sample_data)
            assert isinstance(result, pd.DataFrame)

    def test_run_second_part(self, processor, sample_data):
        fix_dict = {"1": {"category": "food"}}
        with patch.multiple(
            processor,
            _load_conflicts=Mock(return_value=(sample_data, fix_dict)),
            _update_categories=Mock(return_value=(sample_data, None)),
        ):
            processor.parsed_data = sample_data
            processor.run_second_part(sample_data, fix_dict)
            assert isinstance(processor.parsed_data, pd.DataFrame)

    @patch("src.statement_processing_pipeline.generate_aggregate_data")
    def test_generate_aggregate_data(
        self, mock_generate_aggregate, processor, sample_data
    ):
        processor.final_data = sample_data
        mock_generate_aggregate.return_value = sample_data
        result = processor.generate_aggregate_data()
        assert isinstance(result, pd.DataFrame)
        mock_generate_aggregate.assert_called_once()

    def test_get_final_data(self, processor, sample_data):
        processor.final_data = sample_data
        result = processor.get_final_data()
        assert isinstance(result, pd.DataFrame)
        assert result.equals(sample_data)

    def test_get_aggregate_data(self, processor, sample_data):
        processor.aggregate_data = sample_data
        result = processor.get_aggregate_data()
        assert isinstance(result, pd.DataFrame)
        assert result.equals(sample_data)

    @patch("src.statement_processing_pipeline.save_output")
    def test_save_output(self, mock_save_output, processor, sample_data):
        processor.raw_data = sample_data
        processor.parsed_data = sample_data
        processor.final_data = sample_data
        processor.aggregate_data = sample_data
        processor.save_output()
        mock_save_output.assert_called_once_with(
            processor.config["file_name_output"],
            sample_data,
            sample_data,
            sample_data,
            sample_data,
            processor.logger,
        )


class TestPipelineStep:

    def test_requires_input(self):
        assert not PipelineStep.LOAD.requires_input
        assert not PipelineStep.LOAD_CONFLICTS.requires_input
        assert PipelineStep.PARSE.requires_input
        assert PipelineStep.GENERATE_SUGGESTIONS.requires_input
        assert PipelineStep.UPDATE_CATEGORIES.requires_input

    def test_produces_output(self):
        assert PipelineStep.LOAD.produces_output
        assert PipelineStep.PARSE.produces_output
        assert not PipelineStep.SAVE.produces_output
        assert not PipelineStep.SAVE_CONFLICTS.produces_output

    def test_requires_fixdict(self):
        assert not PipelineStep.LOAD.requires_fixdict
        assert not PipelineStep.PARSE.requires_fixdict
        assert PipelineStep.UPDATE_CATEGORIES.requires_fixdict
