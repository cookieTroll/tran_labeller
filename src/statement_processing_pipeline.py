"""The module provides an encapsulation of the whole transaction processing pipeline"""

import logging
from enum import Enum
from typing import Optional

import pandas as pd

from src.data_transformer import parse_data
from src.format_validators import FullConfigSchema, validate_data
from src.loaders import read_csv, read_write_json
from src.postprocessor import (generate_aggregate_data, save_output,
                               update_categories)
from src.recommender import generate_suggestions, obtain_index_to_fix


class PipelineStep(Enum):
    LOAD = "load"
    PARSE = "parse"
    GENERATE_SUGGESTIONS = "generate_suggestions"
    SAVE_CONFLICTS = "save_conflicts"
    LOAD_CONFLICTS = "load_conflicts"
    UPLOAD_SUGGESTIONS = "upload_suggestions"
    UPDATE_CATEGORIES = "update_categories"
    SAVE = "save"

    @property
    def requires_input(self) -> bool:
        """Whether this step requires input data from previous step"""
        return self not in [PipelineStep.LOAD, PipelineStep.LOAD_CONFLICTS]

    @property
    def produces_output(self) -> bool:
        """Whether this step produces output data"""
        return self not in [PipelineStep.SAVE, PipelineStep.SAVE_CONFLICTS]

    @property
    def requires_fixdict(self) -> bool:
        """Whether this step requires a fix dictionary to be passed in"""
        return self in [PipelineStep.UPDATE_CATEGORIES]


FULL_PIPELINE = (
    PipelineStep.LOAD,
    PipelineStep.PARSE,
    PipelineStep.GENERATE_SUGGESTIONS,
    PipelineStep.SAVE_CONFLICTS,
    PipelineStep.LOAD_CONFLICTS,
    PipelineStep.UPDATE_CATEGORIES,
)


class StatementProcessor:
    """
    Manages the processing of financial statements through a pipeline of defined steps.

    The `StatementProcessor` class orchestrates the processing of financial statement
    data by executing a series of steps defined in the `pipeline`. These steps include
    loading data, parsing it, generating suggestions, handling conflicts, and updating
    categories. It maintains intermediate and final processing states, and provides
    methods to retrieve and save outputs. The configuration settings and input/output
    parameters are specified through the provided configuration schema.

    Attributes:
        config (FullConfigSchema): Configuration schema containing properties required for processing.
        pipeline (tuple[PipelineStep]): Sequence of pipeline steps to execute.
        logger: Logger instance for logging operations during processing.
        step_handlers (dict[PipelineStep, Callable]): Mapping of pipeline steps to their respective handlers.
        first_part (set[PipelineStep]): Subset of pipeline related to the initial stages of processing.
        second_part (list[PipelineStep]): Subset of pipeline related to conflict resolution and categorization.
        raw_data (Optional[pd.DataFrame]): Data loaded during the initial phase of processing.
        parsed_data (Optional[pd.DataFrame]): Data after being parsed and processed in the initial steps.
        final_data (Optional[pd.DataFrame]): Categorized data after all processing steps are completed.
        aggregate_data (Optional[pd.DataFrame]): Aggregated data grouped by months and categories.
    """
    def __init__(
        self,
        config: FullConfigSchema,
        logger,
        pipeline: tuple[PipelineStep] = FULL_PIPELINE,
    ):
        """
        Initialize an instance of the pipeline executor, setting up configuration,
        pipeline steps, logging mechanism, and data management structures for
        processing and categorizing data.

        Args:
            config: FullConfigSchema
                Configuration object conforming to the FullConfigSchema.
            logger: Any
                Logging object used to log pipeline activity and debugging information.
            pipeline: tuple[PipelineStep], optional
                Sequence of processing steps to execute within the pipeline. Defaults
                to FULL_PIPELINE.

        Attributes:
            config: FullConfigSchema
                Provides configuration for the pipeline process, including paths,
                processing options, and other necessary settings.
            pipeline: tuple[PipelineStep]
                Defines the ordered steps of the pipeline, encompassing data loading,
                parsing, suggestion generation, conflict handling, and categorization.
            logger: Any
                Logger instance responsible for capturing and recording messages or
                events throughout the pipeline execution.
            step_handlers: dict[PipelineStep, Callable]
                Maps each pipeline step to its corresponding function handler
                responsible for specific data processing logic.
            first_part: set[PipelineStep]
                Represents the subset of steps executed during the first segment of
                the pipeline, such as loading and initial data parsing.
            second_part: list[PipelineStep]
                Specifies the sequence of steps executed during the second segment of
                the pipeline, focusing on post-conflict data updates.
            raw_data: Any | None
                Contains the initially loaded data prior to any transformations. Set
                to None until the data loading step runs.
            parsed_data: Any | None
                Stores the output of the data parsing step, representing processed
                data ready for downstream steps. Initially None until parsing.
            final_data: Any | None
                Holds the fully processed and updated data with categorization. Updated
                once the categorization step is executed.
            aggregate_data: Any | None
                Aggregate representation of categorized data, typically grouped by month
                or similar dimensions, initialized to None until aggregation occurs.
        """
        self.config = config
        self.pipeline = pipeline
        self.logger = logger

        self.step_handlers = {
            PipelineStep.LOAD: self._load_data,
            PipelineStep.PARSE: self._parse_data,
            PipelineStep.GENERATE_SUGGESTIONS: self._generate_suggestions,
            PipelineStep.SAVE_CONFLICTS: self._save_conflicts,
            PipelineStep.LOAD_CONFLICTS: self._load_conflicts,
            PipelineStep.UPDATE_CATEGORIES: self._update_categories,
        }

        self.first_part = {
            PipelineStep.LOAD,
            PipelineStep.PARSE,
            PipelineStep.GENERATE_SUGGESTIONS,
            PipelineStep.SAVE_CONFLICTS,
        }
        self.second_part = [PipelineStep.LOAD_CONFLICTS, PipelineStep.UPDATE_CATEGORIES]

        self.raw_data = None  # inital loaded data
        self.parsed_data = None  # data after inital processing
        self.final_data = None  # data with updated categorization
        self.aggregate_data = None  # aggregation by month and categories

    def process_step(
        self,
        step: PipelineStep,
        data: Optional[pd.DataFrame] = None,
        fix_dict: Optional[dict[str, dict[str, str]]] = None,
    ) -> tuple[Optional[pd.DataFrame], None]:
        """
        Processes a pipeline step by executing its associated handler method and returns
        processed data and fix dictionary. Validates required inputs for the step.

        Parameters:
        step (PipelineStep): The pipeline step to be processed.
        data (Optional[pd.DataFrame]): Optionally, the input data required by the step for
            processing.
        fix_dict (Optional[dict[str, dict[str, str]]]): Optionally, the fix dictionary
            required for certain steps that depend on specific key-value mappings.

        Raises:
        ValueError: If the step requires input data but none is provided.
        ValueError: If the step requires a fix dictionary but none is provided.
        ValueError: If there is no handler defined for the given step.

        Returns:
        tuple[Optional[pd.DataFrame], None]: A tuple containing the processed data and
            an optional fix dictionary. Both values can be None.
        """
        if step.requires_input and data is None:
            raise ValueError(f"Step {step.name} requires input data")

        if step.requires_fixdict and fix_dict is None:
            raise ValueError(f"Step {step.name} requires a fix dictionary")

        handler = self.step_handlers.get(step)
        if not handler:
            raise ValueError(f"No handler for step {step.name}")
        logging.info(f"Running step %s", step.name)
        d, fd = handler(data, fix_dict)
        logging.info(f"Finished step %s", step.name)
        return d, fd

    def run_first_part(self, data: pd.DataFrame = None):
        """
        Processes the initial set of steps in a defined pipeline that belong to the first part.

        This function iterates over the steps included in the first part of the pipeline,
        processing each step in sequence using the `process_step` method. The processed
        data from each step is passed to the subsequent step. If no data is provided,
        a default value is used.

        Parameters:
            data (pd.DataFrame, optional): The input data to be processed. If omitted,
            the value defaults to None.

        Returns:
            pd.DataFrame: The output data after processing all the steps in the first part
            of the pipeline.
        """
        for step in [st for st in self.pipeline if st in self.first_part]:
            data, _ = self.process_step(step, data)
        return data

    def run_second_part(self, data=None, fix_dict=None):
        """
        Processes the second part of a pipeline.

        This method processes the second part of a pipeline by iterating through
        the defined steps that belong to the second part. If no data is provided,
        it uses the parsed data. Each step modifies the data and optionally updates
        the fix dictionary.

        Parameters:
            data (Any, optional): The input data to process. Defaults to None. If
            None, the method will use `self.parsed_data`.

            fix_dict (dict, optional): A dictionary to keep track of fixes or updates
            made during the processing steps. Defaults to None.

        Returns:
            tuple[Any, dict]: A tuple containing the processed data and the updated fix
            dictionary.
        """
        if not data is None:
            data = self.parsed_data

        for step in [st for st in self.pipeline if st in self.second_part]:
            data, fix_dict = self.process_step(step, data, fix_dict)

    def get_final_data(self) -> pd.DataFrame:
        return self.final_data

    def generate_aggregate_data(self) -> pd.DataFrame:

        o_f = self.config["output_config"]
        return generate_aggregate_data(
            self.final_data,
            o_f["payment_category"],
            o_f.get("categories_excl_from_agg", []),
            o_f["sort_by"],
            o_f["amount"],
        )

    def get_aggregate_data(self) -> pd.DataFrame:
        return self.aggregate_data

    def save_output(self) -> None:
        """
        Saves the output data to a specified file. This method delegates the actual
        saving logic to an external utility function `save_output` which performs
        the file operations. The output includes different processed data formats
        and aggregates as configured.

        Arguments:
            self: Refers to the instance of the class.

        Returns:
            None
        """
        save_output(
            self.config["file_name_output"],
            self.raw_data,
            self.parsed_data,
            self.final_data,
            self.aggregate_data,
            self.logger,
        )

    def _load_data(
        self,
        data: Optional[pd.DataFrame] = None,
        fix_dict: Optional[dict[str, dict[str, str]]] = None,
    ) -> tuple[Optional[pd.DataFrame], None]:
        """
        Loads and validates input data based on the provided configurations. The function
        ensures that the data is either loaded from a file or processed from a provided
        DataFrame. It also validates the format of the data against a predefined schema.

        Args:
            data: Optional; A pandas DataFrame object. If provided, data will be copied.
                  If not provided, data will be loaded from the input file specified
                  in the configuration.
            fix_dict: Optional; A dictionary containing rules for fixing data. This argument
                      is currently unused but preserved for potential future functionality.

        Returns:
            A tuple containing:
                - A pandas DataFrame with the loaded or copied data if successful,
                  or None if an error occurs during the process.
                - Always returns None as the second tuple element.

        Raises:
            Validation errors, if the input data does not conform to the defined schema.
        """
        if data is None:
            data = read_csv(
                self.config["file_name_input"],
                self.config["input_config"]["parsing"]["delimiter"],
                self.config["input_config"]["parsing"]["encoding"],
            )
        else:
            data = data.copy()

        validate_data(data, self.config["input_config"]["input_format"])
        self.raw_data = data
        return data, None

    def _parse_data(
        self, data: pd.DataFrame, fix_dict: Optional[dict[str, dict[str, str]]] = None
    ) -> tuple[Optional[pd.DataFrame], None]:
        """
        Parses the given DataFrame based on the specified configuration, optionally applying
        fixes from the provided dictionary.

        Args:
            data (pd.DataFrame): The data to be parsed.
            fix_dict (Optional[dict[str, dict[str, str]]]): A dictionary containing fixes
                to apply, organized as top-level keys mapping to nested dictionaries of
                keys and replacement values.

        Returns:
            tuple[Optional[pd.DataFrame], None]: A tuple where the first element is the
                parsed DataFrame or None if processing fails and the second element
                is always None.
        """
        parsed_data = parse_data(data=data, config=self.config, errors="raise")
        self.parsed_data = parsed_data
        return parsed_data, None

    def _generate_suggestions(
        self, data: pd.DataFrame, fix_dict: Optional[dict[str, dict[str, str]]] = None
    ) -> tuple[Optional[pd.DataFrame], None]:
        """
        Generate suggestions based on provided data and configuration settings.

        This method processes the input DataFrame, applying a suggestion generation
        function to selected rows based on configuration parameters. The resulting
        suggested values are stored in a new column within the DataFrame. An optional
        fix dictionary can be provided but is not explicitly utilized in this
        implementation.

        Parameters:
        data (pd.DataFrame): The input DataFrame containing rows to process.
        fix_dict (Optional[dict[str, dict[str, str]]]): A dictionary containing fixes or
            mappings for suggestions (optional).

        Returns:
        tuple[Optional[pd.DataFrame], None]: A tuple where the first element is the modified
            DataFrame with suggestions added and the second element is always None.
        """
        d = data.copy()
        d[self.config["output_config"]["generated_suggestions"]] = d.apply(
            lambda row: generate_suggestions(
                row,
                [
                    self.config["output_config"]["counterparty"],
                    self.config["output_config"]["message"],
                ],
                self.config["keywords_config"],
            ),
            axis=1,
        )
        return d, None

    def _save_conflicts(
        self, data: pd.DataFrame, fix_dict: Optional[dict[str, dict[str, str]]] = None
    ) -> tuple[Optional[pd.DataFrame], None]:
        """
        Handles the saving of non-matching records and updates a fix dictionary.

        This function identifies non-matching records based on specific criteria and updates the
        fix dictionary to assist in resolving data conflicts. The updated dictionary is saved to
        a file for later reference. The method returns the processed DataFrame and the updated
        fix dictionary.

        Parameters:
        data : pd.DataFrame
            The input DataFrame containing data to check for conflicts.

        fix_dict : Optional[dict[str, dict[str, str]]]
            Optional dictionary for mapping conflicts to their resolutions. If None, a new
            fix dictionary will be generated and updated.

        Returns:
        tuple[Optional[pd.DataFrame], None]
            A tuple containing the modified DataFrame and the updated fix dictionary, or None
            if processing does not yield results.
        """
        non_matching_ind, fix_dict = obtain_index_to_fix(
            data=data,
            config=self.config,
            incl_inbound=True,
            incl_missing_suggestions=True,
            incl_conflicts=True,
        )

        read_write_json(self.config["fix_file_name"], mode="w", payload=fix_dict)
        return data, fix_dict

    def _load_conflicts(
        self,
        data: pd.DataFrame = None,
        fix_dict: Optional[dict[str, dict[str, str]]] = None,
    ) -> tuple[Optional[pd.DataFrame], Optional[dict[str, str]]]:
        """
        Handles the loading of conflicts data and optional fixes dictionary, returning
        both the data frame and a dictionary result. This method can be used to handle
        the retrieval of data and fixes in a structured format, combining them for easy
        use in subsequent operations.

        Args:
            data (pd.DataFrame): Input data frame containing potential conflicts. Defaults to None.
            fix_dict (Optional[dict[str, dict[str, str]]]): Dictionary containing fixes
                in the form of nested mappings. Defaults to None.

        Returns:
            tuple[Optional[pd.DataFrame], Optional[dict[str, str]]]: A tuple where the
                first element is the data frame with conflicts or None, and the
                second element is the result of loading a JSON file as a dictionary.
        """
        return data, read_write_json(self.config["fix_file_name"], mode="r")

    def _update_categories(
        self, data: pd.DataFrame, fix_dict: Optional[dict[str, dict[str, str]]] = None
    ) -> tuple[Optional[pd.DataFrame], Optional[dict[str, str]]]:
        """
        Updates categories in the provided DataFrame based on the configuration and optional
        category fix dictionary.

        This method applies category updates onto the provided DataFrame using the rules
        defined in the `output_config["payment_category"]` of the instance's configuration.
        It optionally applies any corrections provided by the fix dictionary.

        Attributes:
            final_data (pd.DataFrame): Stores the updated DataFrame after category modification.

        Args:
            data (pd.DataFrame): The input DataFrame on which category updates are performed.
            fix_dict (Optional[dict[str, dict[str, str]]]): A dictionary containing optional
                corrections or modifications to categories. Keys are column names, and values
                are mappings of old category values to new ones.

        Returns:
            tuple[Optional[pd.DataFrame], Optional[dict[str, str]]]: The updated DataFrame with
            applied category changes is returned alongside a placeholder (None) for the second
            tuple value. The specific purpose for returning this second optional value is left
            unutilized in the current logic.
        """
        data = update_categories(
            data, self.config["output_config"]["payment_category"], fix_dict
        )
        self.final_data = data
        return data, None
