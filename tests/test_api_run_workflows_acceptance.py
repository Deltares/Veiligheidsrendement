import hashlib
import shutil
from pathlib import Path
from typing import Iterator
import copy
import pytest
from peewee import SqliteDatabase

from tests import get_clean_test_results_dir, test_data, test_externals
from tests.api_acceptance_cases.acceptance_test_case import (
    AcceptanceTestCase,
    vrtool_db_default_name,
)
from tests.api_acceptance_cases.run_full_validator import RunFullValidator
from tests.api_acceptance_cases.run_step_assessment_validator import (
    RunStepAssessmentValidator,
)
from tests.api_acceptance_cases.run_step_measures_validator import (
    RunStepMeasuresValidator,
)
from tests.api_acceptance_cases.run_step_optimization_validator import (
    RunStepOptimizationValidator,
)
from vrtool.api import (
    run_full,
    run_step_assessment,
    run_step_measures,
    run_step_optimization,
)
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.orm import models as orm
from vrtool.orm.orm_controllers import (
    clear_assessment_results,
    clear_measure_results,
    clear_optimization_results,
    get_all_measure_results_with_supported_investment_years,
    open_database,
    vrtool_db,
)


acceptance_test_cases = list(
    map(
        lambda x: pytest.param(x, id=x.case_name),
        AcceptanceTestCase.get_cases(),
    )
)

acceptance_test_cases_with_optimization_adjusted_timing = list(
    _case for _case in acceptance_test_cases if _case.values[0].run_adjusted_timing
)

acceptance_test_cases_with_optimization_filtering = list(
    _case for _case in acceptance_test_cases if _case.values[0].run_filtered
)

acceptance_test_cases_with_optimization_run_settings = list(
    map(
        lambda x: pytest.param(x, x.run_adjusted_timing, x.run_filtered, id=x.case_name),
        AcceptanceTestCase.get_cases(),
    )
)


def _get_list_of_sections_for_measure_ids(
    vrtool_config: VrtoolConfig,
    measure_ids: list[int],
) -> list[int]:
    """
    gets a list of the sectionIds for the measureIds provided in a list.

    Args:
        vrtool_config (VrtoolConfig):
            Configuration contanining database connection details.
        measure_ids (list[int]): List of measure ids to get the sections for.

    Returns:
        list[int]: List of section ids.
    """
    with open_database(vrtool_config.input_database_path).connection_context():
        _sections = (
            orm.MeasurePerSection.select(orm.MeasurePerSection.section_id)
            .join(orm.MeasureResult)
            .where(orm.MeasureResult.id.in_(measure_ids))
        )
    # return the sections for each MeasureResult
    return [x.section.get_id() for x in _sections]


def _get_all_measure_results_of_specific_type(
    vrtool_config: VrtoolConfig,
    measure_type: MeasureTypeEnum,
) -> list[int]:
    """
    Gets all available measure results (`MeasureResult`) from the database for a specific type of measure

    Args:
        vrtool_config (VrtoolConfig):
            Configuration contanining database connection details.
        measure_type (MeasureTypeEnum): The measure type to get the results for

    Returns:
        list[tuple[int, int]]: List of measure result - investment year pairs.
    """
    with open_database(vrtool_config.input_database_path).connection_context():
        # We do not want measures that have a year variable >0 initially, as then the interpolation is messed up.
        _supported_measures = (
            orm.MeasureResult.select()
            .join(orm.MeasurePerSection)
            .join(orm.Measure)
            .join(orm.MeasureType)
            .where(orm.MeasureType.name == measure_type.legacy_name)
        )
    # get all ids of _supported_measures
    return [x.get_id() for x in _supported_measures]


@pytest.mark.slow
class TestApiRunWorkflowsAcceptance:
    @pytest.fixture(name="api_vrtool_config")
    def _get_api_vrtool_config(
        self, request: pytest.FixtureRequest
    ) -> Iterator[VrtoolConfig]:
        _test_case: AcceptanceTestCase = request.param
        _test_input_directory = Path.joinpath(test_data, _test_case.model_directory)
        assert _test_input_directory.exists()

        _test_results_directory = get_clean_test_results_dir(request)

        # Define the VrtoolConfig
        _test_config = VrtoolConfig()
        _test_config.input_directory = _test_input_directory
        _test_config.output_directory = _test_results_directory
        _test_config.traject = _test_case.traject_name
        _test_config.excluded_mechanisms = _test_case.excluded_mechanisms
        _test_config.externals = test_externals

        # We need to create a copy of the database on the input directory.
        _test_db_name = "vrtool_input_{}.db".format(
            hashlib.shake_128(_test_results_directory.__bytes__()).hexdigest(4)
        )
        _test_config.input_database_name = _test_db_name

        # Create a copy of the database to avoid parallellization runs locked databases.
        _reference_db_file = _test_input_directory.joinpath(vrtool_db_default_name)
        assert _reference_db_file.exists(), "No database found at {}.".format(
            _reference_db_file
        )

        if _test_config.input_database_path.exists():
            # Somehow it was not removed in the previous test run.
            _test_config.input_database_path.unlink(missing_ok=True)

        shutil.copy(_reference_db_file, _test_config.input_database_path)
        assert (
            _test_config.input_database_path.exists()
        ), "No database found at {}.".format(_reference_db_file)

        yield _test_config

        # Make sure that the database connection will be closed even if the test fails.
        if isinstance(vrtool_db, SqliteDatabase) and not vrtool_db.is_closed():
            vrtool_db.close()

        # Copy the test database to the results directory so it can be manually reviewed.
        if _test_config.input_database_path.exists():
            _results_db_name = _test_config.output_directory.joinpath(
                "vrtool_result.db"
            )
            shutil.move(_test_config.input_database_path, _results_db_name)

            # Copy the postprocessing report if it exists.
            # For now it assumes it's created at the same level as the results
            _report = _test_config.input_database_path.parent.joinpath(
                "postprocessing_report_" + _test_config.input_database_path.stem
            )
            if _report.exists():
                shutil.move(
                    _report,
                    _test_config.output_directory.joinpath("postprocessing_report"),
                )

    @pytest.mark.parametrize(
        "api_vrtool_config",
        acceptance_test_cases[0:6],
        indirect=True,
    )
    def test_run_step_assessment(self, api_vrtool_config: VrtoolConfig):
        # 1. Define test data.
        clear_assessment_results(api_vrtool_config)
        _validator = RunStepAssessmentValidator()
        _validator.validate_preconditions(api_vrtool_config)

        # 2. Run test.
        run_step_assessment(api_vrtool_config)

        # 3. Verify expectations.
        assert api_vrtool_config.output_directory.exists()
        assert any(api_vrtool_config.output_directory.glob("*"))

        # 4. Validate exporting results is possible
        _validator.validate_results(api_vrtool_config)

    @pytest.mark.parametrize(
        "api_vrtool_config",
        acceptance_test_cases,
        indirect=True,
    )
    def test_run_step_measure(self, api_vrtool_config: VrtoolConfig):
        # 1. Define test data.
        _validator = RunStepMeasuresValidator()

        clear_measure_results(api_vrtool_config)
        _validator.validate_preconditions(api_vrtool_config)

        # 2. Run test.
        run_step_measures(api_vrtool_config)

        # 3. Verify expectations.
        _validator.validate_results(api_vrtool_config)

    def _run_step_optimization_for_strategy(
        self,
        vrtool_config: VrtoolConfig,
        design_method: str,
        request: pytest.FixtureRequest,
    ):
        # 1. Define test data.
        _new_optimization_name = "test_optimization_{}".format(
            request.node.callspec.id.replace(" ", "_").replace(",", "").lower()
        )
        # Only the selected design method for this case:
        vrtool_config.design_methods = [design_method]

        # We reuse existing measure results, but we clear the optimization ones.
        clear_optimization_results(vrtool_config)
        _validator = RunStepOptimizationValidator()
        _validator.validate_preconditions(vrtool_config)
        # We actually run using ALL the available measure results.
        _measures_input = get_all_measure_results_with_supported_investment_years(
            vrtool_config
        )

        # 2. Run test.
        run_step_optimization(vrtool_config, _new_optimization_name, _measures_input)

        # 3. Verify expectations.
        _validator.validate_results(vrtool_config)

    @pytest.mark.parametrize(
        "api_vrtool_config",
        acceptance_test_cases,
        indirect=True,
    )
    def test_run_step_optimization_for_target_reliability(
        self, api_vrtool_config: VrtoolConfig, request: pytest.FixtureRequest
    ):
        self._run_step_optimization_for_strategy(
            api_vrtool_config, "Doorsnede-eisen", request
        )

    @pytest.mark.parametrize(
        "api_vrtool_config",
        acceptance_test_cases,
        indirect=True,
    )
    def test_run_step_optimization_for_greedy_optimization(
        self, api_vrtool_config: VrtoolConfig, request: pytest.FixtureRequest
    ):
        self._run_step_optimization_for_strategy(
            api_vrtool_config, "Veiligheidsrendement", request
        )
    
    def _run_step_optimization_with_filtering(
            self,
            vrtool_config: VrtoolConfig,
            validator: RunStepOptimizationValidator,
            request: pytest.FixtureRequest):
        
        # Run the optimization step with the filtered measures.

        # 1. Define test data.
        # We reuse existing measure results, but we clear the optimization ones.
        _new_optimization_name = "test_filtered_optimization_{}".format(
            request.node.callspec.id.replace(" ", "_").replace(",", "").lower()
        )
        clear_optimization_results(vrtool_config)

        validator.validate_preconditions(vrtool_config)

        # get the measure ids. We only consider soil reinforcement, revetment (if available) and VZG
        _measure_ids = [
            _get_all_measure_results_of_specific_type(vrtool_config, measure_type)
            for measure_type in [
                MeasureTypeEnum.SOIL_REINFORCEMENT,
                MeasureTypeEnum.REVETMENT,
                MeasureTypeEnum.VERTICAL_PIPING_SOLUTION,
            ]
        ]
        # flatten list of _measure_ids
        _measure_ids = [item for sublist in _measure_ids for item in sublist]
        # each measure should be executed in year 0 so generate tuples of id and 0
        _measures_input = [(measure_id, 0) for measure_id in _measure_ids]

        # 2. Run test.
        run_step_optimization(
            vrtool_config, _new_optimization_name, _measures_input
        )

    @pytest.mark.parametrize(
        "api_vrtool_config",
        acceptance_test_cases_with_optimization_filtering,
        indirect=True,
    )
    def test_run_step_optimization_with_filtering(
        self, api_vrtool_config: VrtoolConfig, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _validator = RunStepOptimizationValidator("_filtered")

        # 2. Run test.
        self._run_step_optimization_with_filtering(api_vrtool_config, _validator, request)

        # 3. Verify expectations.
        _validator.validate_results(api_vrtool_config)
    
    def _run_step_optimization_with_adjusted_timing(
            self,
            vrtool_config: VrtoolConfig,
            validator: RunStepOptimizationValidator,
            request: pytest.FixtureRequest):
        
        # Run the optimization step with the adjusted timing.

        # 1. Define test data.
        # We reuse existing measure results, but we clear the optimization ones.
        _new_optimization_name = "test_adjusted_timing_optimization_{}".format(
            request.node.callspec.id.replace(" ", "_").replace(",", "").lower()
        )
        clear_optimization_results(vrtool_config)

        validator.validate_preconditions(vrtool_config)

        # get the measure ids. We only consider soil reinforcement, revetment (if available) and VZG
        _measure_ids = [
            _get_all_measure_results_of_specific_type(vrtool_config, measure_type)
            for measure_type in [
                MeasureTypeEnum.SOIL_REINFORCEMENT,
                MeasureTypeEnum.REVETMENT,
                MeasureTypeEnum.VERTICAL_PIPING_SOLUTION,
            ]
        ]
        # flatten list of _measure_ids ans sort
        _measure_ids = sorted([item for sublist in _measure_ids for item in sublist])

        # get the sections for each measure
        _sections_per_measure_id = _get_list_of_sections_for_measure_ids(
            vrtool_config, _measure_ids
        )

        # list of investment years (note that first value is not used)
        _investment_years_per_section = [3, 3, 23, 13]
        # each measure should be executed in year 0 so generate tuples of id and 0
        _measures_input = [
            (measure_id, _investment_years_per_section[_sections_per_measure_id[count]])
            for count, measure_id in enumerate(_measure_ids)
        ]

        # 2. Run test.
        run_step_optimization(
            vrtool_config, _new_optimization_name, _measures_input
        )

    @pytest.mark.parametrize(
        "api_vrtool_config",
        acceptance_test_cases_with_optimization_adjusted_timing,
        indirect=True,
    )
    def test_run_step_optimization_with_adjusted_timing(
        self, api_vrtool_config: VrtoolConfig, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        # The post validation is not reliable as the way to calculate lcc's does not
        # seem to be accurate when dealing with filtered timings.
        _validator = RunStepOptimizationValidator(
            "_adjusted_timing", False
        )

        # 2. Run test.
        self._run_step_optimization_with_adjusted_timing(api_vrtool_config, _validator, request)

        # 3. Verify expectations.
        _validator.validate_results(api_vrtool_config)

    @pytest.mark.parametrize(
        "api_vrtool_config",
        acceptance_test_cases[0:1],
        indirect=True,
    )
    def test_run_full_given_simple_test_case(self, api_vrtool_config: VrtoolConfig):
        """
        This test so far only checks the output values after optimization.
        """
        # 1. Define test data.
        _validator = RunFullValidator()
        _validator.validate_preconditions(api_vrtool_config)

        # 2. Run test.
        run_full(api_vrtool_config)

        # 3. Verify final expectations.
        _validator.validate_results(api_vrtool_config)

    @pytest.mark.skip(
        reason="Only used for generating new reference databases. Run with the --no-skip option."
    )
    @pytest.mark.parametrize(
        "api_vrtool_config, run_adjusted_timing, run_filtered",
        acceptance_test_cases_with_optimization_run_settings,
        indirect=["api_vrtool_config"],
    )
    @pytest.mark.regenerate_test_db
    def test_run_full_to_generate_results(self, api_vrtool_config: VrtoolConfig, run_adjusted_timing: bool, run_filtered: bool, request: pytest.FixtureRequest):
        """
        This test is only meant to regenerate the references for the large test cases.
        You can run this test from command line with:
        `pytest -m "regenerate_test_db" --no-skips`

        For cases that are configured to run with filtering and/or adjusted timing,
        an additional optimization run is done with the measure results of the full run of the main case.
        This generates an additional database with the suffix "_filtered" or "_adjusted_timing".
        """

        # Run the main case
        run_full(api_vrtool_config)

        def _get_copied_vrtool_config(suffix: str) -> VrtoolConfig:
            # Copy the config.
            _copied_vrtool_config = copy.deepcopy(api_vrtool_config)

            # Get copy of the base input database.
            _base_db_path = api_vrtool_config.input_database_path
            _copied_vrtool_config.input_database_name =  api_vrtool_config.input_database_name.replace("vrtool_input", f"vrtool_input{suffix}")
            if _copied_vrtool_config.input_database_path.exists():
                _copied_vrtool_config.input_database_path.unlink(missing_ok=True)
            shutil.copy(_base_db_path, _copied_vrtool_config.input_database_path)

            return _copied_vrtool_config

        def _copy_results(vrtool_config: VrtoolConfig, suffix: str) -> None:
            # Copy the results.
            if vrtool_config.input_database_path.exists():
                _results_db_name = vrtool_config.output_directory.joinpath(
                    f"vrtool_result{suffix}.db"
                )
                shutil.move(vrtool_config.input_database_path, _results_db_name)

        # Run the optimization step with filtered measures.
        if run_filtered:
            _suffix = "_filtered"

            # Copy the config.
            _filtered_vrtool_config = _get_copied_vrtool_config(_suffix)

            _validator = RunStepOptimizationValidator(_suffix)
            self._run_step_optimization_with_filtering(_filtered_vrtool_config, _validator, request)

            # Copy the results
            _copy_results(_filtered_vrtool_config, _suffix)

        # Run the optimization step with adjusted timing.
        if run_adjusted_timing:
            _suffix = "_adjusted_timing"

             # Copy the config.
            _filtered_vrtool_config = _get_copied_vrtool_config(_suffix)

            _validator = RunStepOptimizationValidator(_suffix, False)
            self._run_step_optimization_with_adjusted_timing(_filtered_vrtool_config, _validator, request)

            # Copy the results
            _copy_results(_filtered_vrtool_config, _suffix)
