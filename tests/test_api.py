import hashlib
import shutil
from pathlib import Path

import pandas as pd
import pytest
from peewee import SqliteDatabase

from tests import (
    get_copy_of_reference_directory,
    get_test_results_dir,
    get_vrtool_config_test_copy,
    test_data,
    test_externals,
    test_results,
)
from tests.api_acceptance_cases import (
    AcceptanceTestCase,
    RunFullValidator,
    RunStepAssessmentValidator,
    RunStepMeasuresValidator,
    RunStepOptimizationValidator,
    vrtool_db_default_name,
)
from vrtool.api import (
    ApiRunWorkflows,
    get_optimization_step_with_lowest_total_cost_table,
    get_valid_vrtool_config,
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


def get_list_of_sections_for_measure_ids(
    valid_vrtool_config: VrtoolConfig,
    measure_ids: list[int],
) -> list[int]:
    """
    gets a list of the sectionIds for the measureIds provided in a list.

    Args:
        valid_vrtool_config (VrtoolConfig):
            Configuration contanining database connection details.
        measure_ids (list[int]): List of measure ids to get the sections for.

    Returns:
        list[int]: List of section ids.
    """
    with open_database(valid_vrtool_config.input_database_path).connection_context():
        _sections = (
            orm.MeasurePerSection.select(orm.MeasurePerSection.section_id)
            .join(orm.MeasureResult)
            .where(orm.MeasureResult.id.in_(measure_ids))
        )
    # return the sections for each MeasureResult
    return [x.section.get_id() for x in _sections]


def get_all_measure_results_of_specific_type(
    valid_vrtool_config: VrtoolConfig,
    measure_type: MeasureTypeEnum,
) -> list[int]:
    """
    Gets all available measure results (`MeasureResult`) from the database for a specific type of measure

    Args:
        valid_vrtool_config (VrtoolConfig):
            Configuration contanining database connection details.
        measure_type_name (str): Name of the measure type to get the results for

    Returns:
        list[tuple[int, int]]: List of measure result - investment year pairs.
    """
    with open_database(valid_vrtool_config.input_database_path).connection_context():
        # We do not want measures that have a year variable >0 initially, as then the interpolation is messed up.
        _supported_measures = (
            orm.MeasureResult.select()
            .join(orm.MeasurePerSection)
            .join(orm.Measure)
            .join(orm.MeasureType)
            .where(orm.Measure.year != 20)
            .where(orm.MeasureType.name == measure_type.get_old_name())
        )
    # get all ids of _supported_measures
    return [x.get_id() for x in _supported_measures]


class TestApi:
    def test_when_get_valid_vrtool_config_given_directory_without_json_raises_error(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _input_dir = test_results / request.node.name
        if not _input_dir.exists():
            _input_dir.mkdir(parents=True)

        assert _input_dir.exists()

        # 2. Run test.
        with pytest.raises(FileNotFoundError) as exception_error:
            get_valid_vrtool_config(_input_dir)

        # 3. Verify expectations.
        assert str(
            exception_error.value
        ) == "No json config file found in the model directory {}.".format(_input_dir)

    def test_when_get_valid_vrtool_config_given_directory_with_too_many_jsons_raises_error(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _input_dir = test_results / request.node.name
        if _input_dir.exists():
            shutil.rmtree(_input_dir)

        _input_dir.mkdir(parents=True)
        Path.joinpath(_input_dir, "first.json").touch()
        Path.joinpath(_input_dir, "second.json").touch()

        # 2. Run test.
        with pytest.raises(ValueError) as exception_error:
            get_valid_vrtool_config(_input_dir)

        # 3. Verify expectations.
        assert str(
            exception_error.value
        ) == "More than one json file found in the directory {}. Only one json at the root directory supported.".format(
            _input_dir
        )

    def test_when_get_valid_vrtool_config_given_directory_with_valid_config_returns_vrtool_config(
        self,
    ):
        # 1. Define test data.
        _input_dir = test_data / "vrtool_config"
        assert _input_dir.exists()

        # 2. Run test.
        _vrtool_config = get_valid_vrtool_config(_input_dir)

        # 3. Verify expectations.
        assert isinstance(_vrtool_config, VrtoolConfig)
        assert _vrtool_config.traject == "MyCustomTraject"
        assert _vrtool_config.input_directory == _input_dir
        assert _vrtool_config.output_directory == _input_dir / "results"

    @pytest.mark.parametrize(
        "vrtool_config",
        [
            pytest.param(None, id="No provided VrtoolConfig"),
            pytest.param(VrtoolConfig("just_a_db.db"), id="With dummy VrtoolConfig"),
        ],
    )
    def test_when_init_api_run_workflows_given_different_vrtool_config_succeeds(
        self, vrtool_config: VrtoolConfig
    ):
        _api_run_workflows = ApiRunWorkflows(vrtool_config)
        assert isinstance(_api_run_workflows, ApiRunWorkflows)

    def test_when_get_optimization_step_with_lowest_total_cost_table_given_db_with_results(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _test_db_path = test_data.joinpath("test_db", "vrtool_with_filtered_results.db")
        assert _test_db_path.exists()

        _opened_db = open_database(_test_db_path)
        _found_dike_traject = orm.DikeTrajectInfo.get()

        # Get a test `VrtoolConfig`.
        _vrtool_config = VrtoolConfig(traject=_found_dike_traject.traject_name)
        _vrtool_config.input_directory = _test_db_path.parent
        _vrtool_config.input_database_name = _test_db_path.name
        _vrtool_config.output_directory = test_results.joinpath(request.node.name)

        # Get a valid test `OptimizationRun`
        _optimization_run = orm.OptimizationRun.get_by_id(1)
        assert _optimization_run.optimization_type.name == "VEILIGHEIDSRENDEMENT"
        _opened_db.close()

        # 2. Run test.
        _result = get_optimization_step_with_lowest_total_cost_table(
            _vrtool_config, _optimization_run.get_id()
        )

        # 3. Verify expectations.
        assert isinstance(_result, tuple)
        # Optimization step with lowest total_lcc + total_risk
        assert _result[0] == 1
        assert isinstance(_result[1], pd.DataFrame)
        assert _result[2] == pytest.approx(2.59, rel=0.01)


acceptance_test_cases = list(
    map(
        lambda x: pytest.param(x, id=x.case_name),
        AcceptanceTestCase.get_cases(),
    )
)


@pytest.mark.slow
class TestApiRunWorkflowsAcceptance:
    @pytest.fixture
    def valid_vrtool_config(self, request: pytest.FixtureRequest) -> VrtoolConfig:
        _test_case: AcceptanceTestCase = request.param
        _test_input_directory = Path.joinpath(test_data, _test_case.model_directory)
        assert _test_input_directory.exists()

        _test_results_directory = get_test_results_dir(request).joinpath(
            _test_case.case_name
        )
        if _test_results_directory.exists():
            shutil.rmtree(_test_results_directory)
        _test_results_directory.mkdir(parents=True)

        # Define the VrtoolConfig
        _test_config = VrtoolConfig()
        _test_config.input_directory = _test_input_directory
        _test_config.output_directory = _test_results_directory
        _test_config.traject = _test_case.traject_name
        _test_config.excluded_mechanisms = _test_case.excluded_mechanisms
        _test_config.externals = test_externals

        # We need to create a copy of the database on the input directory.
        _test_db_name = "test_{}.db".format(
            hashlib.shake_128(_test_results_directory.__bytes__()).hexdigest(4)
        )
        _test_config.input_database_name = _test_db_name

        # Create a copy of the database to avoid parallelization runs locked databases.
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

    @pytest.mark.parametrize(
        "valid_vrtool_config",
        acceptance_test_cases[0:6],
        indirect=True,
    )
    def test_run_step_assessment(self, valid_vrtool_config: VrtoolConfig):
        # 1. Define test data.
        clear_assessment_results(valid_vrtool_config)
        _validator = RunStepAssessmentValidator()
        _validator.validate_preconditions(valid_vrtool_config)

        # 2. Run test.
        run_step_assessment(valid_vrtool_config)

        # 3. Verify expectations.
        assert valid_vrtool_config.output_directory.exists()
        assert any(valid_vrtool_config.output_directory.glob("*"))

        # 4. Validate exporting results is possible
        _validator.validate_results(valid_vrtool_config)

    @pytest.mark.parametrize(
        "valid_vrtool_config",
        acceptance_test_cases,
        indirect=True,
    )
    def test_run_step_measure(self, valid_vrtool_config: VrtoolConfig):
        # 1. Define test data.
        _validator = RunStepMeasuresValidator()

        clear_measure_results(valid_vrtool_config)
        _validator.validate_preconditions(valid_vrtool_config)

        # 2. Run test.
        run_step_measures(valid_vrtool_config)

        # 3. Verify expectations.
        _validator.validate_results(valid_vrtool_config)

    @pytest.mark.skip(reason="Only used for generating new reference databases.")
    @pytest.mark.parametrize(
        "valid_vrtool_config",
        acceptance_test_cases,
        indirect=True,
    )
    def test_run_step_optimization_acceptance_test_case(
        self, valid_vrtool_config: VrtoolConfig
    ):
        # 1. Define test data.
        _new_optimization_name = "Basisberekening"

        # We reuse existing measure results, but we clear the optimization ones.
        clear_optimization_results(valid_vrtool_config)

        _validator = RunStepOptimizationValidator()
        _validator.validate_preconditions(valid_vrtool_config)

        # We actually run using ALL the available measure results.
        _measures_input = get_all_measure_results_with_supported_investment_years(
            valid_vrtool_config
        )

        # 2. Run test.
        run_step_optimization(
            valid_vrtool_config, _new_optimization_name, _measures_input
        )

        # 3. Verify expectations.
        _validator.validate_results(valid_vrtool_config)

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
        "valid_vrtool_config",
        acceptance_test_cases,
        indirect=True,
    )
    def test_run_step_optimization_for_target_reliability(
        self, valid_vrtool_config: VrtoolConfig, request: pytest.FixtureRequest
    ):
        self._run_step_optimization_for_strategy(
            valid_vrtool_config, "Doorsnede-eisen", request
        )

    @pytest.mark.parametrize(
        "valid_vrtool_config",
        acceptance_test_cases,
        indirect=True,
    )
    def test_run_step_optimization_for_greedy_optimization(
        self, valid_vrtool_config: VrtoolConfig, request: pytest.FixtureRequest
    ):
        self._run_step_optimization_for_strategy(
            valid_vrtool_config, "Veiligheidsrendement", request
        )

    @pytest.mark.parametrize(
        "valid_vrtool_config",
        acceptance_test_cases[0:2],
        indirect=True,
    )
    def test_run_step_optimization_with_filtering(
        self, valid_vrtool_config: VrtoolConfig, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        # We reuse existing measure results, but we clear the optimization ones.
        _new_optimization_name = "test_filtered_optimization_{}".format(
            request.node.callspec.id.replace(" ", "_").replace(",", "").lower()
        )
        clear_optimization_results(valid_vrtool_config)

        _validator = RunStepOptimizationValidator("_filtered")
        _validator.validate_preconditions(valid_vrtool_config)

        # get the measure ids. We only consider soil reinforcement, revetment (if available) and VZG
        _measure_ids = [
            get_all_measure_results_of_specific_type(valid_vrtool_config, measure_type)
            for measure_type in [
                MeasureTypeEnum.SOIL_REINFORCEMENT,
                MeasureTypeEnum.REVETMENT,
                MeasureTypeEnum.VERTICAL_GEOTEXTILE,
            ]
        ]
        # flatten list of _measure_ids
        _measure_ids = [item for sublist in _measure_ids for item in sublist]
        # each measure should be executed in year 0 so generate tuples of id and 0
        _measures_input = [(measure_id, 0) for measure_id in _measure_ids]

        # 2. Run test.
        run_step_optimization(
            valid_vrtool_config, _new_optimization_name, _measures_input
        )

        # 3. Verify expectations.
        _validator.validate_results(valid_vrtool_config)

    @pytest.mark.parametrize(
        "valid_vrtool_config",
        acceptance_test_cases[0:2],
        indirect=True,
    )
    def test_run_step_optimization_with_adjusted_timing(
        self, valid_vrtool_config: VrtoolConfig, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        # We reuse existing measure results, but we clear the optimization ones.
        _new_optimization_name = "test_optimization_adjusted_timing_{}".format(
            request.node.callspec.id.replace(" ", "_").replace(",", "").lower()
        )
        clear_optimization_results(valid_vrtool_config)

        _validator = RunStepOptimizationValidator("_adjusted_timing")
        _validator.validate_preconditions(valid_vrtool_config)

        # get the measure ids. We only consider soil reinforcement, revetment (if available) and VZG
        _measure_ids = [
            get_all_measure_results_of_specific_type(valid_vrtool_config, measure_type)
            for measure_type in [
                MeasureTypeEnum.SOIL_REINFORCEMENT,
                MeasureTypeEnum.REVETMENT,
                MeasureTypeEnum.VERTICAL_GEOTEXTILE,
            ]
        ]
        # flatten list of _measure_ids and sort
        _measure_ids = sorted([item for sublist in _measure_ids for item in sublist])

        # get the sections for each measure
        _sections_per_measure_id = get_list_of_sections_for_measure_ids(
            valid_vrtool_config, _measure_ids
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
            valid_vrtool_config, _new_optimization_name, _measures_input
        )

        # 3. Verify expectations.
        _validator.validate_results(valid_vrtool_config)

    @pytest.mark.parametrize(
        "valid_vrtool_config",
        acceptance_test_cases[0:1],
        indirect=True,
    )
    def test_run_full_given_simple_test_case(self, valid_vrtool_config: VrtoolConfig):
        """
        This test so far only checks the output values after optimization.
        """
        # 1. Define test data.
        _validator = RunFullValidator()
        _validator.validate_preconditions(valid_vrtool_config)

        # 2. Run test.
        run_full(valid_vrtool_config)

        # 3. Verify final expectations.
        _validator.validate_results(valid_vrtool_config)


@pytest.mark.slow
class TestApiReportedBugs:
    @pytest.mark.parametrize(
        "directory_name",
        [
            pytest.param(
                "test_stability_multiple_scenarios",
                id="Stability case with multiple scenarios [VRTOOL-340]",
            ),
            pytest.param(
                "test_revetment_step_transition_level",
                id="Revetment case with many transition levels [VRTOOL-330]",
            ),
        ],
    )
    def test_given_case_from_reported_bug_run_all_succeeds(
        self, directory_name: str, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _test_case_dir = get_copy_of_reference_directory(directory_name)

        _vrtool_config = get_vrtool_config_test_copy(
            _test_case_dir.joinpath("config.json"), request.node.name
        )
        assert not any(_vrtool_config.output_directory.glob("*"))

        # 2. Run test.
        ApiRunWorkflows(_vrtool_config).run_all()

        # 3. Verify expectations.
        assert any(_vrtool_config.output_directory.glob("*"))
