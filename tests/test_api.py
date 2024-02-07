import hashlib
import shutil
from pathlib import Path

import pandas as pd
import pytest
from peewee import SqliteDatabase

from tests import get_test_results_dir, test_data, test_externals, test_results
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
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.optimization.optimization_run import OptimizationRun
from vrtool.orm.models.optimization.optimization_selected_measure import (
    OptimizationSelectedMeasure,
)
from vrtool.orm.models.optimization.optimization_step import OptimizationStep
from vrtool.orm.models.optimization.optimization_step_result_mechanism import (
    OptimizationStepResultMechanism,
)
from vrtool.orm.models.optimization.optimization_step_result_section import (
    OptimizationStepResultSection,
)
from vrtool.orm.models.optimization.optimization_type import OptimizationType
from vrtool.orm.orm_controllers import (
    clear_assessment_results,
    clear_measure_results,
    clear_optimization_results,
    get_all_measure_results_with_supported_investment_years,
    open_database,
    vrtool_db,
)


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
        _found_dike_traject = DikeTrajectInfo.get()

        # Get a test `VrtoolConfig`.
        _vrtool_config = VrtoolConfig(traject=_found_dike_traject.traject_name)
        _vrtool_config.input_directory = _test_db_path.parent
        _vrtool_config.input_database_name = _test_db_path.name
        _vrtool_config.output_directory = test_results.joinpath(request.node.name)

        # Get a valid test `OptimizationRun`
        _optimization_run = OptimizationRun.get_by_id(1)
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
        acceptance_test_cases,
        indirect=True,
    )
    def test_run_step_assessment_given_valid_vrtool_config(
        self, valid_vrtool_config: VrtoolConfig
    ):
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
    def test_run_step_measures_given_valid_vrtool_config(
        self, valid_vrtool_config: VrtoolConfig
    ):
        # 1. Define test data.
        _validator = RunStepMeasuresValidator()

        clear_measure_results(valid_vrtool_config)
        _validator.validate_preconditions(valid_vrtool_config)

        # 2. Run test.
        run_step_measures(valid_vrtool_config)

        # 3. Verify expectations.
        _validator.validate_results(valid_vrtool_config)

    @pytest.mark.parametrize(
        "valid_vrtool_config",
        acceptance_test_cases,
        indirect=True,
    )
    def test_run_step_optimization_given_valid_vrtool_config(
        self, valid_vrtool_config: VrtoolConfig, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _new_optimization_name = "test_optimization_{}".format(
            request.node.callspec.id.replace(" ", "_").replace(",", "").lower()
        )

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
        _validator.validate_phased_out_csv_files(valid_vrtool_config)

    @pytest.mark.parametrize(
        "valid_vrtool_config",
        acceptance_test_cases[5:6],
        indirect=True,
    )
    def test_run_step_optimization_given_valid_vrtool_config_with_filtering(
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

        # Get the available measure results with supported investment years.
        # For this test, we only use measure results with odd ids.
        _measures_input = list(
            filter(
                lambda x: (x[0] % 2 != 0),
                get_all_measure_results_with_supported_investment_years(
                    valid_vrtool_config
                ),
            )
        )

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

    def _get_vrtool_config_test_copy(
        self, config_file: Path, test_name: str
    ) -> VrtoolConfig:
        """
        Gets a `VrtoolConfig` with a copy of the database to avoid version issues.
        """
        # Create a results directory (ignored by git)
        _test_results_directory = test_results.joinpath(test_name)
        if _test_results_directory.exists():
            shutil.rmtree(_test_results_directory)
        _test_results_directory.mkdir(parents=True)

        # Get the current configuration
        _vrtool_config = VrtoolConfig.from_json(config_file)

        # Create a db copy.
        _new_db_name = "test_{}.db".format(
            hashlib.shake_128(_test_results_directory.__bytes__()).hexdigest(4)
        )
        _new_db_path = _test_results_directory.joinpath(_new_db_name)
        if _new_db_path.exists():
            # Somehow it was not removed in the previous test run.
            _new_db_path.unlink(missing_ok=True)

        shutil.copy(_vrtool_config.input_database_path, _new_db_path)

        # Set new configuration values.
        _vrtool_config.input_directory = _test_results_directory
        _vrtool_config.input_database_name = _new_db_name
        _vrtool_config.output_directory = _test_results_directory.joinpath("output")
        _vrtool_config.output_directory.mkdir()

        return _vrtool_config

    def test_given_multiple_scenario_with_dstability_all_scenarios_are_run(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _multiple_scenarios_dir = test_data.joinpath(
            "test_stability_multiple_scenarios"
        )
        assert _multiple_scenarios_dir.exists()
        _vrtool_config = self._get_vrtool_config_test_copy(
            _multiple_scenarios_dir.joinpath("config.json"), request.node.name
        )

        assert not any(_vrtool_config.output_directory.glob("*"))

        # 2. Run test.
        ApiRunWorkflows(_vrtool_config).run_all()

        # 3. Verify expectations.
        assert any(_vrtool_config.output_directory.glob("*"))
