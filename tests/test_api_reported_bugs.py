import hashlib
import shutil
from pathlib import Path
from typing import Iterator

import pytest
from peewee import SqliteDatabase

from tests import (
    get_copy_of_reference_directory,
    get_vrtool_config_test_copy,
    test_externals,
)
from vrtool.api import ApiRunWorkflows
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.orm.orm_controllers import (
    clear_optimization_results,
    get_all_measure_results_with_supported_investment_years,
)
from vrtool.orm.orm_db import vrtool_db
from vrtool.run_workflows.optimization_workflow.results_optimization import (
    ResultsOptimization,
)


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
        _test_case_dir = get_copy_of_reference_directory(
            str(Path("reported_bugs", directory_name))
        )

        _vrtool_config = get_vrtool_config_test_copy(
            _test_case_dir.joinpath("config.json"), request.node.name
        )
        assert not any(_vrtool_config.output_directory.glob("*"))

        # 2. Run test.
        ApiRunWorkflows(_vrtool_config).run_all()

        # 3. Verify expectations.
        assert any(_vrtool_config.output_directory.glob("*"))

    @pytest.fixture(name="api_vrtool_config")
    def _get_api_vrtool_config_from_dir_and_db_str(
        self, request: pytest.FixtureRequest
    ) -> Iterator[VrtoolConfig]:
        _test_case_dict = request.param
        _test_input_directory = get_copy_of_reference_directory(
            str(Path("reported_bugs", _test_case_dict["test_dir"]))
        )
        assert _test_input_directory.exists()

        _test_results_directory = _test_input_directory.joinpath("results")

        # Define the VrtoolConfig
        _test_config = VrtoolConfig()
        _test_config.input_directory = _test_input_directory
        _test_config.output_directory = _test_results_directory
        _test_config.traject = _test_case_dict["traject_name"]
        _test_config.externals = test_externals
        _test_config.excluded_mechanisms = _test_case_dict["excluded_mechanisms"]

        # We need to create a copy of the database on the input directory.
        _test_db_name = "test_{}.db".format(
            hashlib.shake_128(_test_results_directory.__bytes__()).hexdigest(4)
        )
        _test_config.input_database_name = _test_db_name

        # Create a copy of the database to avoid parallelization runs locked databases.
        _reference_db_file = _test_input_directory.joinpath(_test_case_dict["db_name"])
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
        [
            pytest.param(
                (
                    dict(
                        test_dir="test_sections_without_selected_measures",
                        db_name="database_16-1_testcase_no_custom.db",
                        traject_name="16-1",
                        excluded_mechanisms=[
                            MechanismEnum.HYDRAULIC_STRUCTURES,
                            MechanismEnum.REVETMENT,
                        ],
                    )
                ),
                id="[VRTOOL-561] Sections without selected optimization measures.",
            )
        ],
        indirect=True,
    )
    def test_given_case_for_run_optimization_without_config_succeeds(
        self, api_vrtool_config: VrtoolConfig
    ):
        # 1. Define test data.
        assert isinstance(api_vrtool_config, VrtoolConfig)

        # 2. Run test.
        clear_optimization_results(api_vrtool_config)
        _ids_to_import = get_all_measure_results_with_supported_investment_years(
            api_vrtool_config
        )
        _results_optimization = ApiRunWorkflows(api_vrtool_config).run_optimization(
            "test_optimization", _ids_to_import
        )

        # 3. Verify expectations.
        assert isinstance(_results_optimization, ResultsOptimization)
        assert any(api_vrtool_config.output_directory.glob("*"))
