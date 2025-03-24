import shutil

import pandas as pd
import pytest

from tests import test_data, test_results
from vrtool.api import (
    ApiRunWorkflows,
    get_optimization_step_with_lowest_total_cost_table,
    get_valid_vrtool_config,
)
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.orm import models as orm
from vrtool.orm.models import SlopePart
from vrtool.orm.orm_controllers import open_database


class TestApi:
    def test_when_get_valid_vrtool_config_with_missing_config_file_raises_error(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _input_dir = test_results / request.node.name
        if not _input_dir.exists():
            _input_dir.mkdir(parents=True)

        assert _input_dir.exists()
        _config_file = _input_dir.joinpath("missing.json")
        assert not _config_file.exists()

        # 2. Run test.
        with pytest.raises(FileNotFoundError) as exception_error:
            get_valid_vrtool_config(_config_file)

        # 3. Verify expectations.
        assert str(exception_error.value) == "Config file {} not found.".format(
            _config_file
        )

    def test_when_get_valid_vrtool_config_with_valid_config_returns_vrtool_config(
        self,
    ):
        # 1. Define test data.
        _input_dir = test_data / "vrtool_config"
        _config_file = _input_dir.joinpath("custom_config.json")
        assert _config_file.exists()

        # 2. Run test.
        _vrtool_config = get_valid_vrtool_config(_config_file)

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
    
    def test_given_invalid_revetment_data_when_get_dike_traject_does_not_raise(
            self, request: pytest.FixtureRequest
        ):
        # 1. Define test data.

        # Create a test database with invalid revetment data (truncate SlopePart table).
        _test_db_name = "vrtool_input.db"
        _test_db_path = test_data.joinpath("31-1 two coastal sections", _test_db_name)
        assert _test_db_path.exists()

        _result_path = test_results.joinpath(request.node.name)
        if _result_path.exists():
            shutil.rmtree(_result_path)
        _result_path.mkdir(parents=True)

        shutil.copyfile(_test_db_path, _result_path.joinpath(_test_db_name))

        _opened_db = open_database(_result_path.joinpath(_test_db_name))
        SlopePart.truncate_table()
        _opened_db.close()

        # Create config without revetment.
        _vrtool_config = VrtoolConfig()
        _vrtool_config.traject = "31-1"
        _vrtool_config.excluded_mechanisms.append(MechanismEnum.REVETMENT)
        _vrtool_config.input_directory = _result_path
        _vrtool_config.input_database_name = _test_db_name

        # 2. Run test.
        ApiRunWorkflows.get_dike_traject(_vrtool_config)
