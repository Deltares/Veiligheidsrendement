import shutil

import pytest

from tests import test_results
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.run_workflows.safety_workflow.run_safety_assessment import (
    RunSafetyAssessment,
)
from vrtool.run_workflows.vrtool_run_protocol import VrToolRunProtocol


class TestRunSafetyAssessment:
    def test_init_with_valid_args(self):
        # 1. Define test data
        _vr_config = VrtoolConfig()
        _dike_traject = DikeTraject()

        # 2. Run test.
        _assessment = RunSafetyAssessment(_vr_config, _dike_traject)

        # 3. Verify expectations.
        assert isinstance(_assessment, RunSafetyAssessment)
        assert isinstance(_assessment, VrToolRunProtocol)

    def test_init_with_invalid_vr_config(self):
        # 1. Run test
        with pytest.raises(ValueError) as exception_error:
            RunSafetyAssessment("paradise", "city")

        # 2. Verify expectations
        assert str(exception_error.value) == "Expected instance of a VrtoolConfig."

    def test_init_with_invalid_selected_traject(self):
        # 1. Define test data.
        _vr_config = VrtoolConfig()

        # 1. Run test
        with pytest.raises(ValueError) as exception_error:
            RunSafetyAssessment(_vr_config, "city")

        # 2. Verify expectations
        assert str(exception_error.value) == "Expected instance of a DikeTraject."

    def test_get_valid_output_dir_creates_missing_directories(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data
        _vr_config = VrtoolConfig()
        _dike_traject = DikeTraject()
        _assessment = RunSafetyAssessment(_vr_config, _dike_traject)
        _assessment.vr_config = VrtoolConfig()
        _assessment.vr_config.output_directory = test_results.joinpath(
            request.node.name
        )
        if _assessment.vr_config.output_directory.exists():
            shutil.rmtree(_assessment.vr_config.output_directory)

        # 2. Run test
        _assessment._get_valid_output_dir(["just", "another", "nested", "dir"])

        # 3. Verify expectations
        assert _assessment.vr_config.output_directory.exists()
        assert (
            _assessment.vr_config.output_directory
            / "just"
            / "another"
            / "nested"
            / "dir"
        ).exists()

    def test_given_invalid_vrtool_config_when_initialize_raises(
        self, invalid_vrtool_config_fixture: tuple[VrtoolConfig, str]
    ):
        # 1. Define test data.
        _invalid_vrtool_config, _expected_error_mssg = invalid_vrtool_config_fixture
        assert isinstance(_invalid_vrtool_config, VrtoolConfig)

        # 2. Run test.
        with pytest.raises(ValueError) as exc_err:
            RunSafetyAssessment(_invalid_vrtool_config, None)

        # 3. Verify expectation
        assert str(exc_err.value) == _expected_error_mssg
