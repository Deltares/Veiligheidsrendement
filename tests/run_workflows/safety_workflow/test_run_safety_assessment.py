import shutil

import pytest

from src.defaults.vrtool_config import VrtoolConfig
from src.run_workflows.safety_workflow.run_safety_assessment import RunSafetyAssessment
from src.run_workflows.vrtool_run_protocol import VrToolRunProtocol
from tests import test_results


class TestRunSafetyAssessment:
    def test_init(self):
        _assessment = RunSafetyAssessment("sth")

        assert isinstance(_assessment, RunSafetyAssessment)
        assert isinstance(_assessment, VrToolRunProtocol)

    def test_get_valid_output_dir_creates_missing_directories(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data
        _assessment = RunSafetyAssessment("sth")
        _assessment.vr_config = VrtoolConfig()
        _assessment.vr_config.output_directory = test_results / request.node.name
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
