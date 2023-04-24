import pytest

from tests.run_workflows import MockedDikeTraject
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.run_workflows.measures_workflow.run_measures import RunMeasures
from vrtool.run_workflows.vrtool_run_protocol import VrToolRunProtocol


class TestRunMeasures:
    def test_init_with_valid_data(self):
        # 1. Define test data
        _vr_config = VrtoolConfig()
        _vr_config.traject = "A traject"
        _traject = MockedDikeTraject()

        # 2. Run test
        _run = RunMeasures(_vr_config, _traject, "sth")

        # 3. Verify expectations
        assert isinstance(_run, RunMeasures)
        assert isinstance(_run, VrToolRunProtocol)

    def test_init_with_invalid_vr_config(self):
        # 1. Run test
        with pytest.raises(ValueError) as exception_error:
            RunMeasures("nothing", "else", "matters")

        # 2. Verify expectations
        assert str(exception_error.value) == "Expected instance of a VrtoolConfig."

    def test_init_with_invalid_selected_traject(self):
        # 1. Define test data.
        _vr_config = VrtoolConfig()

        # 1. Run test
        with pytest.raises(ValueError) as exception_error:
            RunMeasures(_vr_config, "else", "matters")

        # 2. Verify expectations
        assert str(exception_error.value) == "Expected instance of a DikeTraject."
