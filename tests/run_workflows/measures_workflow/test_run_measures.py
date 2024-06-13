import pytest

from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.run_workflows.measures_workflow.run_measures import RunMeasures
from vrtool.run_workflows.vrtool_run_protocol import VrToolRunProtocol


class TestRunMeasures:
    def test_init_with_valid_data(self, mocked_dike_traject: DikeTraject):
        # 1. Define test data
        _vr_config = VrtoolConfig()
        _vr_config.traject = "A traject"

        # 2. Run test
        _run = RunMeasures(_vr_config, mocked_dike_traject)

        # 3. Verify expectations
        assert isinstance(_run, RunMeasures)
        assert isinstance(_run, VrToolRunProtocol)

    def test_init_with_invalid_vr_config(self):
        # 1. Run test
        with pytest.raises(ValueError) as exception_error:
            RunMeasures("paradise", "city")

        # 2. Verify expectations
        assert str(exception_error.value) == "Expected instance of a VrtoolConfig."

    def test_init_with_invalid_selected_traject(self):
        # 1. Define test data.
        _vr_config = VrtoolConfig()

        # 1. Run test
        with pytest.raises(ValueError) as exception_error:
            RunMeasures(_vr_config, "city")

        # 2. Verify expectations
        assert str(exception_error.value) == "Expected instance of a DikeTraject."
