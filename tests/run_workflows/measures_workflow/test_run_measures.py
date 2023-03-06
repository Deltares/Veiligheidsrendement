from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.run_workflows.measures_workflow.run_measures import RunMeasures
from vrtool.run_workflows.vrtool_run_protocol import VrToolRunProtocol
from vrtool.flood_defence_system.dike_traject import DikeTraject
import pytest

class TestRunMeasures:
    def test_init_with_valid_data(self):
        # 1. Define test data
        _vr_config = VrtoolConfig()
        _traject = DikeTraject(_vr_config)

        # 2. Run test
        _run = RunMeasures(_vr_config, _traject, "sth")

        # 3. Verify expectations
        assert isinstance(_run, RunMeasures)
        assert isinstance(_run, VrToolRunProtocol)
    
    def test_init_with_invalid_vr_config(self):
        # 1. Run test
        with pytest.raises(ValueError) as exc_err:
            RunMeasures("nothing", "else", "matters")

        # 2. Verify expectations
        assert str(exc_err.value) == "Expected instance of a VrtoolConfig."

    def test_init_with_invalid_selected_traject(self):
        # 1. Define test data.
        _vr_config = VrtoolConfig()

        # 1. Run test
        with pytest.raises(ValueError) as exc_err:
            RunMeasures(_vr_config, "else", "matters")

        # 2. Verify expectations
        assert str(exc_err.value) == "Expected instance of a DikeTraject."