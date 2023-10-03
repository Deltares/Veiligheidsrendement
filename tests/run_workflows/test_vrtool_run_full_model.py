import pytest

from tests.run_workflows import MockedDikeTraject
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.run_workflows.vrtool_run_full_model import RunFullModel
from vrtool.run_workflows.vrtool_run_protocol import VrToolRunProtocol


class TestVrtoolRunFullModel:
    def test_init_with_valid_data(self):
        # 1. Define test data
        _vr_config = VrtoolConfig()
        _traject = MockedDikeTraject()

        # 2. Run test
        _run = RunFullModel(_vr_config, _traject)

        # 3. Verify expectations
        assert isinstance(_run, RunFullModel)
        assert isinstance(_run, VrToolRunProtocol)

    def test_init_with_invalid_vr_config(self):
        # 1. Run test
        with pytest.raises(ValueError) as exception_error:
            RunFullModel("paradise", "city")

        # 2. Verify expectations
        assert str(exception_error.value) == "Expected instance of a VrtoolConfig."

    def test_init_with_invalid_selected_traject(self):
        # 1. Define test data.
        _vr_config = VrtoolConfig()

        # 2. Run test
        with pytest.raises(ValueError) as exception_error:
            RunFullModel(_vr_config, "city")

        # 3. Verify expectations
        assert str(exception_error.value) == "Expected instance of a DikeTraject."
