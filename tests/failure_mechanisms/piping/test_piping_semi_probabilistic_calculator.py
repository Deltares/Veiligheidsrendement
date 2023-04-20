import numpy as np
import pytest

from vrtool.failure_mechanisms import FailureMechanismCalculatorProtocol
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.failure_mechanisms.piping import PipingSemiProbabilisticCalculator
from vrtool.common.hydraulic_loads.load_input import LoadInput
from vrtool.flood_defence_system.dike_traject_info import DikeTrajectInfo


class TestPipingSemiProbabilisticCalculator:
    def test_init_with_valid_data(self):
        # Setup
        _mechanism_input = MechanismInput("")
        _load = LoadInput([])

        # Call
        _calculator = PipingSemiProbabilisticCalculator(
            _mechanism_input, _load, 0, DikeTrajectInfo(traject_name="")
        )

        # Assert
        assert isinstance(_calculator, PipingSemiProbabilisticCalculator)
        assert isinstance(_calculator, FailureMechanismCalculatorProtocol)

    def test_init_with_invalid_mechanism_input(self):
        # Setup
        _load = LoadInput([])

        # Call
        with pytest.raises(ValueError) as exception_error:
            PipingSemiProbabilisticCalculator(
                "NotMechanismInput", _load, 0, DikeTrajectInfo(traject_name="")
            )

        # Assert
        assert str(exception_error.value) == "Expected instance of a MechanismInput."

    def test_init_with_invalid_load_input(self):
        # Setup
        _mechanism_input = MechanismInput("")
        _load = LoadInput([])

        # Call
        with pytest.raises(ValueError) as exception_error:
            PipingSemiProbabilisticCalculator(
                _mechanism_input, "NotLoad", 0, DikeTrajectInfo(traject_name="")
            )

        # Assert
        assert str(exception_error.value) == "Expected instance of a LoadInput."

    def test_init_with_invalid_initial_year(self):
        # Setup
        _mechanism_input = MechanismInput("")
        _load = LoadInput([])

        # Call
        with pytest.raises(ValueError) as exception_error:
            PipingSemiProbabilisticCalculator(
                _mechanism_input,
                _load,
                "NotInitialYear",
                DikeTrajectInfo(traject_name=""),
            )

        # Assert
        assert str(exception_error.value) == "Expected instance of a int."

    def test_init_with_invalid_trajec_info(self):
        # Setup
        _mechanism_input = MechanismInput("")
        _load = LoadInput([])

        # Call
        with pytest.raises(ValueError) as exception_error:
            PipingSemiProbabilisticCalculator(
                _mechanism_input, _load, 0, "NotATrajectInfo"
            )

        # Assert
        assert str(exception_error.value) == "Expected instance of a DikeTrajectInfo."
