import numpy as np
import pytest

from vrtool.failure_mechanisms import FailureMechanismCalculatorProtocol
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.failure_mechanisms.overflow import (
    OverflowSimpleCalculator,
    OverflowSimpleInput,
)
from vrtool.common.hydraulic_loads.load_input import LoadInput


class TestOverflowSimpleCalculator:
    def test_init_with_valid_data(self):
        # Setup
        _mechanism_input = MechanismInput("")
        _mechanism_input.input["dhc(t)"] = np.array(0.1)
        _mechanism_input.input["h_crest"] = np.array(0.2)
        _mechanism_input.input["q_crest"] = np.array(0.3)
        _mechanism_input.input["h_c"] = np.array(0.4)
        _mechanism_input.input["q_c"] = np.array(0.5)
        _mechanism_input.input["beta"] = np.array(0.6)

        _input = OverflowSimpleInput.from_mechanism_input(_mechanism_input)
        _load = LoadInput([])

        # Call
        _calculator = OverflowSimpleCalculator(_input, _load)

        # Assert
        assert isinstance(_calculator, OverflowSimpleCalculator)
        assert isinstance(_calculator, FailureMechanismCalculatorProtocol)

    def test_init_with_invalid_mechanism_input(self):
        # Setup
        _load = LoadInput([])

        # Call
        with pytest.raises(ValueError) as exception_error:
            OverflowSimpleCalculator("NotInput", _load)

        # Assert
        assert (
            str(exception_error.value) == "Expected instance of a OverflowSimpleInput."
        )

    def test_init_with_invalid_load_input(self):
        # Setup
        _mechanism_input = MechanismInput("")
        _mechanism_input.input["dhc(t)"] = np.array(0.1)
        _mechanism_input.input["h_crest"] = np.array(0.2)
        _mechanism_input.input["q_crest"] = np.array(0.3)
        _mechanism_input.input["h_c"] = np.array(0.4)
        _mechanism_input.input["q_c"] = np.array(0.5)
        _mechanism_input.input["beta"] = np.array(0.6)
        _input = OverflowSimpleInput.from_mechanism_input(_mechanism_input)

        # Call
        with pytest.raises(ValueError) as exception_error:
            OverflowSimpleCalculator(_input, "NotLoad")

        # Assert
        assert str(exception_error.value) == "Expected instance of a LoadInput."
