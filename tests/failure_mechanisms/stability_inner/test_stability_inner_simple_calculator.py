import numpy as np
import pytest

from vrtool.common.enums.computation_type_enum import ComputationTypeEnum
from vrtool.failure_mechanisms import FailureMechanismCalculatorProtocol
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.failure_mechanisms.stability_inner import (
    StabilityInnerSimpleCalculator,
    StabilityInnerSimpleInput,
)


class TestStabilityInnerSimpleCalculator:
    def test_init_with_valid_data(self):
        # Setup
        _mechanism_input = MechanismInput("")
        _mechanism_input.input["sf_2025"] = np.array([0.1], dtype=float)
        _mechanism_input.input["sf_2075"] = np.array([0.2], dtype=float)

        _input = StabilityInnerSimpleInput.from_mechanism_input(_mechanism_input)

        # Call
        _calculator = StabilityInnerSimpleCalculator(_input)

        # Assert
        assert isinstance(_calculator, StabilityInnerSimpleCalculator)
        assert isinstance(_calculator, FailureMechanismCalculatorProtocol)

    def test_init_with_invalid_data(self):
        # Call
        with pytest.raises(ValueError) as exception_error:
            StabilityInnerSimpleCalculator(ComputationTypeEnum.SIMPLE)

        # Assert
        assert (
            str(exception_error.value)
            == "Expected instance of a StabilityInnerSimpleInput."
        )
