import pytest

from vrtool.failure_mechanisms import FailureMechanismCalculatorProtocol
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.failure_mechanisms.stability_inner import (
    StabilityInnerSimpleCalculator,
    StabilityInnerSimpleInput,
)


class TestStabilityInnerSimpleCalculator:
    def test_init_with_valid_data(self, mechanism_input_fixture: MechanismInput):
        # Setup
        assert isinstance(mechanism_input_fixture, MechanismInput)
        _input = StabilityInnerSimpleInput.from_mechanism_input(mechanism_input_fixture)

        # Call
        _calculator = StabilityInnerSimpleCalculator(_input)

        # Assert
        assert isinstance(_calculator, StabilityInnerSimpleCalculator)
        assert isinstance(_calculator, FailureMechanismCalculatorProtocol)

    def test_init_with_invalid_data(self):
        # Call
        with pytest.raises(ValueError) as exception_error:
            StabilityInnerSimpleCalculator("simple")

        # Assert
        assert (
            str(exception_error.value)
            == "Expected instance of a StabilityInnerSimpleInput."
        )
