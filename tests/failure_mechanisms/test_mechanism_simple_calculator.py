import pytest

from vrtool.common.enums.computation_type_enum import ComputationTypeEnum
from vrtool.failure_mechanisms import (
    FailureMechanismCalculatorProtocol,
    MechanismInput,
    MechanismSimpleCalculator,
    MechanismSimpleInput,
)


class TestStabilityInnerSimpleCalculator:
    def test_init_with_valid_data(self, mechanism_input_fixture: MechanismInput):
        # Setup
        assert isinstance(mechanism_input_fixture, MechanismInput)
        _input = MechanismSimpleInput.from_mechanism_input(mechanism_input_fixture)

        # Call
        _calculator = MechanismSimpleCalculator(_input)

        # Assert
        assert isinstance(_calculator, MechanismSimpleCalculator)
        assert isinstance(_calculator, FailureMechanismCalculatorProtocol)

    def test_init_with_invalid_data(self):
        # Call
        with pytest.raises(ValueError) as exception_error:
            MechanismSimpleCalculator(ComputationTypeEnum.SIMPLE)

        # Assert
        assert (
            str(exception_error.value)
            == "Expected instance of a MechanismSimpleInput."
        )
