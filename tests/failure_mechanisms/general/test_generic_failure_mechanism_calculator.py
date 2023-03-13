import pytest

from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.failure_mechanisms.failure_mechanism_calculator_protocol import (
    FailureMechanismCalculatorProtocol,
)

from vrtool.failure_mechanisms.general import (
    GenericFailureMechanismCalculator,
    GenericFailureMechanismInput,
)


class TestGenericFailureMechanismCalculator:
    def test_init_with_valid_data(self):
        # Setup
        beta_table = {1: 0.1, 2: 0.2, 3: 0.3}
        _mechanism_input = MechanismInput("")
        _mechanism_input.input["beta"] = beta_table

        _input = GenericFailureMechanismInput.from_mechanism_input(_mechanism_input)

        # Call
        _calculator = GenericFailureMechanismCalculator(_input)

        # Assert
        assert isinstance(_calculator, GenericFailureMechanismCalculator)
        assert isinstance(_calculator, FailureMechanismCalculatorProtocol)

    def test_init_with_invalid_data(self):
        # Call
        with pytest.raises(ValueError) as exception_error:
            GenericFailureMechanismCalculator("value")

        # Assert
        assert (
            str(exception_error.value)
            == "Expected instance of a GenericFailureMechanismInput."
        )
