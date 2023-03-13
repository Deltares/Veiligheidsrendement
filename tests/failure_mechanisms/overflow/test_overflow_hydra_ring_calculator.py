from vrtool.failure_mechanisms import (FailureMechanismCalculatorProtocol)
from vrtool.failure_mechanisms.overflow import (
    OverflowHydraRingInput,
    OverflowHydraRingCalculator,
)

from vrtool.failure_mechanisms.mechanism_input import MechanismInput

import pandas as pd
import pytest


class TestOverflowHydraRingCalculator:
    def test_init_with_valid_data(self):
        # Setup
        _mechanism_input = MechanismInput("")
        _mechanism_input.input["h_crest"] = 0.1
        _mechanism_input.input["d_crest"] = 0.2
        _mechanism_input.input["hc_beta"] = pd.DataFrame(
            {"col1": [1, 2], "col2": [3, 4]}
        )

        _input = OverflowHydraRingInput.from_mechanism_input(_mechanism_input)
        _initial_year = 2023

        # Call
        _calculator = OverflowHydraRingCalculator(_input, _initial_year)

        # Assert
        assert isinstance(_calculator, OverflowHydraRingCalculator)
        assert isinstance(_calculator, FailureMechanismCalculatorProtocol)

    def test_init_with_invalid_mechanism_input(self):
        # Setup
        _initial_year = 2023

        # Call
        with pytest.raises(ValueError) as exception_error:
            OverflowHydraRingCalculator("NotInput", _initial_year)

        # Assert
        assert (
            str(exception_error.value)
            == "Expected instance of a OverflowHydraRingInput."
        )

    def test_init_with_invalid_load_input(self):
        # Setup
        _mechanism_input = MechanismInput("")
        _mechanism_input.input["h_crest"] = 0.1
        _mechanism_input.input["d_crest"] = 0.2
        _mechanism_input.input["hc_beta"] = pd.DataFrame(
            {"col1": [1, 2], "col2": [3, 4]}
        )

        _input = OverflowHydraRingInput.from_mechanism_input(_mechanism_input)

        # Call
        with pytest.raises(ValueError) as exception_error:
            OverflowHydraRingCalculator(_input, "NotAYear")

        # Assert
        assert str(exception_error.value) == "Expected instance of a int."
