from pathlib import Path

import numpy as np
import pytest

from vrtool.failure_mechanisms import FailureMechanismCalculatorProtocol
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.failure_mechanisms.stability_inner.stability_inner_d_stability_calculator import \
    StabilityInnerDStabilityInput, StabilityInnerDStabilityCalculator


class TestStabilityInnerSimpleCalculator:
    def test_init_with_valid_data(self):
        # Setup
        _mechanism_input = MechanismInput("")
        _path_test_stix = Path(__file__).parent.parent.parent / "test_data/stix" /"RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        _mechanism_input.input["STIXNAAM"] = str(_path_test_stix)

        _input = StabilityInnerDStabilityInput.from_stix_input(_mechanism_input)

        # Call
        _calculator = StabilityInnerDStabilityCalculator(_input)

        # Assert
        assert isinstance(_calculator, StabilityInnerDStabilityCalculator)
        assert isinstance(_calculator, FailureMechanismCalculatorProtocol)
        assert isinstance(_input.safety_factor, float)
