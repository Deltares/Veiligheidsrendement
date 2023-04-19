import numpy as np
import pytest

from vrtool.failure_mechanisms import FailureMechanismCalculatorProtocol
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.failure_mechanisms.stability_inner.stability_inner_d_stability_calculator import (
    StabilityInnerDStabilityCalculator,
    StabilityInnerDStabilityInput,
)
from tests import test_data


class TestStabilityInnerSimpleCalculator:
    def test_init_with_valid_data(self):
        # Setup
        _mechanism_input = MechanismInput("")
        _path_test_stix = test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        _mechanism_input.input["STIXNAAM"] = str(_path_test_stix)

        _input = StabilityInnerDStabilityInput.from_stix_input(_mechanism_input)

        # Call
        _calculator = StabilityInnerDStabilityCalculator(_input)

        # Assert
        assert isinstance(_calculator, StabilityInnerDStabilityCalculator)
        assert isinstance(_calculator, FailureMechanismCalculatorProtocol)
        assert isinstance(_input.safety_factor, np.ndarray)
        assert pytest.approx(1.3380575991293264) == _input.safety_factor

    def test_init_with_valid_specified_stage_id(self):
        # Setup
        _mechanism_input = MechanismInput("")
        _path_test_stix = test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        _mechanism_input.input["STIXNAAM"] = str(_path_test_stix)
        _mechanism_input.input["STAGE_ID_RESULT"] = 1

        _input = StabilityInnerDStabilityInput.from_stix_input(_mechanism_input)

        # Call
        _calculator = StabilityInnerDStabilityCalculator(_input)

        # Assert
        assert isinstance(_calculator, StabilityInnerDStabilityCalculator)
        assert isinstance(_calculator, FailureMechanismCalculatorProtocol)
        assert isinstance(_input.safety_factor, np.ndarray)
        assert pytest.approx(1.3380575991293264) == _input.safety_factor

    def test_init_with_invalid_specified__stage_id(self):
        # Setup
        _mechanism_input = MechanismInput("")
        _path_test_stix = test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        _mechanism_input.input["STIXNAAM"] = str(_path_test_stix)
        _mechanism_input.input["STAGE_ID_RESULT"] = 0

        # Call
        with pytest.raises(Exception) as exception_error:
            _input = StabilityInnerDStabilityInput.from_stix_input(_mechanism_input)
            _calculator = StabilityInnerDStabilityCalculator(_input)

        # Assert
        assert (
                str(exception_error.value)
                == "The requested stage id None does not have saved results in the provided stix RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix, please rerun DStability"
        )
