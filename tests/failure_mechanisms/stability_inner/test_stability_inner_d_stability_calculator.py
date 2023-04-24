from pathlib import Path

from tests import test_data
from vrtool.failure_mechanisms import FailureMechanismCalculatorProtocol
from vrtool.failure_mechanisms.stability_inner.dstability_wrapper import DStabilityWrapper
from vrtool.failure_mechanisms.stability_inner.stability_inner_d_stability_calculator import \
    StabilityInnerDStabilityCalculator


class TestStabilityInnerSimpleCalculator:
    def test_init_with_valid_data(self):
        # Setup
        _path_test_stix = test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        _dstab_wrapper = DStabilityWrapper(stix_path=Path(_path_test_stix),
                                           externals_path=Path(''))
        _input = _dstab_wrapper.get_safety_factor(stage_id_result=None)

        # Call
        _calculator = StabilityInnerDStabilityCalculator(_input)

        # Assert
        assert isinstance(_calculator, StabilityInnerDStabilityCalculator)
        assert isinstance(_calculator, FailureMechanismCalculatorProtocol)

