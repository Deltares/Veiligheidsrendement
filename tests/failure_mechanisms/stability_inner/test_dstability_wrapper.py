from pathlib import Path

import numpy as np
import pytest

from tests import test_data
from vrtool.failure_mechanisms.stability_inner.dstability_wrapper import DStabilityWrapper


class TestDStabilityWrapper:

    def test_get_safety_factor_no_run_no_stage(self):
        """
        Test the get_safety_factor method of the DStabilityWrapper class for stage_id_result=None and without running D-Stability
        """
        # Setup
        _path_test_stix = test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        _dstab_wrapper = DStabilityWrapper(stix_path=Path(_path_test_stix),
                                           externals_path=Path(''))

        # Call
        _safety_factor = _dstab_wrapper.get_safety_factor(stage_id_result=None)

        # Assert
        assert isinstance(_safety_factor, np.ndarray)
        assert pytest.approx(1.3380575991293264) == _safety_factor

    def test_get_safety_factor_no_run_specified_valid_id(self):
        """
        Test the get_safety_factor method of the DStabilityWrapper class for a valid stage id and without running D-Stability
        """
        # Setup
        _path_test_stix = test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        _dstab_wrapper = DStabilityWrapper(stix_path=Path(_path_test_stix),
                                           externals_path=Path(''))

        # Call
        _safety_factor = _dstab_wrapper.get_safety_factor(stage_id_result=1)

        # Assert
        assert isinstance(_safety_factor, float)
        assert pytest.approx(1.3380575991293264) == _safety_factor

    def test_get_safety_factor_no_run_specified_invalid_id(self):
        """
        Test the get_safety_factor method of the DStabilityWrapper class for a valid stage id and raise and Exception
        """
        # Setup
        _path_test_stix = test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        _dstab_wrapper = DStabilityWrapper(stix_path=Path(_path_test_stix),
                                           externals_path=Path(''))

        # Call
        with pytest.raises(Exception) as exception_error:
            _safety_factor = _dstab_wrapper.get_safety_factor(stage_id_result=0)
        # Assert
        assert (
                str(exception_error.value)
                == "The requested stage id None does not have saved results in the provided stix RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix, please rerun DStability"
        )
