import filecmp
import shutil

import pytest

from tests import test_data, test_externals, test_results
from vrtool.failure_mechanisms.stability_inner.dstability_wrapper import (
    DStabilityWrapper,
)


class TestDStabilityWrapper:
    def test_initialize_with_valid_values(self):
        pass

    def test_initialize_missing_stix_path_raises(self):
        with pytest.raises(ValueError) as exception_error:
            DStabilityWrapper(None, test_externals)

        assert str(exception_error.value) == "Missing argument value stix_path."

    def test_initialize_missing_externals_path_raises(self):
        _path_test_stix = (
            test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        )

        with pytest.raises(ValueError) as exception_error:
            DStabilityWrapper(_path_test_stix, None)

        assert str(exception_error.value) == "Missing argument value externals_path."

    def test_rerun_stix_with_invalid_externals_path_raises(
        self, request: pytest.FixtureRequest
    ):
        _path_test_stix = (
            test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        )
        _invalid_externals = test_data / request.node.name
        assert not _invalid_externals.exists(), "This (test) folder should not exist."

        _expected_error = "Console executable not found at {}.".format(
            _invalid_externals.joinpath("DStabilityConsole\\D-Stability Console.exe")
        )

        with pytest.raises(Exception) as exception_error:
            DStabilityWrapper(_path_test_stix, _invalid_externals).rerun_stix()

        # This CalculationError exception comes from d-geolib.
        assert str(exception_error.value.message) == _expected_error

    @pytest.mark.externals
    @pytest.mark.slow
    def test_rerun_stix_with_valid_externals_path(self, request: pytest.FixtureRequest):
        # 1. Define test data.
        assert test_externals.joinpath(
            "DStabilityConsole"
        ).exists(), "No d-stability console available for testing."
        _path_test_stix = (
            test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        )
        assert _path_test_stix.exists(), "No valid stix file available."

        # Create a copy of the file to avoid issues on other tests.
        _test_file = test_results / request.node.name / "file_to_rerun.stix"
        if _test_file.exists():
            shutil.rmtree(_test_file.parent)

        _test_file.parent.mkdir(parents=True)
        shutil.copy(str(_path_test_stix), str(_test_file))
        assert _test_file.exists()

        # Verify both files are the same.
        assert filecmp.cmp(str(_path_test_stix), str(_test_file))

        # 2. Run test.
        DStabilityWrapper(_test_file, test_externals).rerun_stix()

        # 3. Verify expectations.
        assert not filecmp.cmp(str(_path_test_stix), str(_test_file))

    def test_get_safety_factor_no_run_no_stage(self):
        """
        Test the get_safety_factor method of the DStabilityWrapper class for stage_id_result=None and without running D-Stability
        """
        # Setup
        _path_test_stix = (
            test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        )
        _dstab_wrapper = DStabilityWrapper(
            stix_path=_path_test_stix, externals_path=test_externals
        )

        # Call
        _safety_factor = _dstab_wrapper.get_safety_factor(stage_id_result=None)

        # Assert
        assert isinstance(_safety_factor, float)
        assert pytest.approx(1.3380575991293264) == _safety_factor

    def test_get_safety_factor_no_run_specified_valid_id(self):
        """
        Test the get_safety_factor method of the DStabilityWrapper class for a valid stage id and without running D-Stability
        """
        # Setup
        _path_test_stix = (
            test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        )
        _dstab_wrapper = DStabilityWrapper(
            stix_path=_path_test_stix, externals_path=test_externals
        )

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
        _path_test_stix = (
            test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        )
        _dstab_wrapper = DStabilityWrapper(
            stix_path=_path_test_stix, externals_path=test_externals
        )

        # Call
        with pytest.raises(Exception) as exception_error:

            _safety_factor = _dstab_wrapper.get_safety_factor(stage_id_result=0)
        # Assert
        assert (
            str(exception_error.value)
            == "The requested stage id None does not have saved results in the provided stix RW001.+096_STBI_maatgevend_Segment_38005_1D1, please rerun DStability"
        )

    def test_add_stability_screen(self):
        # Setup
        _path_test_stix = (
            test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        )
        _dstab_wrapper = DStabilityWrapper(
            stix_path=_path_test_stix, externals_path=test_externals
        )

        # Call the method
        _dstab_wrapper.add_stability_screen(depth=5, location=10)

        # Assert that
        assert len(_dstab_wrapper.get_dstability_model.datastructure.reinforcements) == 2
        assert len(_dstab_wrapper.get_dstability_model.datastructure.reinforcements[0].ForbiddenLines) == 1
