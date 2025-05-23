import filecmp
import shutil
import sys

import pytest

from tests import test_data, test_externals, test_results
from vrtool.failure_mechanisms.stability_inner.dstability_wrapper import (
    DStabilityWrapper,
)


class TestDStabilityWrapper:
    def test_initialize_missing_stix_path_raises(self):
        with pytest.raises(ValueError) as exception_error:
            DStabilityWrapper(stix_path=None, externals_path=test_externals)

        assert str(exception_error.value) == "Missing argument value stix_path."

    def test_initialize_missing_externals_path_raises(self):
        _path_test_stix = (
            test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        )

        with pytest.raises(ValueError) as exception_error:
            DStabilityWrapper(stix_path=_path_test_stix, externals_path=None)

        assert str(exception_error.value) == "Missing argument value externals_path."

    def test_rerun_stix_with_invalid_externals_path_raises(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _path_test_stix = test_data.joinpath(
            "stix", "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        )
        _invalid_externals = test_data.joinpath(request.node.name)
        _console_path = DStabilityWrapper.get_dstability_console_path(
            _invalid_externals
        )
        assert not _console_path.exists(), "This (test) file should not exist."

        _expected_error = "Console executable not found at {}.".format(_console_path)

        with pytest.raises(Exception) as exception_error:
            DStabilityWrapper(
                stix_path=_path_test_stix, externals_path=_invalid_externals
            ).rerun_stix()

        # This CalculationError exception comes from d-geolib and contains a message field.
        assert str(exception_error.value.message) == _expected_error

    @pytest.mark.externals
    @pytest.mark.skipif(
        sys.platform != "win32", reason="Pywin32 only available for windows"
    )
    def test_validate_dstability_version(self):
        # 1. Define test data.
        import win32api

        _supported_major_version = "2024"
        _dstability_exe = test_externals.joinpath(
            "DStabilityConsole", "D-Stability Console.exe"
        )

        assert (
            _dstability_exe.is_file()
        ), "No d-stability console available for testing."

        # 2. Run test.
        _version_info = win32api.GetFileVersionInfo(str(_dstability_exe), "\\")
        _found_version = str(win32api.HIWORD(_version_info["FileVersionMS"]))

        # 3. Verify expectations.
        assert _found_version == _supported_major_version

    @pytest.mark.externals
    def test_rerun_stix_with_valid_externals_path(self, request: pytest.FixtureRequest):
        # 1. Define test data.
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
        DStabilityWrapper(
            stix_path=_test_file,
            externals_path=test_externals,
        ).rerun_stix()

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
        _safety_factor = _dstab_wrapper.get_safety_factor()

        # Assert
        assert isinstance(_safety_factor, float)
        assert pytest.approx(1.3380575991293264) == _safety_factor

    def test_get_safety_factor_no_rerun(self):
        """
        Test the get_safety_factor method of the DStabilityWrapper class for a valid stage id and without running D-Stability
        """
        # Setup
        _path_test_stix = (
            test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        )
        _dstab_wrapper = DStabilityWrapper(
            stix_path=_path_test_stix,
            externals_path=test_externals,
        )

        # Call
        _safety_factor = _dstab_wrapper.get_safety_factor()

        # Assert
        assert isinstance(_safety_factor, float)
        assert pytest.approx(1.3380575991293264) == _safety_factor

    def test_get_safety_factor_with_rerun(self, request: pytest.FixtureRequest):
        """
        Test the get_safety_factor method of the DStabilityWrapper class for a valid stage id and raise and Exception
        """
        # Setup
        _stix_name = (
            "RW001.+096_STBI_maatgevend_Segment_38005_1D1_no_results_saved.stix"
        )
        _path_test_stix = test_data.joinpath("stix", _stix_name)
        _dstab_wrapper = DStabilityWrapper(
            stix_path=_path_test_stix,
            externals_path=test_externals,
        )
        _export_path = test_results / request.node.name

        # Save the rerun file to avoid overwriting the original file.
        if not _export_path.exists():
            _export_path.mkdir(parents=True)
        _dstab_wrapper.save_dstability_model(_export_path / _stix_name)

        # Call
        _safety_factor = _dstab_wrapper.get_safety_factor()

        # Assert
        assert isinstance(_safety_factor, float)
        assert pytest.approx(1.3400996495000572) == _safety_factor

    def test_add_stability_screen(self):
        # Setup
        _path_test_stix = (
            test_data / "stix" / "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix"
        )
        _dstab_wrapper = DStabilityWrapper(
            stix_path=_path_test_stix,
            externals_path=test_externals,
        )

        # Call the method
        _dstab_wrapper.add_stability_screen(bottom_screen=5, location=10)

        # Assert that
        assert (
            len(_dstab_wrapper.get_dstability_model.datastructure.reinforcements) == 3
        )
        assert (
            len(
                _dstab_wrapper.get_dstability_model.datastructure.reinforcements[
                    0
                ].ForbiddenLines
            )
            == 1
        )
