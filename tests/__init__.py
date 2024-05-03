import hashlib
import shutil
from pathlib import Path

from pytest import FixtureRequest

from vrtool.defaults.vrtool_config import VrtoolConfig

test_data = Path(__file__).parent / "test_data"
test_results = Path(__file__).parent / "test_results"
test_externals = Path(__file__).parent / "test_externals"

if not test_results.is_dir():
    test_results.mkdir(parents=True)


def get_clean_test_results_dir(request: FixtureRequest) -> Path:
    """
    Gets a new results directory considering the test name and "cases".
    When the results directory exists it gets removed to prevent data
    from becoming corrupted.

    Args:
        request (FixtureRequest): Contains information of the test name and cases.

    Returns:
        Path: Generated directory where results can be exported.
    """
    _test_dir = test_results.joinpath(request.node.originalname)

    if hasattr(request.node, "callspec"):
        # It's a parametrized test:
        _normalized_case = (
            request.node.callspec.id.replace(":", "__")
            .replace(",", "__")
            .replace(" ", "_")
        )
        _test_dir = _test_dir.joinpath(_normalized_case)

    if _test_dir.exists():
        shutil.rmtree(_test_dir)

    _test_dir.mkdir(parents=True, exist_ok=True)
    return _test_dir


def get_copy_of_reference_directory(directory_name: str) -> Path:
    """
    Gets a copy of the reference directory to avoid locking databases.

    Args:
        directory_name (str): Name of the subdirectory in the `test_data`.

    Returns:
        Path: Location of the `directory_name` relative to the `test_data` path.
    """
    # Check if reference path exists.
    _reference_path = test_data.joinpath(directory_name)
    assert _reference_path.exists()

    # Ensure new path does not exist yet.
    _new_path = test_results.joinpath(directory_name)
    if _new_path.exists():
        shutil.rmtree(_new_path)

    # Copy the reference to new location.
    shutil.copytree(_reference_path, _new_path)
    assert _new_path.exists()

    # Return new path location.
    return _new_path


def get_vrtool_config_test_copy(config_file: Path, test_name: str) -> VrtoolConfig:
    """
    Gets a `VrtoolConfig` with a copy of the database to avoid version issues.
    """
    # Create a results directory (ignored by git)
    _test_results_directory = test_results.joinpath(test_name)
    if _test_results_directory.exists():
        shutil.rmtree(_test_results_directory)
    _test_results_directory.mkdir(parents=True)

    # Get the current configuration
    _vrtool_config = VrtoolConfig.from_json(config_file)

    # Create a db copy.
    _new_db_name = "test_{}.db".format(
        hashlib.shake_128(_test_results_directory.__bytes__()).hexdigest(4)
    )
    _new_db_path = _test_results_directory.joinpath(_new_db_name)
    if _new_db_path.exists():
        # Somehow it was not removed in the previous test run.
        _new_db_path.unlink(missing_ok=True)

    shutil.copy(_vrtool_config.input_database_path, _new_db_path)

    # Set new configuration values.
    _vrtool_config.input_directory = _test_results_directory
    _vrtool_config.input_database_name = _new_db_name
    _vrtool_config.output_directory = _test_results_directory.joinpath("output")
    _vrtool_config.output_directory.mkdir()

    return _vrtool_config
