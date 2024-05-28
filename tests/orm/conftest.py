import shutil
from typing import Iterator

import pytest

from tests import get_clean_test_results_dir
from vrtool.defaults.vrtool_config import VrtoolConfig


@pytest.fixture(name="custom_measures_vrtool_config")
def get_vrtool_config_for_custom_measures_db(
    request: pytest.FixtureRequest,
) -> Iterator[VrtoolConfig]:
    """
    Retrieves a valid `VrtoolConfig` instance ready to run a database
    for / with custom measures.
    In order to use it the test needs to provide the database path by using
    a marker such as :
        @pytest.mark.fixture_database(Path//to//database)
    For now this test assumes the selected traject is '38-1'.
    """
    # 1. Define test data.
    _marker = request.node.get_closest_marker("fixture_database")
    if _marker is None:
        _test_db = request.param
    else:
        _test_db = _marker.args[0]

    _output_directory = get_clean_test_results_dir(request)

    # Create a copy of the database to avoid locking it
    # or corrupting its data.
    _copy_db = _output_directory.joinpath("vrtool_input.db")
    shutil.copyfile(_test_db, _copy_db)

    # Generate a custom `VrtoolConfig`
    _vrtool_config = VrtoolConfig(
        input_directory=_copy_db.parent,
        input_database_name=_copy_db.name,
        traject="38-1",
        output_directory=_output_directory,
    )
    assert _vrtool_config.input_database_path.is_file()

    yield _vrtool_config
