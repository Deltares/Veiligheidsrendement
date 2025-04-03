import shutil
from pathlib import Path
from typing import Iterator

import pytest
from peewee import SqliteDatabase

from tests import get_clean_test_results_dir, test_data
from vrtool.orm.orm_controllers import initialize_database

_database_ref_dir = test_data.joinpath("38-1_custom_measures")


@pytest.fixture(name="custom_measure_db_context")
def get_initialized_db_without_custom_measures_fixture(
    request: pytest.FixtureRequest,
) -> Iterator[SqliteDatabase]:
    """
    Yields a database's context which does not contain yet any ccustom measure.
    At the end of the test this fixture will close said context.
    """

    def _get_db_copy(reference_db: Path, request: pytest.FixtureRequest) -> Path:
        # Creates a copy of the database to avoid locking it
        # or corrupting its data.
        _output_directory = get_clean_test_results_dir(request)
        _copy_db = _output_directory.joinpath("vrtool_input.db")
        shutil.copyfile(reference_db, _copy_db)
        assert _copy_db.is_file()

        return _copy_db

    # 1. Define test data.
    _db_name = "without_custom_measures.db"
    _db_copy = _get_db_copy(_database_ref_dir.joinpath(_db_name), request)

    # Stablish connection
    _test_db_context = initialize_database(_db_copy)

    # Yield item to tests.
    yield _test_db_context

    # Close connection
    _test_db_context.close()
