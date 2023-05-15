import pytest
from peewee import SqliteDatabase

from tests import test_results
from vrtool.orm.orm_controllers import initialize_database


@pytest.fixture(autouse=False)
def empty_db_fixture(request: pytest.FixtureRequest):
    _parts = request.node.nodeid.split("::")
    _db_file = test_results / _parts[-2] / f"{_parts[-1]}_db.db"
    if _db_file.exists():
        _db_file.unlink()

    _db = initialize_database(_db_file)
    assert isinstance(_db, SqliteDatabase)

    yield _db
