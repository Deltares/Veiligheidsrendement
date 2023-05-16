import pytest
from peewee import SqliteDatabase

from tests import test_data
from vrtool.orm.orm_controllers import open_database


@pytest.fixture(autouse=False, scope="module")
def db_fixture():
    _db_file = test_data / "test_db" / "vrtool_db.db"
    assert _db_file.is_file()

    _db = open_database(_db_file)
    assert isinstance(_db, SqliteDatabase)

    yield _db
