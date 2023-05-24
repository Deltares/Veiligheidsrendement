import pytest
from peewee import SqliteDatabase

from tests import test_data
from vrtool.orm.orm_controllers import open_database


@pytest.fixture(autouse=False)
def empty_db_fixture():
    _db_file = test_data / "test_db" / f"empty_db.db"
    _db = open_database(_db_file)
    assert isinstance(_db, SqliteDatabase)

    with _db.atomic() as transaction:
        yield _db
        transaction.rollback()
    _db.close()
