import shutil
import sqlite3
from collections import OrderedDict
from pathlib import Path

import peewee
import pytest

from tests import test_data, test_results
from vrtool.orm.migration.migrate_database import MigrateDb
from vrtool.orm.models.version import Version as DbVersion
from vrtool.orm.orm_controllers import open_database
from vrtool.orm.version.orm_version import OrmVersion


class TestMigrateDatabase:
    @pytest.fixture(name="empty_dabase_path_v777")
    def _get_empty_database_path_v777_fixture(self, empty_db_path: Path):
        with open_database(empty_db_path).connection_context():
            _version = DbVersion.get_or_none()
            _version.orm_version = "7.7.7"
            _version.save()

        yield empty_db_path

    def test_initialize(self):
        # 1. Run test
        _migrate_db = MigrateDb()

        # 2. Verify expectations
        assert _migrate_db is not None
        assert _migrate_db.scripts_dict is not None
        assert _migrate_db.orm_version is not None

    def test_apply_migration_script(self, empty_dabase_path_v777: Path):
        # 1. Define test data
        _script = test_data.joinpath("orm", "populated_folder", "v7_7_8.sql")
        _query = "SELECT * FROM TestTable"
        _connected_db = open_database(empty_dabase_path_v777)
        with pytest.raises(peewee.OperationalError):
            _connected_db.execute_sql(_query)
        _connected_db.close()

        # 2. Run test
        MigrateDb.apply_migration_script(empty_dabase_path_v777, _script)

        # 3. Verify expectations
        _connected_db = open_database(empty_dabase_path_v777)
        _result = _connected_db.execute_sql(_query)
        _connected_db.close()
        assert isinstance(_result, sqlite3.Cursor)

    def test_apply_erroneous_script_raises(self, empty_dabase_path_v777: Path):
        # 1. Define test data
        _script = test_data.joinpath("orm", "version_file.py")

        # 2. Run test
        with pytest.raises(Exception) as exc_err:
            MigrateDb.apply_migration_script(empty_dabase_path_v777, _script)

        # 3. Verify expectations
        assert "syntax error" in str(exc_err.value)

    def test_migrate_single_db_stops_on_major(self, empty_dabase_path_v777: Path):
        """
        This test loops over the different conversion scripts available
        and executes some of them:
            v7_7_7.sql: will not be executed
            v7_7_8.sql: will be executed (creates table TestTable with record id 778)
            v7_8_0.sql: will be executed (creates record id 780)
            v8_0_0.sql: will be executed (creates record id 800)
            v8_0_1.sql: will not be executed as v8_0_0 is a major version
        """
        # 1. Define test data
        _scripts_dir = test_data.joinpath("orm", "populated_folder")
        _migrate_db = MigrateDb()
        _migrate_db.scripts_dict = _migrate_db._parse_scripts_dir(_scripts_dir)
        _version_path = test_data.joinpath("orm", "version_file.py")
        _migrate_db.orm_version = OrmVersion(_version_path)

        _query = "SELECT id from TestTable ORDER BY id"
        _connected_db = open_database(empty_dabase_path_v777)
        with pytest.raises(peewee.OperationalError):
            _ = _connected_db.execute_sql(_query)
        _connected_db.close()

        # 2. Run test
        _migrate_db.migrate_single_db(empty_dabase_path_v777)

        # 3. Verify expectations
        _connected_db = open_database(empty_dabase_path_v777)
        _result = _connected_db.execute_sql(_query)
        _rows = _result.fetchall()
        assert len(_rows) == 3
        for _id in (778, 780, 800):
            assert ((_id) in _row[0] for _row in _rows)
        _connected_db.close()

    def test_migrate_single_db_no_migration(self, empty_dabase_path_v777: Path):
        # 1. Define test data
        _scripts_dir = test_data.joinpath("orm", "populated_folder")
        _migrate_db = MigrateDb()
        _migrate_db.scripts_dict = OrderedDict(
            {(7, 7, 7): _scripts_dir.joinpath("v7_7_7.sql")}
        )
        _version_path = test_data.joinpath("orm", "version_file.py")
        _migrate_db.orm_version = OrmVersion(_version_path)

        # 2. Run test
        _migrate_db.migrate_single_db(empty_dabase_path_v777)

        # 3. Verify expectations
        _query = "SELECT id from TestTable ORDER BY id"
        _connected_db = open_database(empty_dabase_path_v777)
        with pytest.raises(peewee.OperationalError):
            _ = _connected_db.execute_sql(_query)
        _connected_db.close()

    def test_migrate_single_db_force_orm(
        self, empty_dabase_path_v777: Path, request: pytest.FixtureRequest
    ):
        # 1. Define test data
        _scripts_dir = test_data.joinpath("orm", "populated_folder")
        _migrate_db = MigrateDb(force_orm=True)
        _migrate_db.scripts_dict = _migrate_db._parse_scripts_dir(_scripts_dir)

        _version_path = test_data.joinpath("orm", "version_file.py")
        _test_version_path = test_results.joinpath(request.node.name, "version_file.py")
        shutil.copyfile(_version_path, _test_version_path)
        _migrate_db.orm_version = OrmVersion(_test_version_path)

        # 2. Run test
        _migrate_db.migrate_single_db(empty_dabase_path_v777)

        # 3. Verify expectations
        _connected_db = open_database(empty_dabase_path_v777)
        _query = "SELECT id from TestTable ORDER BY id"
        _result = _connected_db.execute_sql(_query)
        _rows = _result.fetchall()
        assert len(_rows) == 4
        for _id in (778, 780, 800, 801):
            assert ((_id) in _row[0] for _row in _rows)
        _connected_db.close()

        assert _migrate_db.orm_version.read_version() == (8, 0, 1)
