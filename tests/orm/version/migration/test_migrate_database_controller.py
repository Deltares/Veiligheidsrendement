import sqlite3
from pathlib import Path

import peewee
import pytest

from tests import test_results
from vrtool.orm.orm_controllers import open_database_without_compatibility_check
from vrtool.orm.version.migration.database_version import DatabaseVersion
from vrtool.orm.version.migration.migrate_database_controller import (
    MigrateDatabaseController,
)
from vrtool.orm.version.orm_version import OrmVersion


class TestMigrateDatabaseController:
    def test_initialize(self, valid_conversion_input: Path):
        # 1. Run test
        _migrate_db = MigrateDatabaseController(valid_conversion_input)

        # 2. Verify expectations
        assert _migrate_db is not None
        assert _migrate_db.orm_version is not None
        assert len(_migrate_db.script_versions) == 5

    def test_initialize_empty_scripts_dir(self):
        # 1. Define test data
        _empty_dir = test_results.joinpath("empty_folder")
        _empty_dir.mkdir()

        # 2. Run test
        _migrate_db = MigrateDatabaseController(_empty_dir)

        # 3. Verify expectations
        assert len(_migrate_db.script_versions) == 0
        _empty_dir.rmdir()

    @pytest.mark.parametrize(
        "increment, expected",
        [
            pytest.param((0, 0, 0), True, id="No increment"),
            pytest.param((1, 0, 0), False, id="Major increment"),
            pytest.param((0, 1, 0), False, id="Minor increment"),
            pytest.param((0, 0, 1), True, id="Patch increment"),
        ],
    )
    def test_is_database_compatible(
        self,
        empty_db_path: Path,
        increment: tuple[int, int, int],
        expected: bool,
    ):
        # 1. Define test data
        _from_db_version = DatabaseVersion.from_database(empty_db_path)
        _orm_version = OrmVersion.from_orm()
        _to_db_version = DatabaseVersion(
            major=increment[0] + _orm_version.major,
            minor=increment[1] + _orm_version.minor,
            patch=increment[2] + _orm_version.patch,
            database_path=_from_db_version.database_path,
        )
        _from_db_version.update_version(_to_db_version)

        # 2. Run test
        _result = MigrateDatabaseController.is_database_compatible(empty_db_path)

        # 3. Verify expectations
        assert _result == expected

    def test__apply_migration_script(self, valid_conversion_input: Path):
        # 1. Define test data
        _script = valid_conversion_input.joinpath("valid_script.txt")
        _database_path = valid_conversion_input.joinpath("test_db.db")

        _query = "SELECT * FROM ValidTable"
        with open_database_without_compatibility_check(_database_path) as _connected_db:
            with pytest.raises(peewee.OperationalError):
                _connected_db.execute_sql(_query)

        # 2. Run test
        MigrateDatabaseController(Path(""))._apply_migration_script(
            _database_path, _script
        )

        # 3. Verify expectations
        with open_database_without_compatibility_check(_database_path) as _connected_db:
            _result = _connected_db.execute_sql(_query)
        assert isinstance(_result, sqlite3.Cursor)

    def test__apply_erroneous_script_raises(self, valid_conversion_input: Path):
        # 1. Define test data
        _script = valid_conversion_input.joinpath("wrong_script.txt")
        _database_path = valid_conversion_input.joinpath("test_db.db")

        # 2. Run test
        with pytest.raises(Exception) as exc_err:
            MigrateDatabaseController(Path(""))._apply_migration_script(
                _database_path, _script
            )

        # 3. Verify expectations
        assert "syntax error" in str(exc_err.value)

    @pytest.mark.parametrize(
        "orm_increment, db_increment",
        [
            pytest.param((0, 0, 0), (0, 0, 0), id="No migration"),
            pytest.param((0, 0, 1), (0, 0, 1), id="Patch migration"),
            pytest.param((0, 1, 1), (0, 1, 1), id="Minor migration"),
            pytest.param((1, 1, 1), (1, 1, 1), id="Major migration"),
            pytest.param((1, 1, 2), (1, 1, 2), id="Patch migration after major"),
        ],
    )
    def test_migrate_single_db(
        self,
        valid_conversion_input: Path,
        orm_increment: tuple[int, int, int],
        db_increment: tuple[int, int, int],
    ):
        """
        This test loops over the different conversion scripts available
        and executes some of them:
            script 0: will not be executed as the database is already this
            script 1: will be executed on a patch (creates table TestTable with record 1)
            script 2: will be executed on a minor (creates record 2)
            script 3: will be executed on a major (creates record 3)
            script 4: will never be executed as previous upgrade is a major version on which migration is interrupted
        """
        # 1. Define test data
        _migrate_db = MigrateDatabaseController(valid_conversion_input)
        _database_path = valid_conversion_input.joinpath("test_db.db")

        _db_version = DatabaseVersion(
            **_migrate_db.orm_version.__dict__, database_path=None
        )
        _query = "SELECT orm_version from Version"
        with open_database_without_compatibility_check(_database_path) as _connected_db:
            _result = _connected_db.execute_sql(_query)
            _rows = _result.fetchall()
        assert _rows[0][0] == str(_db_version)

        # 2. Run test
        _migrate_db.orm_version.major += orm_increment[0]
        _migrate_db.orm_version.minor += orm_increment[1]
        _migrate_db.orm_version.patch += orm_increment[2]
        _migrate_db.migrate_single_db(_database_path)

        # 3. Verify expectations
        _db_version.major += db_increment[0]
        _db_version.minor += db_increment[1]
        _db_version.patch += db_increment[2]
        with open_database_without_compatibility_check(_database_path) as _connected_db:
            _result = _connected_db.execute_sql(_query)
            _rows = _result.fetchall()
        assert _rows[0][0] == str(_db_version)
