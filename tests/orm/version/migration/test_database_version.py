from pathlib import Path

from vrtool.orm.orm_db import vrtool_db
from vrtool.orm.version.migration.database_version import DatabaseVersion


class TestDatabaseVersion:
    def test_from_database(self, empty_db_path: Path):
        # 1. Execute test
        _database_version = DatabaseVersion.from_database(empty_db_path)

        # 2. Verify expectations
        assert isinstance(_database_version, DatabaseVersion)
        assert isinstance(_database_version.major, int)
        assert isinstance(_database_version.minor, int)
        assert isinstance(_database_version.patch, int)
        assert _database_version.database_path == empty_db_path

    def test_update_version(self, empty_db_path: Path):
        # 1. Define test data
        _from_db_version = DatabaseVersion.from_database(empty_db_path)
        _to_db_version = DatabaseVersion(
            major=_from_db_version.major + 1,
            minor=_from_db_version.minor + 1,
            patch=_from_db_version.patch + 1,
            database_path=_from_db_version.database_path,
        )
        assert _from_db_version != _to_db_version

        # 2. Execute test
        _from_db_version.update_version(_to_db_version)

        # 3. Verify expectations
        assert _from_db_version == _to_db_version
        assert vrtool_db.is_closed()
