from pathlib import Path

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
        _db_version = DatabaseVersion.from_database(empty_db_path)
        _db_version.major += 1
        _db_version.minor += 1
        _db_version.patch += 1

        # 2. Execute test
        _db_version.update_version(_db_version)

        # 3. Verify expectations
        _updated_version = DatabaseVersion.from_database(empty_db_path)
        assert _updated_version == _db_version
