from vrtool.orm.version.migration.database_version import DatabaseVersion


class TestDatabaseVersion:
    def test_from_database(self, valid_conversion_input):
        # 1. Define test data
        _database_path = valid_conversion_input.joinpath("test_db.db")

        # 2. Execute test
        _database_version = DatabaseVersion.from_database(_database_path)

        # 3. Verify expectations
        assert isinstance(_database_version, DatabaseVersion)
        assert isinstance(_database_version.major, int)
        assert isinstance(_database_version.minor, int)
        assert isinstance(_database_version.patch, int)
        assert _database_version.database_path == _database_path
