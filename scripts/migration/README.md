# Migration test databases

All available test databases can be migrated with `poetry run migrate_test_db`.
Make sure the right migration scripts are available in `vrtool.orm.version.migration.scripts` and the version of `vrtool.orm` is updated accordingly.