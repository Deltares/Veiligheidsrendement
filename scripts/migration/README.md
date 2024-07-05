# Database migration

This subproject contains the logic for database migrations between different versions of the `ra2ce.orm`.

## Versions

Folder `versions` contains scripts to upgrade to a specific version from the previous version.
E.g. `v0_1_1.sql` migrates a database from version 0.1.0 to 0.1.1.

## Migration

From a user perspective a database can be migrated with `python migrate_database.py <path to database>`.
If a database is migrated multiple versions, the different migrations are done consequetively.

### Test databases
From a developer perspective all available test databases can be migrated with `poetry run migrate_test_db`.
In this case the `orm` version of the `vrtool` will be updated according to the latest available migration script.
