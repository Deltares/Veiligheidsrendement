# Database migration

This subproject contains the logic for upgrading a database to the latest version of the `vrtool.orm`.

## Versions

Folder `scripts` contains scripts to upgrade to a specific version from the previous version.
E.g. `v0_1_1.sql` migrates a database from version 0.1.0 to 0.1.1.

## Migration

From a user perspective a database can be migrated with `python vrtool migrate_db <path to database>`.
Alternatively databases in a certain folder can be converted with `python vrtool migrate_db_dir <path to folder with database(s)>`.
If a database is upgraded multiple versions, the different migrations are done consecutively.

### Major upgrade
A major upgrade is assumed to be non-compatible with the previous version, meaning the migration can't be fully supported.
In case a major upgrade is done, the migration is interrupted after executing the supported part of the migration.
To have a compatible database, the user has to finish the migration step by adding the right data to the database before continuing with the remaining migration steps.
Instruction on finishing a major upgrade can be found in a file with the same name as the migration script, but with extension `.txt`.
So next to `v1_0_0.sql` the instruction can be found in `v1_0_0.txt`.

### Test databases
From a developer perspective all available test databases can be migrated with `poetry run migrate_test_db`.
In this case the `orm` version of the `vrtool` will be updated according to the latest available migration script.
