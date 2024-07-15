# Database migration

This subproject contains the logic for upgrading a database to the latest version of the `vrtool.orm`.

## Versions

Folder `scripts` contains scripts to upgrade to a specific version from the previous version.
E.g. `v0_1_1.sql` migrates a database from version 0.1.0 to 0.1.1.

## Migration

A database can be migrated with `python vrtool migrate_db <path to database>`.
Alternatively databases in a certain folder can be converted with `python vrtool migrate_db_dir <path to folder with database(s)>`.
If a database is upgraded multiple versions, the different migrations are done consecutively.

**DISCLAIMER**: Databases are overwritten during migration, so it is advised to create a backup beforehand.
