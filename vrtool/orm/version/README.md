# Versioning.

To ensure proper operation of `vrtool` when importing/exporting objects from/to the database, a specific version of the `orm` is given a version number (e.g. '1.2.3') according to semantic versioning. This version is updated on any change of the `orm` and unrelated to the versioning of `vrtool`.
In the context of the `orm`:
* a major release has functional impact and is NOT backwards compatible (e.g. adding a table that contains required data for a `vrtool` analysis)
* a minor release has functional impact and is backwards compatible (e.g. adding a column with a default value)
* a patch release has no functional impact and is backwards compatible (e.g. tweaking a database or table property)

## Compatibility check.
On starting the `vrtool` with a database to import/export data from/to the version of the database is checked against the version or the `orm`.
In case the version of the database is not compatible with the `orm`, the operation is terminated with an error. The user is responsible for creating databases that are compatible. A provided conversion script with instruction can support this process.
If the database is compatible, a warning or info message is issued if the minor or patch version is different, respectively. The user is encouraged to migrate the database to the current version of the `orm` with a provided conversion script.
Migration scripts are located in `orm\migration\versions`.

## Database migration.
The subpackage `migration` describes the procedure for upgrading databases to the latest version.
