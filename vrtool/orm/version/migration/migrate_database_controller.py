# Inspired by https://stackoverflow.com/a/19473206
import logging
import sqlite3
from pathlib import Path

from vrtool.orm.version.increment_type_enum import IncrementTypeEnum
from vrtool.orm.version.migration.database_version import DatabaseVersion
from vrtool.orm.version.migration.script_version import ScriptVersion
from vrtool.orm.version.orm_version import OrmVersion


class MigrateDatabaseController:
    orm_version: OrmVersion
    script_versions: list[ScriptVersion]
    scripts_dir: Path

    def __init__(self, scripts_dir: Path):
        self.orm_version = OrmVersion.from_orm()
        self.script_versions = self._read_scripts(scripts_dir)

    def _read_scripts(self, scripts_dir: Path) -> list[ScriptVersion]:
        return sorted(
            list(
                ScriptVersion.from_script(_script)
                for _script in scripts_dir.rglob("*.sql")
            )
        )

    @staticmethod
    def is_database_compatible(db_filepath: Path) -> bool:
        """
        Checks if the database file is compatible with the provided ORM version.

        Args:
            db_filepath (Path): Path to the database file to check.
            orm_version (OrmVersion): ORM version to check compatibility with.

        Returns:
            bool: True if the database is compatible with the ORM version, False otherwise.
        """
        _orm_version = OrmVersion.from_orm()
        _db_version = DatabaseVersion.from_database(db_filepath)
        _increment_type = _orm_version.get_increment_type(_db_version)
        return _increment_type not in (IncrementTypeEnum.MAJOR, IncrementTypeEnum.MINOR)

    def _apply_migration_script(self, db_filepath: Path, script_filepath: Path) -> None:
        """
        Apply the migration script to the database file.

        Args:
            db_filepath (Path): Path to the database file to migrate.
            script_filepath (Path): Path to the migration script to apply.
        """

        _db_connection = sqlite3.connect(db_filepath)
        logging.info(
            "Applying migration script: %s to %s", script_filepath.stem, db_filepath
        )
        try:
            _db_connection.executescript(script_filepath.read_text(encoding="utf-8"))
        except Exception as _exc_err:
            raise _exc_err
        finally:
            _db_connection.close()

    def migrate_single_db(self, db_filepath: Path):
        """
        Applies all SQL statements from a migration file to the provided
        database file.
        All available migrations scripts with a version higher than the current
        version in the database will be applied from a lower to a higher version.
        The target version is the version of the orm.

        Args:
            database_file (Path): Database file to migrate (`*.db`)

        """
        # Check the database version.
        _db_version = DatabaseVersion.from_database(db_filepath)
        if _db_version >= self.orm_version:
            logging.info(
                "Database %s heeft al een versie (%s) die gelijk of hoger is dan de VRTool (%s). Geen migratie nodig.",
                db_filepath,
                _db_version,
                self.orm_version,
            )
            return

        # Loop over the migration scripts and apply them if necessary.
        _script_version = self.orm_version
        for _script_version in filter(
            lambda x: _db_version < x <= self.orm_version, self.script_versions
        ):
            try:
                self._apply_migration_script(db_filepath, _script_version.script_path)
            except Exception as _err:
                logging.error(
                    "Er is een fout opgetreden tijdens de migratie van %s. Details: %s",
                    db_filepath,
                    _err,
                )
                break

            # Update the database version
            _db_version.update_version(_script_version)

    def migrate_databases_in_dir(self, database_dir: Path):
        """
        Migrates all existing databases in the given directory (and subdirectories)
        with the provided migration file.

        Args:
            database_dir (Path): Directory containing the databases to migrate.
        """
        for _db_to_migrate in database_dir.rglob("*.db"):
            self.migrate_single_db(_db_to_migrate)
