# Inspired by https://stackoverflow.com/a/19473206
import logging
import sqlite3
from pathlib import Path

from vrtool.orm.models.version import Version as DbVersion
from vrtool.orm.orm_controllers import open_database
from vrtool.orm.version.increment_type_enum import IncrementTypeEnum
from vrtool.orm.version.migration.database_version import DatabaseVersion
from vrtool.orm.version.migration.script_version import ScriptVersion
from vrtool.orm.version.orm_version import OrmVersion


class MigrateDatabaseController:
    orm_version: OrmVersion
    scripts_dir: Path
    script_versions: list[ScriptVersion]

    def __init__(self, **kwargs):
        self.orm_version = OrmVersion.from_orm()
        if "scripts_dir" in kwargs:
            self.scripts_dir = kwargs["scripts_dir"]
        else:
            self.scripts_dir = Path(__file__).parent.joinpath("scripts")
        self.script_versions = self._read_scripts()

    def _read_scripts(self) -> list[ScriptVersion]:
        return sorted(
            list(
                ScriptVersion.from_script(_script)
                for _script in self.scripts_dir.rglob("*.sql")
            )
        )

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

        def set_db_version(version: OrmVersion) -> None:
            with open_database(db_filepath).connection_context():
                _version, _ = DbVersion.get_or_create()
                _version.orm_version = str(version)
                _version.save()

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
                return

            # Check if the migration needs to be interrupted on major upgrade.
            if (
                _script_version.get_increment_type(_db_version)
                == IncrementTypeEnum.MAJOR
            ):
                logging.error(
                    "Er is een major versie upgrade detecteerd (%s); de migratie wordt afgebroken. Rond de huidige upgrade af voordat wordt doorgegaan met eventuele volgende stappen.",
                    _script_version,
                )
                break

        # Update the database version to the last executed script
        # (or orm_version if no script is executed).
        set_db_version(_script_version)

    def migrate_databases_in_dir(self, database_dir: Path):
        """
        Migrates all existing databases in the given directory (and subdirectories)
        with the provided migration file.

        Args:
            database_dir (Path): Directory containing the databases to migrate.
        """
        for _db_to_migrate in database_dir.rglob("*.db"):
            self.migrate_single_db(_db_to_migrate)


def migrate_test_databases():
    """
    Migrates all existing test databases (in the `tests` directory) to
    the latest version.
    The orm version will be updated according to the migration scripts.

    Can be run with `poetry run migrate_test_db`
    """
    # Fetch the dir containing the migration scripts.
    _root_dir = Path(__file__).parent.parent.parent.parent

    # Fetch the tests directory.
    _tests_dir = _root_dir.joinpath("tests", "test_data")
    assert _tests_dir.exists(), "No tests directory found."

    # Apply migration.
    # Force the ORM version to be upgraded according to the migration scripts.
    MigrateDatabaseController(force_orm=True).migrate_databases_in_dir(_tests_dir)
