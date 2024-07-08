# Inspired by https://stackoverflow.com/a/19473206
import sqlite3
from collections import OrderedDict
from pathlib import Path

import click
import peewee

from vrtool.orm.models.version import Version as DbVersion
from vrtool.orm.orm_controllers import open_database
from vrtool.orm.version.increment_type_enum import IncrementTypeEnum
from vrtool.orm.version.orm_version import OrmVersion


@click.group()
def cli():
    pass


class MigrateDb:
    scripts_dict: OrderedDict[tuple[int, int, int], Path]
    orm_version: OrmVersion
    force_orm: bool

    def __init__(self, **kwargs):
        _scripts_dir = Path(__file__).parent.joinpath("scripts")
        self.scripts_dict = self._parse_scripts_dir(_scripts_dir)
        self.orm_version = OrmVersion(None)
        self.force_orm = False
        if "force_orm" in kwargs:
            self.force_orm = kwargs["force_orm"]

    def _parse_scripts_dir(self, scripts_dir: Path) -> dict[int, Path]:
        _scripts_dict = dict()
        for _script in scripts_dir.rglob("*.sql"):
            _version = OrmVersion.parse_version(_script.stem)
            _scripts_dict[_version] = _script
        # Ensure ordering from low to high version.
        return OrderedDict(sorted(_scripts_dict.items()))

    @staticmethod
    def apply_migration_script(db_filepath: Path, script_filepath: Path) -> None:
        """
        Apply the migration script to the database file.

        Args:
            db_filepath (Path): Path to the database file to migrate.
            script_filepath (Path): Path to the migration script to apply.
        """

        with sqlite3.connect(db_filepath) as _db_connection:
            print(f"Migrating database file with {script_filepath.stem}: {db_filepath}")
            try:
                _db_connection.executescript(
                    script_filepath.read_text(encoding="utf-8")
                )
            except Exception as exc_err:
                _db_connection.close()
                raise exc_err

    def migrate_single_db(self, db_filepath: Path):
        """
        Applies all SQL statements from a migration file to the provided
        database file.
        All available migrations scripts with a version higher than the current
        version in the database will be applied from a lower to a higher version.

        The target version is the version of the orm, unless force_orm is set to True.
        In that case the target version is the highest version available in the migration scripts.
        The orm version will then be updated according to the migration scripts.

        Can be run with `python -m migrate_db <db_filepath>`

        Args:
            database_file (Path): Database file to migrate (`*.db`)

        """

        def get_db_version() -> tuple[int, int, int]:
            with open_database(db_filepath).connection_context():
                try:
                    _version = DbVersion.select()
                    _db_version_str = _version.get().orm_version
                except peewee.OperationalError:
                    _db_version_str = "0.1.0"
            return OrmVersion.parse_version(_db_version_str)

        def set_db_version(version: tuple[int, int, int]) -> None:
            with open_database(db_filepath).connection_context():
                _version = DbVersion.get_or_none()
                _version.orm_version = OrmVersion.construct_version_string(version)
                _version.save()

        # Determine the current db version.
        _db_version = get_db_version()

        # Determine the target version.
        _orm_version = self.orm_version.read_version()
        _to_version = _orm_version
        if self.force_orm and self.scripts_dict:
            _to_version = list(self.scripts_dict.keys())[-1]

        # Loop over the migration scripts and apply them if necessary.
        _version = (0, 0, 0)
        for _version, _script in self.scripts_dict.items():
            if _db_version < _version <= _to_version:
                try:
                    self.apply_migration_script(db_filepath, _script)
                except Exception as _err:
                    print(f"Error during migration of {db_filepath}, details: {_err}")
                    break
                set_db_version(_version)
                if (
                    not self.force_orm
                    and OrmVersion.get_increment_type(_db_version, _version)
                    == IncrementTypeEnum.MAJOR
                ):
                    print(
                        "Major version upgrade detected, aborting. Please finish the migration step before continuing."
                    )
                    break

        # Update the ORM version if necessary.
        if self.force_orm:
            self.orm_version.write_version(_version)

    def migrate_databases_in_dir(self, database_dir: Path):
        """
        Migrates all existing databases in the given directory (and subdirectories)
        with the provided migration file.

        Args:
            database_dir (Path): Directory containing the databases to migrate.
        """
        for _db_to_migrate in database_dir.rglob("*.db"):
            self.migrate_single_db(_db_to_migrate)


@cli.command(
    name="migrate_db",
    help="Migrate the provided database file.",
)
@click.argument("db_filepath", type=click.Path(exists=True), nargs=1)
def migrate_db(db_filepath: str):
    """
    Can be run with `python migrate_database.py migrate_db db_filepath`
    """
    MigrateDb().migrate_single_db(Path(db_filepath))


@cli.command(
    name="migrate_db_dir",
    help="Migrates all provided database files in a directory.",
)
@click.argument("database_dir", type=click.Path(exists=True), nargs=1)
def migrate_databases_in_dir(database_dir: str):
    """
    Can be run with `python migrate_database.py migrate_db_dir database_dir`
    """
    MigrateDb().migrate_databases_in_dir(Path(database_dir))


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
    MigrateDb(force_orm=True).migrate_databases_in_dir(_tests_dir)


if __name__ == "__main__":
    cli()
