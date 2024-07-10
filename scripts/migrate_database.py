# Inspired by https://stackoverflow.com/a/19473206
import sqlite3
from pathlib import Path

import click


@click.group()
def cli():
    pass


class MigrateDb:
    def migrate_single_db(self, db_filepath: Path, sql_filepath: Path):
        """
        Applies all SQL statements from a migration file to the provided
        database file.

        Can be run with `python -m migrate_db db_filepath sql_filepath`

        Args:
            database_file (Path): Database file to migrate (`*.db`)
            migration_file (Path): Migration file to apply (`*.sql`)

        """
        # Show which migration will be done, expected format `v0_2_0__to__v0_3_0.sql`
        def format_version(version_value: str) -> str:
            return version_value.replace("_", ".")

        _from_version, _to_version = tuple(
            map(format_version, sql_filepath.stem.split("__to__"))
        )

        # Open and read the file as a single buffer
        try:
            with sqlite3.connect(db_filepath) as _db_connection:
                print(
                    f"Migrating database file [{_from_version} to {_to_version}]: {db_filepath}"
                )
                _db_connection.executescript(sql_filepath.read_text(encoding="utf-8"))
        except Exception as _err:
            print(f"Error during migration of {db_filepath}, details: {_err}")

    def migrate_databases_in_dir(self, database_dir: Path, sql_file: Path):
        """
        Migrates all existing databases in the given directory (and subdirectories)
        with the provided migration file.

        Args:
            database_dir (Path): Directory containing the databases to migrate.
            sql_file (Path): SQL file with migration statements.
        """
        for _db_to_migrate in database_dir.rglob("*.db"):
            self.migrate_single_db(_db_to_migrate, sql_file)


@cli.command(
    name="migrate_db",
    help="Applies all SQL statements from a migration file to the provided database file.",
)
@click.argument("db_filepath", type=click.Path(exists=True), nargs=1)
@click.argument("sql_filepath", type=click.Path(exists=True), nargs=1)
def migrate_db(db_filepath: str, sql_filepath: str):
    """
    Can be run with `python migrate_database.py migrate_db db_filepath sql_filepath`
    """
    MigrateDb().migrate_single_db(Path(db_filepath), Path(sql_filepath))


@cli.command(
    name="migrate_db_dir",
    help="Applies all SQL statements from a migration file to the provided database files in a directory.",
)
@click.argument("database_dir", type=click.Path(exists=True), nargs=1)
@click.argument("sql_file", type=click.Path(exists=True), nargs=1)
def migrate_databases_in_dir(database_dir: str, sql_file: str):
    """
    Can be run with `python migrate_database.py migrate_db_dir database_dir sql_file`
    """
    MigrateDb().migrate_databases_in_dir(Path(database_dir), Path(sql_file))


def migrate_test_databases():
    """
    Migrates all existing test databases (in the `tests` directory) with
    the latest available migration file.

    Can be run with `poetry run migrate_test_db`
    """
    # Fetch the SQL script.
    _scripts_dir = Path(__file__).parent
    _migration_file = _scripts_dir.joinpath("v0_3_0__to__v0_3_2.sql")
    assert _migration_file.exists(), "No migration file found."

    # Fetch the tests directory.
    _tests_dir = _scripts_dir.parent.joinpath("tests", "test_data")
    assert _tests_dir.exists(), "No tests directory found."

    # Apply migration.
    MigrateDb().migrate_databases_in_dir(_tests_dir, _migration_file)


if __name__ == "__main__":
    cli()
