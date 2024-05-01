# Inspired by https://stackoverflow.com/a/19473206
import sqlite3
from pathlib import Path


def apply_database_migration(db_filepath: str, sql_filepath: str):
    """
    Applies all SQL statements from a migration file to the provided
     database file.

    Can be run with `poetry run migrate_db 'db_filepath' 'sql_filepath'

    Args:
        database_file (str): Database file to migrate (`*.db`)
        migration_file (str): Migration file to apply (`*.sql`)

    """
    _migration_filepath = Path(sql_filepath)

    # Show which migration will be done, expected format `v0_2_0__to__v0_3_0.sql`
    def format_version(version_value: str) -> str:
        return version_value.replace("_", ".")

    _from_version, _to_version = tuple(
        map(format_version, _migration_filepath.stem.split("__to__"))
    )
    print(f"Migrating database file [{_from_version} to {_to_version}]: {db_filepath}")

    # Open and read the file as a single buffer
    with sqlite3.connect(db_filepath) as _db_connection:
        _db_connection.executescript(_migration_filepath.read_text(encoding="utf-8"))


def migrate_databases_in_dir(database_dir: str, sql_file: str):
    """
    Migrates all existing databases in the given directory (and subdirectories)
    with the provided migration file.

    Can be run with `poetry run migrate_db_dir`

    Args:
        database_dir (str): Directory containing the databases to migrate.
        sql_file (str): SQL file with migration statements.
    """
    _migration_dirpath = Path(database_dir)
    for _db_to_migrate in _migration_dirpath.rglob("*.db"):
        apply_database_migration(str(_db_to_migrate), sql_file)


def migrate_test_databases():
    """
    Migrates all existing test databases (in the `tests` directory) with
    the latest available migration file.

    Can be run with `poetry run migrate_test_db`
    """
    # Fetch the SQL script.
    _scripts_dir = Path(__file__).parent
    _migration_file = _scripts_dir.joinpath("v0_2_0__to__v0_3_0.sql")
    assert _migration_file.exists(), "No migration file found."

    # Fetch the tests directory.
    _tests_dir = _scripts_dir.parent.joinpath("tests")
    assert _tests_dir.exists(), "No tests directory found."

    # Apply migration.
    migrate_databases_in_dir(str(_tests_dir), str(_migration_file))


if __name__ == "__main__":
    migrate_test_databases()
