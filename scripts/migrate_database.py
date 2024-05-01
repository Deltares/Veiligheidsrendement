# Inspired by https://stackoverflow.com/a/19473206
import sqlite3
from pathlib import Path
from sqlite3 import OperationalError


def apply_database_migration(db_filepath: str, sql_filepath: str):
    """
    Applies all SQL statements from a migration file to the provided
     database file.

    Can be run with `poetry run migrate_db 'db_filepath' 'sql_filepath'

    Args:
        database_file (str): Database file to migrate (`*.db`)
        migration_file (str): Migration file to apply (`*.sql`)

    """
    # Open and read the file as a single buffer
    _migration_filepath = Path(sql_filepath)
    _sql_file = _migration_filepath.read_text()
    _db_connection = sqlite3.connect(db_filepath)
    _db_cursor = _db_connection.cursor()

    # all SQL commands (split on ';')
    _sql_commands = _sql_file.split(";")

    # Execute every command from the input file
    # Show which migration will be done, expected format `v0_2_0__to__v0_3_0.sql`
    def format_version(version_value: str) -> str:
        return version_value.replace("_", ".")

    _from_version, _to_version = tuple(
        map(format_version, _migration_filepath.stem.split("__to__"))
    )
    print(f"Migrating database {db_filepath}, [{_from_version} --> {_to_version}]")
    for _command in _sql_commands:
        # This will skip and report errors
        # For example, if the tables do not yet exist, this will skip over
        # the DROP TABLE commands
        try:
            _db_cursor.execute(_command)
        except OperationalError as error_mssg:
            print("Command skipped: ", error_mssg)


def migrate_databases_in_dir(migration_dir: str, migration_file: str):
    """
    Migrates all existing databases in the given directory (and subdirectories)
    with the provided migration file.

    Args:
        migration_dir (str): Directory containing the databases to migrate.
        migration_file (str): SQL file with migration statements.
    """
    _migration_dirpath = Path(migration_dir)
    for _db_to_migrate in _migration_dirpath.rglob(".db"):
        apply_database_migration(str(_db_to_migrate), migration_file)


def migrate_test_databases():
    """
    Migrates all existing test databases (in the `tests` directory) with
    the latest available migration file.

    Can be run with `poetry run migrate_all_test_db`
    """
    # Fetch the SQL script.
    _scripts_dir = Path(__file__).parent
    _migration_file = _scripts_dir.joinpath("v0_2_0__to__v0_3_0.sql")
    assert _migration_file.exists()

    # Fetch the tests directory.
    _tests_dir = _scripts_dir.parent.joinpath("tests")
    assert _tests_dir.exists()

    # Apply migration.
    migrate_databases_in_dir(str(_scripts_dir), str(_migration_file))
