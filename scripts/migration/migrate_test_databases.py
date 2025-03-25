import logging
from pathlib import Path

from vrtool.orm.version.migration import default_scripts_dir
from vrtool.orm.version.migration.migrate_database_controller import (
    MigrateDatabaseController,
)
from vrtool.vrtool_logger import VrToolLogger


def migrate_test_databases():
    """
    Migrates all existing test databases (in the `tests` directory) to
    the latest version.
    The orm version will be updated according to the migration scripts.

    Can be run with `poetry run migrate_test_db`
    """
    # Fetch the dir containing the migration scripts.
    _root_dir = Path(__file__).parent.parent.parent

    # Fetch the tests directory.
    _tests_dir = _root_dir.joinpath("tests", "test_data")
    assert _tests_dir.exists(), "No tests directory found."

    # Apply migration.
    VrToolLogger.init_console_handler(logging.INFO)
    logging.info("Migrating test databases in %s.", _tests_dir)
    MigrateDatabaseController(default_scripts_dir).migrate_databases_in_dir(_tests_dir)
