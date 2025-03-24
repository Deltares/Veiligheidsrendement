import logging
from datetime import datetime
from pathlib import Path

import click

from vrtool import api
from vrtool.orm.version.migration import default_scripts_dir
from vrtool.orm.version.migration.migrate_database_controller import (
    MigrateDatabaseController,
)
from vrtool.vrtool_logger import VrToolLogger


@click.group()
def cli():
    """
    Set of general available calls for VeiligheidsrendementTool.
    """
    pass


def _initialize_log_file(log_dir: click.Path | None):
    # Logging dir.
    if log_dir is None:
        log_dir = Path.cwd()

    # Define logging filename and initialize handler
    _current_date = datetime.today().strftime("%Y%m%d_%I%M")
    _log_file = Path(log_dir).joinpath(f"vrtool_logging_{_current_date}.log")
    VrToolLogger.init_file_handler(_log_file, logging_level=logging.INFO)
    logging.info("Start logging vanuit %s", str(_log_file))


@cli.command(
    name="assessment", help="Assesses the model with the given configuration file."
)
@click.argument("config_file", type=click.Path(exists=True), nargs=1)
@click.option("-ld", "--log-dir", type=click.Path())
def run_step_assessment(config_file: click.Path, log_dir: click.Path | None):
    """
    Runs the step assessment.
    """
    # Retrieve parameter and initialize logging.
    _initialize_log_file(log_dir)

    logging.info("Start beoordeling met configuratie %s", str(config_file))

    # Get the selected Traject.
    _vr_config = api.get_valid_vrtool_config(Path(config_file))
    api.run_step_assessment(_vr_config)


@cli.command(
    name="measures",
    help="Calculates the reliability and cost for all measures with the given configuration file.",
)
@click.argument("config_file", type=click.Path(exists=True), nargs=1)
@click.option("-ld", "--log-dir", type=click.Path())
def run_step_measures(config_file: click.Path, log_dir: click.Path | None):
    """
    Runs step measures.
    """
    # Retrieve parameter and initialize logging.
    _initialize_log_file(log_dir)

    logging.info(
        "Start berekenen betrouwbaarheid en kosten maatregelen met configuratie %s",
        str(config_file),
    )

    # Define VrToolConfig and Selected Traject
    _vr_config = api.get_valid_vrtool_config(Path(config_file))
    api.run_step_measures(_vr_config)


@cli.command(
    name="optimization",
    help="Optimizes the model measures with the given configuration file.",
)
@click.argument("config_file", type=click.Path(exists=True), nargs=1)
@click.argument("measure_result_ids", type=click.INT, nargs=-1)
@click.option("-ld", "--log-dir", type=click.Path())
def run_step_optimization(
    config_file: click.Path, log_dir: click.Path | None, measure_result_ids: tuple[int]
):
    """
    Runs step optimization.
    """
    # Retrieve parameter and initialize logging.
    _initialize_log_file(log_dir)

    _config_file = Path(config_file)
    logging.info("Start optimalisatie met configuratie %s", str(config_file))

    # Define VrToolConfig and Selected Traject
    _vr_config = api.get_valid_vrtool_config(_config_file)
    _measure_result_tuples = []
    if any(measure_result_ids):
        _iterator = iter(measure_result_ids)
        _measure_result_tuples = list(zip(_iterator, _iterator))
    api.run_step_optimization(_vr_config, _config_file.parent, _measure_result_tuples)


@cli.command(
    name="run_full", help="Full run of the model with the given configuration."
)
@click.argument("config_file", type=click.Path(exists=True), nargs=1)
@click.option("-ld", "--log-dir", type=click.Path())
def run_full(config_file: click.Path, log_dir: click.Path | None):
    """
    Runs all the veiligheidsrendement steps (assessment, measures and optimization).
    """
    # Retrieve parameter and initialize logging.
    _initialize_log_file(log_dir)
    logging.info("Start volledige berekening met configuratie %s!", str(config_file))

    # Define VrToolConfig and Selected Traject
    _vr_config = api.get_valid_vrtool_config(Path(config_file))
    api.run_full(_vr_config)


@cli.command(name="migrate_db", help="Migrate the provided database file.")
@click.argument("db_filepath", type=click.Path(exists=True), nargs=1)
def migrate_db(db_filepath: str):
    """
    Migrates the provided database file to the latest version possible.

    Args:
        db_filepath (str): Database file location to migrate.
    """
    logging.info("Migreren van database %s.", db_filepath)
    MigrateDatabaseController(default_scripts_dir).migrate_single_db(Path(db_filepath))


@cli.command(
    name="migrate_db_dir", help="Migrates all provided database files in a directory."
)
@click.argument("database_dir", type=click.Path(exists=True), nargs=1)
def migrate_databases_in_dir(database_dir: str):
    """
    Migrates all the database files within a given directory.

    Args:
        database_dir (str): Directory path location.
    """
    logging.info("Migreren van databases in %s.", database_dir)
    MigrateDatabaseController(default_scripts_dir).migrate_databases_in_dir(
        Path(database_dir)
    )


if __name__ == "__main__":
    VrToolLogger.init_console_handler(logging.INFO)
    cli()
