import logging
from datetime import datetime
from pathlib import Path

import click

from vrtool import __version__, api
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.orm.version.migration import default_scripts_dir
from vrtool.orm.version.migration.migrate_database_controller import (
    MigrateDatabaseController,
)
from vrtool.vrtool_logger import VrToolLogger

__externals_path: Path | None = None


@click.group()
@click.version_option(__version__)
@click.option(
    "--externals", type=click.Path(path_type=Path), help="Path to externals directory."
)
def cli(externals: Path | None):
    """
    Set of general available calls for VeiligheidsrendementTool.
    """
    global __externals_path
    __externals_path = externals if externals else None


def _initialize_log_file(log_dir: Path):
    if log_dir is None:
        raise ValueError("Log directory cannot be None.")

    # Define logging filename and initialize handler
    _current_date = datetime.today().strftime("%Y%m%d_%H%M")
    _log_file = log_dir.joinpath(f"vrtool_{_current_date}.log")
    VrToolLogger.init_file_handler(_log_file, logging_level=logging.INFO)
    logging.info("Start logging vanuit %s", _log_file)


def _set_externals_path(vr_config: VrtoolConfig):
    logging.info("Config externals path: %s.", vr_config.externals)
    logging.info("CLI externals path: %s.", __externals_path)
    if vr_config.externals is not None or __externals_path is None:
        logging.info("No externals path set, using config value.")
        return
    logging.info("Setting externals path from CLI parameter.")
    vr_config.externals = __externals_path


@cli.command(
    name="assessment", help="Assesses the model with the given configuration file."
)
@click.argument("config_file", type=click.Path(exists=True, path_type=Path), nargs=1)
@click.option("-ld", "--log-dir", type=click.Path(path_type=Path))
def run_step_assessment(config_file: Path, log_dir: Path | None):
    """
    Runs the step assessment.
    """
    # Retrieve parameter and initialize logging.
    if log_dir is None:
        log_dir = config_file.parent
    _initialize_log_file(log_dir)

    logging.info("Start beoordeling met configuratie %s", config_file)

    # Get the selected Traject.
    _vr_config = api.get_valid_vrtool_config(config_file)
    _set_externals_path(_vr_config)
    api.run_step_assessment(_vr_config)


@cli.command(
    name="measures",
    help="Calculates the reliability and cost for all measures with the given configuration file.",
)
@click.argument("config_file", type=click.Path(exists=True, path_type=Path), nargs=1)
@click.option("-ld", "--log-dir", type=click.Path(path_type=Path))
def run_step_measures(config_file: Path, log_dir: Path | None):
    """
    Runs step measures.
    """

    # Retrieve parameter and initialize logging.
    if log_dir is None:
        log_dir = config_file.parent
    _initialize_log_file(log_dir)

    logging.info(
        "Start berekenen betrouwbaarheid en kosten maatregelen met configuratie %s",
        config_file,
    )

    # Define VrToolConfig and Selected Traject
    _vr_config = api.get_valid_vrtool_config(config_file)
    _set_externals_path(_vr_config)
    api.run_step_measures(_vr_config)


@cli.command(
    name="optimization",
    help="Optimizes the model measures with the given configuration file.",
)
@click.argument("config_file", type=click.Path(exists=True, path_type=Path), nargs=1)
@click.argument("measure_result_ids", type=click.INT, nargs=-1)
@click.option("-ld", "--log-dir", type=click.Path(path_type=Path))
def run_step_optimization(
    config_file: Path, log_dir: Path | None, measure_result_ids: tuple[int]
):
    """
    Runs step optimization.
    """
    # Retrieve parameter and initialize logging.
    if log_dir is None:
        log_dir = config_file.parent
    _initialize_log_file(log_dir)

    logging.info("Start optimalisatie met configuratie %s", config_file)

    # Define VrToolConfig and Selected Traject
    _vr_config = api.get_valid_vrtool_config(config_file)
    _set_externals_path(_vr_config)
    _measure_result_tuples = []
    if any(measure_result_ids):
        _iterator = iter(measure_result_ids)
        _measure_result_tuples = list(zip(_iterator, _iterator))
    api.run_step_optimization(_vr_config, config_file.parent, _measure_result_tuples)


@cli.command(
    name="run_full", help="Full run of the model with the given configuration."
)
@click.argument("config_file", type=click.Path(exists=True, path_type=Path), nargs=1)
@click.option("-ld", "--log-dir", type=click.Path(path_type=Path))
def run_full(config_file: Path, log_dir: Path | None):
    """
    Runs all the veiligheidsrendement steps (assessment, measures and optimization).
    """
    # Retrieve parameter and initialize logging.
    if log_dir is None:
        log_dir = config_file.parent
    _initialize_log_file(log_dir)
    logging.info("Start volledige berekening met configuratie %s!", config_file)

    # Define VrToolConfig and Selected Traject
    _vr_config = api.get_valid_vrtool_config(config_file)
    _set_externals_path(_vr_config)
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
    MigrateDatabaseController(default_scripts_dir).migrate_single_db(db_filepath)


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
        database_dir
    )


if __name__ == "__main__":
    VrToolLogger.init_console_handler(logging.INFO)
    cli()
