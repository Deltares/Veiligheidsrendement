import logging
from pathlib import Path

import click

from vrtool import api
from vrtool.vrtool_logger import VrToolLogger


@click.group()
def cli():
    """
    Set of general available calls for VeiligheidsrendementTool.
    """
    pass


@cli.command(name="assessment", help="Assesses the model in the given directory.")
@click.argument("model_directory", type=click.Path(exists=True), nargs=1)
def run_step_assessment(**kwargs):
    logging.info("Start beoordeling voor database in {0}".format(kwargs["model_directory"]))

    # Get the selected Traject.
    _vr_config = api.get_valid_vrtool_config(Path(kwargs["model_directory"]))
    api.run_step_assessment(_vr_config)


@cli.command(
    name="measures",
    help="Berekening voor betrouwbaarheid en kosten voor alle maatregelen.",
)
@click.argument("model_directory", type=click.Path(exists=True), nargs=1)
def run_step_measures(**kwargs):
    logging.info("Start berekenen betrouwbaarheid en kosten maatregelen voor database in {0}".format(kwargs["model_directory"]))

    # Define VrToolConfig and Selected Traject
    _vr_config = api.get_valid_vrtool_config(Path(kwargs["model_directory"]))
    api.run_step_measures(_vr_config)


@cli.command(
    name="optimization", help="Optimizes the model measures in the given directory."
)
@click.argument("model_directory", type=click.Path(exists=True), nargs=1)
@click.argument("measure_result_ids", type=click.INT, nargs=-1)
def run_step_optimization(**kwargs):
    logging.info("Start optimalisatie voor bestanden in {0}".format(kwargs["model_directory"]))

    # Define VrToolConfig and Selected Traject
    _vr_config = api.get_valid_vrtool_config(Path(kwargs["model_directory"]))
    api.run_step_optimization(
        _vr_config, kwargs["model_directory"], kwargs.get("measure_result_ids", [])
    )


@cli.command(name="run_full", help="Full run of the model in the given directory.")
@click.argument("model_directory", type=click.Path(exists=True), nargs=1)
def run_full(**kwargs):
    logging.info("Start volledige berekening voor bestanden in {0}!".format(kwargs["model_directory"]))

    # Define VrToolConfig and Selected Traject
    _vr_config = api.get_valid_vrtool_config(Path(kwargs["model_directory"]))
    api.run_full(_vr_config)


if __name__ == "__main__":
    VrToolLogger.init_console_handler(logging.INFO)
    cli()
