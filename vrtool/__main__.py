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
    logging.info("Assess, {0}!".format(kwargs["model_directory"]))

    # Get the selected Traject.
    _vr_config = api.get_valid_vrtool_config(Path(kwargs["model_directory"]))
    api.run_step_assessment(_vr_config)


@cli.command(
    name="measures",
    help="Calculates all measures for all specified mechanisms in the model.",
)
@click.argument("model_directory", type=click.Path(exists=True), nargs=1)
def run_step_measures(**kwargs):
    logging.info("Measure, {0}!".format(kwargs["model_directory"]))

    # Define VrToolConfig and Selected Traject
    _vr_config = api.get_valid_vrtool_config(Path(kwargs["model_directory"]))
    api.run_step_measures(_vr_config)


@cli.command(
    name="optimization", help="Optimizes the model measures in the given directory."
)
@click.argument("model_directory", type=click.Path(exists=True), nargs=1)
@click.argument("measure_result_ids", type=click.INT, nargs=-1)
def run_step_optimization(**kwargs):
    logging.info("Optimize, {0}!".format(kwargs["model_directory"]))

    # Define VrToolConfig and Selected Traject
    _vr_config = api.get_valid_vrtool_config(Path(kwargs["model_directory"]))
    api.run_step_optimization(
        _vr_config, kwargs["model_directory"], kwargs.get("measure_result_ids", [])
    )


@cli.command(name="run_full", help="Full run of the model in the given directory.")
@click.argument("model_directory", type=click.Path(exists=True), nargs=1)
def run_full(**kwargs):
    logging.info("Optimize, {0}!".format(kwargs["model_directory"]))

    # Define VrToolConfig and Selected Traject
    _vr_config = api.get_valid_vrtool_config(Path(kwargs["model_directory"]))
    api.run_full(_vr_config)


if __name__ == "__main__":
    VrToolLogger.init_console_handler(logging.INFO)
    cli()
