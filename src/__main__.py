import datetime
import logging
from pathlib import Path

import click


@click.group()
def cli():
    pass


@cli.command(name="assessment", help="Validation of the model in the given directory.")
@click.argument("model_directory", type=click.Path(exists=True), nargs=1)
def run_step_assessment(**kwargs):
    logging.info("Assess, {0}!".format(kwargs["model_directory"]))


@cli.command(
    name="measures",
    help="Measurement of all specified mechanisms in the model.",
)
@click.argument("model_directory", type=click.Path(exists=True), nargs=1)
def run_step_measures(**kwargs):
    logging.info("Measure, {0}!".format(kwargs["model_directory"]))


@cli.command(
    name="optimization", help="Optimization of the model in the given directory."
)
@click.argument("model_directory", type=click.Path(exists=True), nargs=1)
def run_step_optimization(**kwargs):
    logging.info("Optimize, {0}!".format(kwargs["model_directory"]))


if __name__ == "__main__":
    cli()
