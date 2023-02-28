import datetime
import logging
from pathlib import Path

import click

from src.defaults.vrtool_config import VrtoolConfig
from src.FloodDefenceSystem.DikeTraject import DikeTraject
from src.run_workflows.measures_workflow.run_measures import RunMeasures
from src.run_workflows.optimization_workflow.run_optimization import RunOptimization
from src.run_workflows.safety_workflow.run_safety_assessment import RunSafetyAssessment
from src.run_workflows.vrtool_plot_mode import VrToolPlotMode


@click.group()
def cli():
    pass


@cli.command(name="assessment", help="Validation of the model in the given directory.")
@click.argument("model_directory", type=click.Path(exists=True), nargs=1)
def run_step_assessment(**kwargs):
    logging.info("Assess, {0}!".format(kwargs["model_directory"]))

    # Define VrToolConfig and Selected Traject
    _vr_config = VrtoolConfig()
    _selected_traject = DikeTraject.from_vr_config(_vr_config)

    # Step 1. Safety assessment.
    _safety_assessment = RunSafetyAssessment(plot_mode=VrToolPlotMode.STANDARD)
    _safety_assessment.selected_traject = _selected_traject
    _safety_assessment.vr_config = _vr_config
    _safety_assessment.run()


@cli.command(
    name="measures",
    help="Measurement of all specified mechanisms in the model.",
)
@click.argument("model_directory", type=click.Path(exists=True), nargs=1)
def run_step_measures(**kwargs):
    logging.info("Measure, {0}!".format(kwargs["model_directory"]))

    # Define VrToolConfig and Selected Traject
    _vr_config = VrtoolConfig()
    _selected_traject = DikeTraject.from_vr_config(_vr_config)

    # Step 2. Measures.
    _measures = RunMeasures(plot_mode=VrToolPlotMode.STANDARD)
    _measures.selected_traject = _selected_traject
    _measures.vr_config = _vr_config
    _measures.run()


@cli.command(
    name="optimization", help="Optimization of the model in the given directory."
)
@click.argument("model_directory", type=click.Path(exists=True), nargs=1)
def run_step_optimization(**kwargs):
    logging.info("Optimize, {0}!".format(kwargs["model_directory"]))

    # Define VrToolConfig and Selected Traject
    _vr_config = VrtoolConfig()
    _selected_traject = DikeTraject.from_vr_config(_vr_config)

    # Step 2. Measures.
    _measures = RunMeasures(plot_mode=VrToolPlotMode.STANDARD)
    _measures.selected_traject = _selected_traject
    _measures.vr_config = _vr_config
    _measures_result = _measures.run()
    
    # Step 3. Optimization.
    _optimization = RunOptimization(_measures_result, plot_mode=_plot_mode)
    _optimization.run()

if __name__ == "__main__":
    cli()
