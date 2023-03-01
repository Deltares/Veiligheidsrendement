import logging
from pathlib import Path
import click

from src.defaults.vrtool_config import VrtoolConfig
from src.FloodDefenceSystem.DikeTraject import DikeTraject
from src.run_workflows.measures_workflow.run_measures import RunMeasures
from src.run_workflows.optimization_workflow.run_optimization import RunOptimization
from src.run_workflows.safety_workflow.run_safety_assessment import RunSafetyAssessment
from src.run_workflows.vrtool_plot_mode import VrToolPlotMode
from src.run_workflows.vrtool_run_full_model import RunFullModel

@click.group()
def cli():
    pass

def _get_valid_vrtool_config(args_dict: dict) -> VrtoolConfig:
    _vr_config = VrtoolConfig()
    _vr_config.input_directory = Path(args_dict["model_directory"])
    _vr_config.traject = args_dict["traject"]
    if not _vr_config.output_directory:
        _vr_config.output_directory = _vr_config.input_directory / "results"
    return _vr_config

@cli.command(name="assessment", help="Validation of the model in the given directory.")
@click.argument("model_directory", type=click.Path(exists=True), nargs=1)
@click.argument("traject", type=str, nargs=1)
def run_step_assessment(**kwargs):
    logging.info("Assess, {0}!".format(kwargs["model_directory"]))

    # Define VrToolConfig and Selected Traject
    _vr_config = _get_valid_vrtool_config(kwargs)
    _selected_traject = DikeTraject.from_vr_config(_vr_config)

    # Step 1. Safety assessment.
    _safety_assessment = RunSafetyAssessment(_vr_config, _selected_traject, plot_mode=VrToolPlotMode.STANDARD)
    _safety_assessment.run()


@cli.command(
    name="measures",
    help="Measurement of all specified mechanisms in the model.",
)
@click.argument("model_directory", type=click.Path(exists=True), nargs=1)
@click.argument("traject", type=str, nargs=1)
def run_step_measures(**kwargs):
    logging.info("Measure, {0}!".format(kwargs["model_directory"]))

    # Define VrToolConfig and Selected Traject
    _vr_config = _get_valid_vrtool_config(kwargs)
    _selected_traject = DikeTraject.from_vr_config(_vr_config)

    # Step 2. Measures.
    _measures = RunMeasures(_vr_config, _selected_traject, plot_mode=VrToolPlotMode.STANDARD)
    _measures.run()


@cli.command(
    name="optimization", help="Optimization of the model in the given directory."
)
@click.argument("model_directory", type=click.Path(exists=True), nargs=1)
@click.argument("traject", type=str, nargs=1)
def run_step_optimization(**kwargs):
    logging.info("Optimize, {0}!".format(kwargs["model_directory"]))

    # Define VrToolConfig and Selected Traject
    _vr_config = _get_valid_vrtool_config(kwargs)
    _selected_traject = DikeTraject.from_vr_config(_vr_config)
    _plot_mode = VrToolPlotMode.STANDARD

    # Step 2. Measures.
    _measures = RunMeasures(_vr_config, _selected_traject, _plot_mode)
    _measures_result = _measures.run()
    
    # Step 3. Optimization.
    _optimization = RunOptimization(_measures_result, _plot_mode)
    _optimization.run()

@cli.command(
    name="run_full", help="Full run of the model in the given directory."
)
@click.argument("model_directory", type=click.Path(exists=True), nargs=1)
@click.argument("traject", type=str, nargs=1)
def run_full(**kwargs):
    logging.info("Optimize, {0}!".format(kwargs["model_directory"]))

    # Define VrToolConfig and Selected Traject
    _vr_config = _get_valid_vrtool_config(kwargs)
    _selected_traject = DikeTraject.from_vr_config(_vr_config)

    # Run all steps with one command.
    _measures = RunFullModel(_vr_config, _selected_traject, VrToolPlotMode.STANDARD)
    _measures.run()

if __name__ == "__main__":
    cli()
