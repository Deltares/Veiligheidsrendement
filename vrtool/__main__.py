import logging
from pathlib import Path

import click

from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.orm.orm_controllers import (
    clear_assessment_results,
    clear_measure_results,
    export_results_safety_assessment,
    export_results_measures,
    get_dike_traject,
)
from vrtool.run_workflows.measures_workflow.run_measures import RunMeasures
from vrtool.run_workflows.optimization_workflow.run_optimization import RunOptimization
from vrtool.run_workflows.safety_workflow.run_safety_assessment import (
    RunSafetyAssessment,
)
from vrtool.run_workflows.vrtool_plot_mode import VrToolPlotMode
from vrtool.run_workflows.vrtool_run_full_model import RunFullModel
from vrtool.vrtool_logger import VrToolLogger


@click.group()
def cli():
    """
    Set of general available calls for VeiligheidsrendementTool.
    """
    pass


def _get_valid_vrtool_config(model_directory: Path) -> VrtoolConfig:
    _found_json = list(model_directory.glob("*.json"))
    if not any(_found_json):
        raise FileNotFoundError(
            "No json config file found in the model directory. {}".format(
                model_directory
            )
        )

    if len(_found_json) > 1:
        raise ValueError(
            "More than one json file found in the directory {}. Only one json at the root directory supported.".format(
                model_directory
            )
        )

    _vr_config = VrtoolConfig.from_json(_found_json[0])
    if not _vr_config.input_directory:
        _vr_config.input_directory = model_directory

    if not _vr_config.output_directory:
        _vr_config.output_directory = _vr_config.input_directory / "results"

    if not _vr_config.output_directory.exists():
        _vr_config.output_directory.mkdir(parents=True)

    return _vr_config


@cli.command(name="assessment", help="Assesses the model in the given directory.")
@click.argument("model_directory", type=click.Path(exists=True), nargs=1)
def run_step_assessment(**kwargs):
    logging.info("Assess, {0}!".format(kwargs["model_directory"]))

    # Get the selected Traject.
    _vr_config = _get_valid_vrtool_config(Path(kwargs["model_directory"]))
    _selected_traject = get_dike_traject(_vr_config)

    # Clear the results
    clear_assessment_results(_vr_config)

    # Step 1. Safety assessment.
    _safety_assessment = RunSafetyAssessment(
        _vr_config, _selected_traject, plot_mode=VrToolPlotMode.STANDARD
    )
    _result = _safety_assessment.run()

    # Export the results.
    export_results_safety_assessment(_result)


@cli.command(
    name="measures",
    help="Calculates all measures for all specified mechanisms in the model.",
)
@click.argument("model_directory", type=click.Path(exists=True), nargs=1)
def run_step_measures(**kwargs):
    logging.info("Measure, {0}!".format(kwargs["model_directory"]))

    # Define VrToolConfig and Selected Traject
    _vr_config = _get_valid_vrtool_config(Path(kwargs["model_directory"]))
    _selected_traject = get_dike_traject(_vr_config)

    # Clear the results
    clear_assessment_results(
        _vr_config
    )  # Assessment results also cleared because it is part of the RunMeasures workflow
    clear_measure_results(_vr_config)

    # Step 2a. Measures.
    _measures = RunMeasures(
        _vr_config, _selected_traject, plot_mode=VrToolPlotMode.STANDARD
    )
    _measures_result = _measures.run()

    # Step 2b. Export solutions to database
    export_results_measures(_measures_result)


@cli.command(
    name="optimization", help="Optimizes the model measures in the given directory."
)
@click.argument("model_directory", type=click.Path(exists=True), nargs=1)
def run_step_optimization(**kwargs):
    logging.info("Optimize, {0}!".format(kwargs["model_directory"]))

    # Define VrToolConfig and Selected Traject
    _vr_config = _get_valid_vrtool_config(Path(kwargs["model_directory"]))
    _selected_traject = get_dike_traject(_vr_config)
    _plot_mode = VrToolPlotMode.STANDARD

    # Step 2. Measures.
    _measures = RunMeasures(_vr_config, _selected_traject, _plot_mode)
    _measures_result = _measures.run()

    # Step 3. Optimization.
    _optimization = RunOptimization(_measures_result, _plot_mode)
    _optimization.run()


@cli.command(name="run_full", help="Full run of the model in the given directory.")
@click.argument("model_directory", type=click.Path(exists=True), nargs=1)
def run_full(**kwargs):
    logging.info("Optimize, {0}!".format(kwargs["model_directory"]))

    # Define VrToolConfig and Selected Traject
    _vr_config = _get_valid_vrtool_config(Path(kwargs["model_directory"]))
    _selected_traject = get_dike_traject(_vr_config)

    # Run all steps with one command.
    _full_model = RunFullModel(_vr_config, _selected_traject, VrToolPlotMode.STANDARD)
    _full_model.run()


if __name__ == "__main__":
    VrToolLogger.init_console_handler(logging.INFO)
    cli()
