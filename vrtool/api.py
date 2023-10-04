from vrtool.defaults.vrtool_config import VrtoolConfig
from pathlib import Path
from vrtool.orm.orm_controllers import (
    clear_assessment_results,
    clear_measure_results,
    export_results_safety_assessment,
    export_solutions,
    get_dike_traject,
)
from vrtool.run_workflows.measures_workflow.run_measures import RunMeasures
from vrtool.run_workflows.optimization_workflow.run_optimization import RunOptimization
from vrtool.run_workflows.safety_workflow.run_safety_assessment import (
    RunSafetyAssessment,
)
from vrtool.run_workflows.vrtool_run_full_model import RunFullModel
from vrtool.vrtool_logger import VrToolLogger


def get_valid_vrtool_config(model_directory: Path) -> VrtoolConfig:
    _found_json = list(model_directory.glob("*.json"))
    if not any(_found_json):
        raise FileNotFoundError(
            "No json config file found in the model directory {}.".format(
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


def run_step_assessment(vrtool_config: VrtoolConfig) -> None:
    """
    Runs a "Safety Assessment" based on the provided configuration and exports
    its results to the database.

    Args:
        vrtool_config (VrtoolConfig): Configuration to use during run.
    """
    _selected_traject = get_dike_traject(vrtool_config)

    # Clear the results
    clear_assessment_results(vrtool_config)

    # Step 1. Safety assessment.
    _safety_assessment = RunSafetyAssessment(vrtool_config, _selected_traject)
    _result = _safety_assessment.run()

    # Export the results.
    export_results_safety_assessment(_result)

def run_step_measures(vrtool_config: VrtoolConfig) -> None:
    """
    Runs a "Measures assessment" based on the provided configuration and exports
    its results to the database.

    Args:
        vrtool_config (VrtoolConfig): Configuration to use during run.
    """
    _selected_traject = get_dike_traject(vrtool_config)

    # Clear the results
    clear_assessment_results(
        vrtool_config
    )
    # Assessment results also cleared because it is part of the RunMeasures workflow
    clear_measure_results(vrtool_config)

    # Step 2a. Measures.
    _measures = RunMeasures(vrtool_config, _selected_traject)
    _measures_result = _measures.run()

    # Step 2b. Export solutions to database
    export_solutions(_measures_result)

def run_step_optimization(vrtool_config: VrtoolConfig) -> None:
    """
    Runs an optimization by assessing and then optimizing the available measures
    in the database. The results are then exported into the database.

    Args:
        vrtool_config (VrtoolConfig): Configuration to use during run.
    """
    _selected_traject = get_dike_traject(vrtool_config)

    # Step 2. Measures.
    _measures = RunMeasures(vrtool_config, _selected_traject)
    _measures_result = _measures.run()

    # Step 3. Optimization.
    _optimization = RunOptimization(_measures_result)
    _optimization.run()

def run_full(vrtool_config: VrtoolConfig) -> None:
    """
    Full run of the model in the database that triggers all the available workflows
    (safety - measures - optimization). The results are consequently exported.

    Args:
        vrtool_config (VrtoolConfig): Configuration to use during run.
    """
    _selected_traject = get_dike_traject(vrtool_config)

    # Run all steps with one command.
    _full_model = RunFullModel(vrtool_config, _selected_traject)
    _full_model.run()