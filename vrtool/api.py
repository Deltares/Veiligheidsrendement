import logging
from datetime import datetime
from pathlib import Path

from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.orm.orm_controllers import (
    clear_assessment_results,
    clear_measure_results,
    clear_optimization_results,
    create_basic_optimization_run,
    create_optimization_run_for_selected_measures,
    export_results_measures,
    export_results_optimization,
    export_results_safety_assessment,
    get_dike_traject,
)
from vrtool.run_workflows.measures_workflow.results_measures import ResultsMeasures
from vrtool.run_workflows.measures_workflow.run_measures import RunMeasures
from vrtool.run_workflows.optimization_workflow.results_optimization import (
    ResultsOptimization,
)
from vrtool.run_workflows.optimization_workflow.run_optimization import RunOptimization
from vrtool.run_workflows.safety_workflow.results_safety_assessment import (
    ResultsSafetyAssessment,
)
from vrtool.run_workflows.safety_workflow.run_safety_assessment import (
    RunSafetyAssessment,
)


def get_valid_vrtool_config(model_directory: Path) -> VrtoolConfig:
    """
    Gets the location of a valid `VrtoolConfig` file within the provided directory.

    Args:
        model_directory (Path): Directory containing a `.json` VRTOOL config file.

    Raises:
        FileNotFoundError: When no `.json` config file was found.
        ValueError: When more than one `.json` config file is present.

    Returns:
        VrtoolConfig: Configuration file representing the model in the given directory.
    """
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
    ApiRunWorkflows(vrtool_config).run_assessment()


def run_step_measures(vrtool_config: VrtoolConfig) -> None:
    """
    Runs a "Measures assessment" based on the provided configuration and exports
    its results to the database.

    Args:
        vrtool_config (VrtoolConfig): Configuration to use during run.
    """
    ApiRunWorkflows(vrtool_config).run_measures()


def run_step_optimization(
    vrtool_config: VrtoolConfig, measure_results_ids: dict[int,int]
) -> None:
    """
    Runs an optimization by optimizing the available measures
    in the database. The results are then exported into the database.

    Args:
        vrtool_config (VrtoolConfig): Configuration to use during run.
        measure_results_ids (list[int]): List of id's for the selected `MeasureResult` entries to use.
    """
    ApiRunWorkflows(vrtool_config).run_optimization(measure_results_ids)


def run_full(vrtool_config: VrtoolConfig) -> None:
    """
    Full run of the model in the database that triggers all the available workflows
    (safety - measures - optimization). The results are consequently exported.

    Args:
        vrtool_config (VrtoolConfig): Configuration to use during run.
    """
    ApiRunWorkflows(vrtool_config).run_all()


class ApiRunWorkflows:
    def __init__(self, vrtool_config: VrtoolConfig) -> None:
        self.vrtool_config = vrtool_config

    @staticmethod
    def get_dike_traject(vrtool_config: VrtoolConfig) -> DikeTraject:
        """
        Imports the `DikeTraject` with all its `DikeSection`, `Mechanism`,
        and so on. Setting as well all available assessments.

        Args:
            vrtool_config (VrtoolConfig): Configuration determening how and from where to import the `DikeTraject`.

        Returns:
            DikeTraject: Imported instance with all related entities also imported.
        """
        # Import the DikeTraject and all present reliability data.
        return get_dike_traject(vrtool_config)

    def run_assessment(self) -> ResultsSafetyAssessment:
        # Clear the results
        clear_assessment_results(self.vrtool_config)

        # Safety assessment.
        _safety_assessment = RunSafetyAssessment(
            self.vrtool_config, self.get_dike_traject(self.vrtool_config)
        )
        _result = _safety_assessment.run()

        # Export the results.
        export_results_safety_assessment(_result)
        return _result

    def run_measures(self) -> ResultsMeasures:
        # Run assessment to achieve a stable step (for now)
        _results_assessment = self.run_assessment()
        # Clear the results
        clear_measure_results(_results_assessment.vr_config)

        # Run Measures.
        _measures = RunMeasures(
            _results_assessment.vr_config, _results_assessment.selected_traject
        )
        _measures_result = _measures.run()

        # Export solutions to database
        export_results_measures(_measures_result)
        return _measures_result

    def run_optimization(self, selected_measures_id: dict[int,int]) -> ResultsOptimization:
        """
        Runs an optimization for the given measure results ID's.

        Args:
            selected_measures (Measureresult): Selected set of measures' results to optimize.

        Returns:
            ResultsOptimization: Optimization results.
        """
        # Create optimization run
        _date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        _results_measures = create_optimization_run_for_selected_measures(
            self.vrtool_config,
            selected_measures_id,
            "Single opt. at: {}".format(_date),
        )

        # Run Optimization.
        _optimization = RunOptimization(_results_measures)
        _optimization_result = _optimization.run()

        # Export results
        export_results_optimization(_optimization_result)
        return _optimization_result

    def run_all(self) -> ResultsOptimization:
        """
        Runs the combination of the three steps at once.
        This (at the moment) is different from running each of them after each other.

        Returns:
            ResultsOptimization: The final results contained in the `ResultsOptimization`.
        """
        # Run all steps with one command.
        if not self.vrtool_config.output_directory.is_dir():
            logging.info(
                "Creating output directories at {}".format(
                    self.vrtool_config.output_directory
                )
            )
            self.vrtool_config.output_directory.mkdir(parents=True, exist_ok=True)
            self.vrtool_config.output_directory.joinpath("figures").mkdir(
                parents=True, exist_ok=True
            )
            self.vrtool_config.output_directory.joinpath(
                "results", "investment_steps"
            ).mkdir(parents=True, exist_ok=True)

        logging.info("Start run full model.")

        # Step 1. Safety assessment.
        _safety_assessment = RunSafetyAssessment(
            self.vrtool_config, self.get_dike_traject(self.vrtool_config)
        )
        _assessment_result = _safety_assessment.run()

        # Step 2. Run measures.
        _measures = RunMeasures(
            _assessment_result.vr_config, _assessment_result.selected_traject
        )
        _measures_result = _measures.run()

        # Step 3. Optimization.
        clear_optimization_results(self.vrtool_config)

        # Create optimization run in the db
        create_basic_optimization_run(self.vrtool_config, "Run full optimization")

        # Run optimization
        _optimization = RunOptimization(_measures_result)
        _optimization_result = _optimization.run()

        logging.info("Finished run full model.")
        return _optimization_result
