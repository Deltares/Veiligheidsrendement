import logging
from pathlib import Path

import pandas as pd

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
    get_all_measure_results_with_supported_investment_years,
    get_dike_traject,
    get_optimization_selected_measure_ids,
    get_optimization_step_with_lowest_total_cost,
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
    vrtool_config: VrtoolConfig,
    optimization_name: str,
    measure_result_id_year: list[tuple[int, int]],
) -> None:
    """
    Runs an optimization by optimizing the available measures
    in the database. The results are then exported into the database.

    Args:
        vrtool_config (VrtoolConfig): Configuration to use during run.
        optimization_name (str): Name given to an optimization run entry.
        measure_results_ids (list[tuple[int, int]]): List of measure result's ids
            paired with an investement year.
    """
    ApiRunWorkflows(vrtool_config).run_optimization(
        optimization_name, measure_result_id_year
    )


def run_full(vrtool_config: VrtoolConfig) -> None:
    """
    Full run of the model in the database that triggers all the available workflows
    (safety - measures - optimization). The results are consequently exported.

    Args:
        vrtool_config (VrtoolConfig): Configuration to use during run.
    """
    ApiRunWorkflows(vrtool_config).run_all()


def get_optimization_step_with_lowest_total_cost_table(
    vrtool_config: VrtoolConfig, optimization_run_id: int
) -> tuple[int, pd.DataFrame, float]:
    """
    Gets the (id) optimization step, all its related betas and the
    total cost of said step.

    Args:
        vrtool_config (VrtoolConfig): Configuration containing connection details.
        optimization_run_id (int): Optimization whose steps need to be analyzed.

    Returns:
        tuple[int, pd.DataFrame, float]: `OptimizationStep.id`, reliability dataframe
        and total cost of said step.
    """
    (
        _optimization_step,
        dataframe_betas,
        total_cost,
    ) = get_optimization_step_with_lowest_total_cost(vrtool_config, optimization_run_id)
    return _optimization_step.get_id(), dataframe_betas, total_cost


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

    def run_optimization(
        self, optimization_name: str, selected_measures_id_year: list[tuple[int, int]]
    ) -> ResultsOptimization:
        """
        Runs an optimization for the given measure results ID's.

        Args:
            optimization_name (str): Name given to an optimization run entry.
            selected_measures_id_year (list[tuple[int, int]]):
                Selected set of measures' results ids with investment year to optimize.

        Returns:
            ResultsOptimization: Optimization results.
        """
        # Create optimization run
        (
            _results_measures,
            _optimization_selected_measure_ids,
        ) = create_optimization_run_for_selected_measures(
            self.vrtool_config,
            optimization_name,
            selected_measures_id_year,
        )

        # Run Optimization.
        _optimization = RunOptimization(
            _results_measures, _optimization_selected_measure_ids
        )
        _optimization_result = _optimization.run()

        # Export results
        export_results_optimization(_optimization_result, _optimization.run_ids)
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

        # Export the results.
        export_results_safety_assessment(_assessment_result)

        # Step 2. Run measures.
        _measures = RunMeasures(
            _assessment_result.vr_config, _assessment_result.selected_traject
        )
        _results_measures = _measures.run()
        
        # Export solutions to database
        export_results_measures(_results_measures)

        _results_measures.ids_to_import = (
            get_all_measure_results_with_supported_investment_years(self.vrtool_config)
        )

        _optimization_selected_measure_ids = get_optimization_selected_measure_ids(
            self.vrtool_config, _results_measures
        )

        # Step 3. Optimization.
        clear_optimization_results(self.vrtool_config)

        # Create optimization run in the db
        create_basic_optimization_run(self.vrtool_config, "Run full optimization")

        # Run optimization
        _optimization = RunOptimization(
            _results_measures, _optimization_selected_measure_ids
        )
        _optimization_result = _optimization.run()

        logging.info("Finished run full model.")
        return _optimization_result
