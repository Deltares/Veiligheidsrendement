from pathlib import Path

from vrtool.common.enums import MechanismEnum
from vrtool.orm.models import *
from vrtool.orm.orm_controllers import open_database


def get_overview_of_runs(db_path):
    """Get an overview of the optimization runs in the database.

    Args:
    db_path: str, path to the database

    Returns:
    list of dicts, each dict contains the run id, run name, optimization type, and the discount rate
    """

    with open_database(db_path).connection_context():
        optimization_types = OptimizationRun.select(
            OptimizationRun, OptimizationType.name.alias("optimization_type_name")
        ).join(
            OptimizationType,
            on=(OptimizationRun.optimization_type_id == OptimizationType.id),
        )

        return list(optimization_types.dicts())  # desired output like this? TODO


def get_optimization_steps_for_run_id(db_path, run_id):
    """Get the optimization steps for a specific run id.

    Args:
    db_path: str, path to the database
    run_id: int, the run id for which the optimization steps are requested

    Returns:
    list of dicts, each dict contains the optimization step number, data on total_lcc and total_risk
    """

    def add_total_cost_to_steps(steps):
        for step in steps:
            step["total_cost"] = step["total_lcc"] + step["total_risk"]
        return steps

    with open_database(db_path).connection_context():
        optimization_steps = (
            OptimizationStep.select(
                OptimizationStep, OptimizationSelectedMeasure.optimization_run_id
            )
            .join(OptimizationSelectedMeasure)
            .where(OptimizationSelectedMeasure.optimization_run_id == run_id)
            .group_by(OptimizationStep.step_number)
            .select(
                OptimizationStep.step_number,
                OptimizationStep.total_lcc,
                OptimizationStep.total_risk,
            )
        )
    return add_total_cost_to_steps(
        list(optimization_steps.dicts())
    )  # desired output like this? TODO


# import AssessmentMechanismResult for a given mechanism and order this by section
def import_original_assessment(database_path, mechanism: MechanismEnum):
    """Imports original assessment for given mechanism.

    Args:
    database_path (Path): path to the database
    mechanism (MechanismEnum): mechanism to import


    Returns:
    dict: dictionary with section_ids as key and a list of time and beta as values"""

    with open_database(database_path):
        assessment = (
            AssessmentMechanismResult.select(
                AssessmentMechanismResult, MechanismPerSection, Mechanism.name
            )
            .join(
                MechanismPerSection,
                on=(
                    AssessmentMechanismResult.mechanism_per_section_id
                    == MechanismPerSection.id
                ),
            )
            .join(Mechanism, on=(MechanismPerSection.mechanism_id == Mechanism.id))
            .where(Mechanism.name == mechanism.legacy_name)
            .order_by(MechanismPerSection.id)
            .dicts()
        )
    # reorder such that each entry has 1 section, and a list of time and beta
    result = {}
    for entry in assessment:
        if entry["section"] not in result:
            result[entry["section"]] = {"time": [], "beta": []}
        result[entry["section"]]["time"].append(entry["time"])
        result[entry["section"]]["beta"].append(entry["beta"])
    return result


def get_measures_for_run_id(database_path, run_id):
    """Get the measures for a specific run id.

    Args:
    database_path: str, path to the database
    run_id: int, the run id for which the measures are requested

    Returns:
    list of dicts, each dict contains the optimization step number, optimization_selected_measure_id, measure_result_id, investment_year, measure_per_section_id, section_id
    """
    with open_database(database_path) as db:
        measures = (
            OptimizationStep.select(
                OptimizationStep.id,
                OptimizationStep.step_number,
                OptimizationStep.optimization_selected_measure_id,
                OptimizationSelectedMeasure.measure_result_id,
                OptimizationSelectedMeasure.investment_year,
                MeasureResult.measure_per_section_id,
                MeasurePerSection.section_id,
            )
            .join(
                OptimizationSelectedMeasure,
                on=(
                    OptimizationStep.optimization_selected_measure_id
                    == OptimizationSelectedMeasure.id
                ),
            )
            .join(
                MeasureResult,
                on=(OptimizationSelectedMeasure.measure_result_id == MeasureResult.id),
            )
            .join(
                MeasurePerSection,
                on=(MeasureResult.measure_per_section_id == MeasurePerSection.id),
            )
            .where(OptimizationSelectedMeasure.optimization_run_id == run_id)
            .order_by(OptimizationStep.id)
            .dicts()
        )
        return list(measures)


def get_measure_costs(measure_result_id, database_path):
    """Get the costs of a measure.

    Args:
    measure_result_id: int, the measure result id for which the costs are requested
    database_path: str, path to the database

    Returns:
    dict, containing the cost of the measure
    """
    with open_database(database_path) as db:
        measure = MeasureResult.get(MeasureResult.id == measure_result_id)
        measure_cost = MeasureResultSection.get(
            MeasureResultSection.measure_result == measure
        ).cost
    return {"cost": measure_cost}


def get_measure_parameters(measure_result_id, database_path):
    """Get the parameters of a measure.

    Args:
    measure_result_id: int, the measure result id for which the parameters are requested
    database_path: str, path to the database

    Returns:
    dict, containing the parameters of the measure
    """
    # get parameters from MeasureResultParameter where measure_result_id = measure_result_id
    with open_database(database_path) as db:
        measure = MeasureResult.get(MeasureResult.id == measure_result_id)
        # get parameters from MeasureResultParameter where measure_result_id = measure_result_id
        try:
            parameters = MeasureResultParameter.select().where(
                MeasureResultParameter.measure_result == measure
            )
            return {parameter.name.lower(): parameter.value for parameter in parameters}
        except:
            return {}


def get_measure_type(measure_result_id, database_path):
    """Get the type of a measure.

    Args:
    measure_result_id: int, the measure result id for which the type is requested
    database_path: str, path to the database

    Returns:
    dict, containing the type of the measure
    """
    with open_database(database_path) as db:
        measure = MeasureResult.get(MeasureResult.id == measure_result_id)
        measure_name = (
            MeasurePerSection.select(MeasurePerSection, Measure.name)
            .join(Measure)
            .where(MeasurePerSection.id == measure.measure_per_section_id)
            .get()
            .measure.name
        )
    return {"name": measure_name}


def get_measure_costs_from_measure_results(
    database_path: Path, measures_per_step: dict
):
    lcc_per_step = []
    with open_database(database_path) as db:
        for _, values in measures_per_step.items():
            lcc = 0
            for count, mr_id in enumerate(values["measure_result"]):
                # get the cost from the database by getting the cost from MeasureResultSection
                measure_cost = MeasureResultSection.get(
                    MeasureResultSection.measure_result_id == mr_id
                ).cost

                # discount if necessary
                if values["investment_year"][count] != 0:
                    measure_cost = measure_cost / (1.03) ** (
                        values["investment_year"][count]
                    )

                lcc += measure_cost
            lcc_per_step.append(lcc)
    return lcc_per_step
