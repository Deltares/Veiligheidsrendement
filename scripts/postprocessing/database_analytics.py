import copy

import numpy as np

from vrtool.common.enums import MechanismEnum
from vrtool.orm.models import *
from vrtool.orm.orm_controllers import open_database
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf


def get_minimal_tc_step(steps):
    """Get the step number with the minimal total cost.

    Args:
    steps: list of dicts, each dict contains the optimization step number, data on total_lcc and total_risk

    Returns:
    int, the step number with the minimal total cost
    """
    return min(steps, key=lambda x: x["total_cost"])["step_number"]


def get_measures_per_step_number(list_of_measures):
    """Groups the different measures by step number.

    Args:
    list_of_measures: list of dicts with input fields optimization_step_id, optimization_selected_measure, measure_result, investment_year, measure_per_section, section_id and step number.

    Returns:
    dict, with step number as key and a list of optimization_step_id, optimization_selected_measure, measure_result, investment_year, measure_per_section, section_id as values
    """

    result = {}
    for step_entry in list_of_measures:
        if step_entry["step_number"] not in result:
            result[step_entry["step_number"]] = {
                "optimization_step_id": [],
                "optimization_selected_measure": [],
                "measure_result": [],
                "investment_year": [],
                "measure_per_section": [],
                "section_id": [],
            }
        result[step_entry["step_number"]]["optimization_step_id"].append(
            step_entry["id"]
        )
        result[step_entry["step_number"]]["optimization_selected_measure"].append(
            step_entry["optimization_selected_measure"]
        )
        result[step_entry["step_number"]]["measure_result"].append(
            step_entry["measure_result"]
        )
        result[step_entry["step_number"]]["measure_per_section"].append(
            step_entry["measure_per_section"]
        )
        result[step_entry["step_number"]]["investment_year"].append(
            step_entry["investment_year"]
        )
        result[step_entry["step_number"]]["section_id"].append(step_entry["section"])
    return result


def get_reliability_for_each_step(database_path, measures_per_step):
    # read the OptimizationStepResultMechanism for those optimization_step_ids in measures_per_step
    # find min and max 'optimization_step_id' in measures_per_step dicts. where these values originates from lists in the dictionary
    def get_step_range(input):
        """Gets range of steps for measures that were given as input.

        Args:
        input: dict with keys as stepnumbers and values as lists of measures

        Returns:
        min_step: int, the lowest optimization_step_id
        max_step: int, the highest optimization_step_id"""
        number, _steps = zip(*list(input.items()))

        min_step = _steps[0]["optimization_step_id"][0]
        max_step = _steps[-1]["optimization_step_id"][0]
        return min_step, max_step

    def restructure_reliability_per_step(reliability_per_step):
        """Restructures the reliability_per_step to a dictionary with stepnumber as key, and reliability and section_id as values.
        Reliability itself is a dict with mechanism as key and lists of beta, time as values.

        Args:
        reliability_per_step: list of dicts with optimization_step_id, section_id, beta, time and mechanism_name

        Returns:
        reliability_steps: dict with stepnumber as key, and reliability and section_id as values.

        """
        reliability_steps = {}
        for step in reliability_per_step:
            if step["step_number"] not in reliability_steps:
                reliability_steps[step["step_number"]] = {}
                reliability_steps[step["step_number"]]["section_id"] = step["section"]
                reliability_steps[step["step_number"]]["reliability"] = {}
            if (
                MechanismEnum.get_enum(step["mechanism_name"])
                not in reliability_steps[step["step_number"]]["reliability"]
            ):
                reliability_steps[step["step_number"]]["reliability"][
                    MechanismEnum.get_enum(step["mechanism_name"])
                ] = {"beta": [], "time": []}
            if (
                step["time"]
                not in reliability_steps[step["step_number"]]["reliability"][
                    MechanismEnum.get_enum(step["mechanism_name"])
                ]["time"]
            ):
                reliability_steps[step["step_number"]]["reliability"][
                    MechanismEnum.get_enum(step["mechanism_name"])
                ]["beta"].append(step["beta"])
                reliability_steps[step["step_number"]]["reliability"][
                    MechanismEnum.get_enum(step["mechanism_name"])
                ]["time"].append(step["time"])
        return reliability_steps

    # first get min and max step to get from DB.
    min_step, max_step = get_step_range(measures_per_step)

    with open_database(database_path) as db:
        reliability_values = (
            OptimizationStepResultMechanism.select(
                OptimizationStepResultMechanism.optimization_step_id,
                OptimizationStepResultMechanism.mechanism_per_section_id,
                OptimizationStepResultMechanism.beta,
                OptimizationStepResultMechanism.time,
                MechanismPerSection.section_id,
                MechanismPerSection.mechanism_id,
                Mechanism.name.alias("mechanism_name"),
                OptimizationStep.step_number,
            )
            .join(
                MechanismPerSection,
                on=(
                    OptimizationStepResultMechanism.mechanism_per_section_id
                    == MechanismPerSection.id
                ),
            )
            .join(Mechanism)
            .join(
                OptimizationStep,
                on=(
                    OptimizationStepResultMechanism.optimization_step_id
                    == OptimizationStep.id
                ),
            )
            .where(
                OptimizationStepResultMechanism.optimization_step_id >= min_step,
                OptimizationStepResultMechanism.optimization_step_id <= max_step,
            )
            .dicts()
        )
    # restructure to a dictionary with stepnumber as key, mechanism as subkey and beta, time and section_id as values

    return restructure_reliability_per_step(list(reliability_values))


def assessment_for_each_step(assessment_input, reliability_per_step):
    """Combines the assessment input with the reliability per step to get the reliability assessment for each step.

    Args:
    assessment_input (dict): dictionary containing the reliability of each section and mechanism for the initial step.
    reliability_per_step (dict): dictionary containing the reliability of the section with an investment for each mechanism for each step.

    Returns:
    list: list of dictionaries containing the reliability of each section and mechanism for each step.
    """
    assessment_per_step = [copy.deepcopy(assessment_input)]
    for step, data in reliability_per_step.items():
        traject_reliability = copy.deepcopy(assessment_per_step[-1])
        for mechanism, reliability in data["reliability"].items():
            traject_reliability[mechanism][data["section_id"]]["beta"] = copy.deepcopy(
                reliability["beta"]
            )
            traject_reliability[mechanism][data["section_id"]]["time"] = reliability[
                "time"
            ]
        assessment_per_step.append(traject_reliability)
    assessment_per_step.pop(0)
    return assessment_per_step


def calculate_traject_probability_for_steps(stepwise_assessment):
    """Computes the system failure probability based on the reliability of sections and mechanisms for each step. Does so for each mechanism separately, and then combines.

    Args:
        stepwise_assessment (list): list of dictionaries containing the reliability of each section and mechanism for each step.

    Returns:
        dict: dictionary containing the system failure probability for each mechanism for each step.
    """

    def convert_beta_to_pf_per_section(traject_reliability):
        time = [t for section in traject_reliability.values() for t in section["time"]]
        beta = [b for section in traject_reliability.values() for b in section["beta"]]
        beta_per_time = {
            t: [b for b, t_ in zip(beta, time) if t_ == t] for t in set(time)
        }
        pf_per_time = {
            t: list(beta_to_pf(np.array(beta))) for t, beta in beta_per_time.items()
        }
        return pf_per_time

    def compute_overflow(traject_reliability):
        pf_per_time = convert_beta_to_pf_per_section(traject_reliability)
        traject_pf_per_time = {t: max(pf) for t, pf in pf_per_time.items()}
        return traject_pf_per_time

    def compute_piping_stability(traject_reliability):
        pf_per_time = convert_beta_to_pf_per_section(traject_reliability)
        traject_pf_per_time = {
            t: 1 - np.prod(np.subtract(1, pf)) for t, pf in pf_per_time.items()
        }
        return traject_pf_per_time

    def compute_revetment(traject_reliability):
        pf_per_time = convert_beta_to_pf_per_section(traject_reliability)
        traject_pf_per_time = {t: max(pf) for t, pf in pf_per_time.items()}
        return traject_pf_per_time

    def compute_system_failure_probability(traject_reliability):
        result = {}
        for mechanism, data in traject_reliability.items():
            if mechanism is MechanismEnum.OVERFLOW:
                result[mechanism] = compute_overflow(data)
            elif (
                mechanism is MechanismEnum.PIPING
                or mechanism is MechanismEnum.STABILITY_INNER
            ):
                result[mechanism] = compute_piping_stability(data)
            elif mechanism is MechanismEnum.REVETMENT:
                result[mechanism] = compute_revetment(data)
            else:
                raise ValueError(f"Mechanism {mechanism} not recognized.")
        return result

    traject_probability = []
    for step in stepwise_assessment:
        traject_probability.append(compute_system_failure_probability(step))
    return traject_probability


def get_measures_per_section_for_step(measures_per_step, final_step_no):
    """Get the measures per section for a given step number.

    Args:
    measures_per_step: dict with step number as key and a list of optimization_step_id, optimization_selected_measure, measure_result, investment_year, measure_per_section, section_id as values
    final_step_no: int, the final step number to consider

    Returns:
    dict: dictionary with section_id as key and a tuple of measure_result and investment_year as values
    """
    measures_per_section = {}
    for step_no, step in measures_per_step.items():
        if step_no <= final_step_no:
            if step["section_id"][0] not in measures_per_section:
                measures_per_section[step["section_id"][0]] = []
            measures_per_section[step["section_id"][0]] = (
                step["measure_result"],
                step["investment_year"],
            )
    return dict(sorted(measures_per_section.items()))


def get_beta_for_each_section_and_mech_at_t(assessment_of_step, t):
    """Get the beta for each section and mechanism at a specific time t.

    Args:
    assessment_of_step: dict with step number as key and a dict with mechanism as key and a dict with section as key and a list of time and beta as values
    t: int, the time at which the beta is requested

    Returns:
    dict: dictionary with section_id as key and a dict with mechanism as key and beta as value
    """
    beta_per_section = {}
    for section_id, section_betas in assessment_of_step.items():
        beta_per_section[section_id] = {}
        for mechanism, mechanism_assessment in section_betas.items():
            # get index of value in list where element is t
            t_idx = np.where(np.array(mechanism_assessment["time"]) == t)[0].item()
            beta_per_section[section_id][mechanism] = mechanism_assessment["beta"][
                t_idx
            ]
    return beta_per_section


def get_unique_years(measures_per_section):
    """Get the unique years from the measures per section.

    Args:
    measures_per_section: dict with section_id as key and a tuple of measure_result and investment_year as values

    Returns:
    set: set of unique years
    """

    years = []
    for measure in measures_per_section.values():
        years.append(measure[1])
    return set([item for sublist in years for item in sublist])
