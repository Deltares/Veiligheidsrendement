import numpy as np

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.aggregated_measures_combination import (
    AggregatedMeasureCombination,
)


def implement_option(
    traject_probability: dict[MechanismEnum, np.ndarray],
    measure_idx: tuple[int, int, int],
    measure: AggregatedMeasureCombination,
) -> dict[MechanismEnum, np.ndarray]:
    """Implements a measure in the traject probability dictionary.

    Args:
        traject_probability (dict[MechanismEnum, np.ndarray]): The probabilities for each mechanism. The arrays have dimensions N x T with N the number of sections and T the number of years
        measure_idx (tuple): The index of the measure to implement (section_index, sh_index, sg_index).
        measure (AggregatedMeasureCombination): The measure to implement.


    Returns:
        dict[MechanismEnum, np.ndarray]: The updated traject probability dictionary. where the measure is implemented.
    """

    t_range = list(traject_probability.values())[0].shape[
        1
    ]  # TODO: this should be made more robust
    for mechanism in traject_probability.keys():
        if mechanism in [
            MechanismEnum.STABILITY_INNER,
            MechanismEnum.PIPING,
        ]:
            traject_probability[mechanism][
                measure_idx[0], :
            ] = measure.sg_combination.mechanism_year_collection.get_probabilities(
                mechanism, np.arange(0, t_range, 1)
            )
    return traject_probability


def compute_annual_failure_probability(
    traject_probability: dict[MechanismEnum, np.ndarray]
):
    """Computes the annual failure probability for each mechanism.

    Args:
        traject_probability (dict[MechanismEnum, np.ndarray]): The collection of the traject probabilities for each mechanism.
            The array has dimensions N x T with N the number of sections and T the number of years

    Returns:
        np.ndarray: The annual failure probability of the traject.
    """
    annual_failure_probability = []
    for mechanism in traject_probability.keys():
        if mechanism in [
            MechanismEnum.STABILITY_INNER,
            MechanismEnum.PIPING,
        ]:
            # 1-prod
            annual_failure_probability.append(
                1 - (1 - traject_probability[mechanism]).prod(axis=0)
            )
        elif mechanism == MechanismEnum.REVETMENT:
            annual_failure_probability.append(
                4 * np.max(traject_probability[mechanism], axis=0)
            )
            # 4 * maximum. TODO This should be made consistent throughout the code. Issue for next sprint/
        elif mechanism == MechanismEnum.OVERFLOW:
            annual_failure_probability.append(
                np.max(traject_probability[mechanism], axis=0)
            )

    return np.sum(annual_failure_probability, axis=0)


def compute_total_risk(
    traject_probability: dict[MechanismEnum, np.ndarray],
    annual_discounted_damage: np.ndarray[float],
) -> float:
    """Computes the total risk of the traject.

    Args:
        traject_probability (dict[MechanismEnum, np.ndarray]): The collection of the traject probabilities for each mechanism.
            The array has dimensions N x T with N the number of sections and T the number of years.
        annual_discounted_damage (np.ndarray[float]): The annual discounted damage of the traject.
            The array has dimension T with T the number of years.

    Returns:
        float: The total risk of the traject.
    """
    annual_failure_probability = compute_annual_failure_probability(traject_probability)
    return np.sum(annual_failure_probability * annual_discounted_damage)
