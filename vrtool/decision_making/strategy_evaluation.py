import copy
import itertools
import logging
import math

import numpy as np
import pandas as pd

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.solutions import Solutions
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject, calc_traject_prob
from vrtool.optimization.measures.aggregated_measures_combination import (
    AggregatedMeasureCombination,
)
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf, pf_to_beta


def calc_life_cycle_risks(
    base0,
    discount_rate: float,
    horizon,
    damage,
    change=None,
    section=None,
    datatype="DataFrame",
    ts=None,
    mechs=False,
    option=None,
    dumpPt=False,
):
    base = copy.deepcopy(base0)
    if datatype == "DataFrame":
        mechs = np.unique(base.index.get_level_values("mechanism").values)
        if isinstance(change, pd.Series):
            for i in mechs:
                # This is not very efficient. Could be improved.
                base.loc[(section, i)] = change.loc[i]
        else:
            pass

        beta_t, p_t = calc_traject_prob(base, horizon=horizon)
    elif datatype == "Array":
        if isinstance(change, dict):
            for i in mechs:
                base[i][section] = change[i][option]
        else:
            pass
        if not (isinstance(ts, np.ndarray) or isinstance(ts, list)):
            ts = np.array(range(0, horizon))
        if not isinstance(mechs, np.ndarray):
            mechs = np.array(list(base.keys()))
        beta_t, p_t = calc_traject_prob(
            base, horizon=horizon, datatype="Arrays", ts=ts, mechs=mechs
        )

    # trange = np.arange(0, horizon + 1, 1)
    trange = np.arange(0, horizon, 1)
    _d_t = damage / (1 + discount_rate) ** trange
    risk_t = p_t * _d_t
    if dumpPt:
        np.savetxt(dumpPt, p_t, delimiter=",")
    TR = np.sum(risk_t)
    return TR


# this function changes the traject probability of a measure is implemented:
def implement_option(
    traject_probability: dict[str, np.ndarray],
    measure_idx: tuple[int, int, int],
    measure: AggregatedMeasureCombination,
) -> dict[str, np.ndarray]:
    """Implements a measure in the traject probability dictionary.

    Args:
        traject_probability (dict[np.ndarray]): The probabilities for each mechanism. The arrays have dimensions N x T with N the number of sections and T the number of years
        measure_idx (tuple): The index of the measure to implement (section_index, sh_index, sg_index).
        measure (AggregatedMeasureCombination): The measure to implement.


    Returns:
        dict[np.ndarray]: The updated traject probability dictionary. where the measure is implemented.
    """

    t_range = list(traject_probability.values())[0].shape[
        1
    ]  # TODO: this should be made more robust
    for mechanism_name in traject_probability.keys():
        if MechanismEnum.get_enum(mechanism_name) in [
            MechanismEnum.STABILITY_INNER,
            MechanismEnum.PIPING,
        ]:
            traject_probability[mechanism_name][
                measure_idx[0], :
            ] = measure.sg_combination.mechanism_year_collection.get_probabilities(
                MechanismEnum.get_enum(mechanism_name), np.arange(0, t_range, 1)
            )
    return traject_probability


def compute_annual_failure_probability(traject_probability: dict[np.ndarray]):
    """Computes the annual failure probability for each mechanism.

    Args:
        traject_probability (dict[np.ndarray]): The collection of the section probabilities for each mechanism. The array has dimensions N x T with N the number of sections and T the number of years

    Returns:
        np.ndarray: The annual failure probability of the traject.
    """
    annual_failure_probability = []
    for mechanism_name in traject_probability.keys():
        if MechanismEnum.get_enum(mechanism_name) in [
            MechanismEnum.STABILITY_INNER,
            MechanismEnum.PIPING,
        ]:
            # 1-prod
            annual_failure_probability.append(
                1 - (1 - traject_probability[mechanism_name]).prod(axis=0)
            )
        elif MechanismEnum.get_enum(mechanism_name) == MechanismEnum.REVETMENT:
            annual_failure_probability.append(
                4 * np.max(traject_probability[mechanism_name], axis=0)
            )
            # 4 * maximum. TODO This should be made consistent throughout the code. Issue for next sprint/
        elif MechanismEnum.get_enum(mechanism_name) == MechanismEnum.OVERFLOW:
            annual_failure_probability.append(
                np.max(traject_probability[mechanism_name], axis=0)
            )

    return np.sum(annual_failure_probability, axis=0)


def compute_total_risk(
    traject_probability: dict[np.ndarray], annual_discounted_damage: np.ndarray[float]
) -> float:
    """Computes the total risk of the traject.

    Args:
        traject_probability (dict[np.ndarray]): The collection of the section probabilities for each mechanism. The array has dimensions N x T with N the number of sections and T the number of years
        annual_discounted_damage (np.ndarray[float]): The annual discounted damage of the traject. The array has dimension T with T the number of years.

    Returns:
        float: The total risk of the traject.
    """
    annual_failure_probability = compute_annual_failure_probability(traject_probability)
    return np.sum(annual_failure_probability * annual_discounted_damage)


def evaluate_risk(
    init_overflow_risk,
    init_revetment_risk,
    init_geo_risk,
    strategy,
    n,
    sh,
    sg,
    config: VrtoolConfig,
):
    for mechanism in config.mechanisms:
        if mechanism == MechanismEnum.OVERFLOW:
            init_overflow_risk[n, :] = strategy.RiskOverflow[n, sh, :]
        elif mechanism == MechanismEnum.REVETMENT:
            init_revetment_risk[n, :] = strategy.RiskRevetment[n, sh, :]
        else:
            init_geo_risk[n, :] = strategy.RiskGeotechnical[n, sg, :]
    return init_overflow_risk, init_revetment_risk, init_geo_risk


def update_probability(init_probability, strategy, index):
    """index = [n,sh,sg]"""
    for i in init_probability:
        from scipy.stats import norm

        if i in [MechanismEnum.OVERFLOW.name, MechanismEnum.REVETMENT.name]:
            init_probability[i][index[0], :] = strategy.Pf[i][index[0], index[1], :]
        else:
            init_probability[i][index[0], :] = strategy.Pf[i][index[0], index[2], :]
    return init_probability
