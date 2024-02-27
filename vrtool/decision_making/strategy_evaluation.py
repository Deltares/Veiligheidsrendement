import copy

import math

import numpy as np
import pandas as pd

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.optimization.measures.aggregated_measures_combination import AggregatedMeasureCombination


# this function changes the traject probability of a measure is implemented:
def implement_option(traject_probability: dict[np.ndarray], 
                     measure_idx: tuple, 
                     measure: AggregatedMeasureCombination):
    t_range = list(traject_probability.values())[0].shape[1] #TODO: this should be made more robust
    for mechanism_name in traject_probability.keys():
        if MechanismEnum.get_enum(mechanism_name) in [MechanismEnum.STABILITY_INNER, MechanismEnum.PIPING]:
            traject_probability[mechanism_name][measure_idx[0],:] = measure.sg_combination.mechanism_year_collection.get_probabilities(MechanismEnum.get_enum(mechanism_name), np.arange(0,t_range,1))
    return traject_probability

def compute_annual_failure_probability(traject_probability: dict[np.ndarray]):
    annual_failure_probability = []
    for mechanism_name in traject_probability.keys():
        if MechanismEnum.get_enum(mechanism_name) in [MechanismEnum.STABILITY_INNER, MechanismEnum.PIPING]:
            #1-prod
            annual_failure_probability.append(1-(1-traject_probability[mechanism_name]).prod(axis=0))
        elif MechanismEnum.get_enum(mechanism_name) in [MechanismEnum.REVETMENT]:
            annual_failure_probability.append(4 * np.max(traject_probability[mechanism_name],axis=0))
            #4 * maximum
            pass
        elif MechanismEnum.get_enum(mechanism_name) in [MechanismEnum.OVERFLOW]:
            annual_failure_probability.append(np.max(traject_probability[mechanism_name],axis=0))         
    
    return np.sum(annual_failure_probability,axis=0)
    

def compute_total_risk(traject_probability: dict[np.ndarray],
                       annual_discounted_damage: np.ndarray[float]):
    annual_failure_probability = compute_annual_failure_probability(traject_probability)
    return np.sum(annual_failure_probability * annual_discounted_damage)
def split_options(
    options: dict[str, pd.DataFrame], available_mechanisms: list[MechanismEnum]
) -> tuple[list[dict[str, pd.DataFrame]], list[dict[str, pd.DataFrame]]]:
    """Splits the options for the measures.

    Args:
        options (_type_): The available options to split.
        available_mechanisms (list[MechanismEnum]): The collection of the names of the available mechanisms for the evaluation.

    Returns:
        list[dict[str, pd.DataFrame]]: The collection of splitted options_dependent
        list[dict[str, pd.DataFrame]]: The collection of splitted options_independent
    """

    def get_dropped_dependent_options(
        available_mechanisms: list[MechanismEnum],
    ) -> list[str]:
        options = []
        for available_mechanism in available_mechanisms:
            if available_mechanism in [
                MechanismEnum.STABILITY_INNER,
                MechanismEnum.PIPING,
            ]:
                options.append(available_mechanism.name)

        options.append("Section")
        return options

    def get_dropped_independent_options(
        available_mechanisms: list[MechanismEnum],
    ) -> list[str]:
        options = []
        for available_mechanism in available_mechanisms:
            if available_mechanism.name in [
                MechanismEnum.OVERFLOW,
                MechanismEnum.REVETMENT,
            ]:
                options.append(available_mechanism.name)

        options.append("Section")
        return options

    options_dependent = copy.deepcopy(options)
    options_independent = copy.deepcopy(options)
    for i in options.keys():

        min_transition_level = (
            options[i].transition_level[options[i].transition_level > 0].min()
        )
        min_beta_target = options[i].beta_target[options[i].beta_target > 0].min()

        # for dependent sections we have all measures where there is a transition_level larger than the minimum, or a beta_target larger than the minimum
        # and the berm should be either non-existent -999 or 0
        def is_dependent_measure_present(option):
            if math.isnan(min_transition_level) or math.isnan(min_beta_target):
                # no revetment measures; just check dberm:
                return option.dberm <= 0
            else:
                return (
                    (option.transition_level >= min_transition_level)
                    & (option.beta_target >= min_beta_target)
                    & (option.dberm <= 0)
                )

        options_dependent[i] = options_dependent[i].loc[
            is_dependent_measure_present(options_dependent[i])
        ]

        # for independent measures dcrest should be 0 or -999, and transition_level and beta_target should be -999
        def is_independent_measure_present(option):
            if math.isnan(min_transition_level) or math.isnan(min_beta_target):
                # no revetment measures; just check dcrest:
                return option.dcrest <= 0.0
            else:
                return (
                    (option.dcrest <= 0.0)
                    & (option.transition_level <= min_transition_level)
                    & (option.beta_target <= min_beta_target)
                )

        options_independent[i] = options_independent[i].loc[
            is_independent_measure_present(options_independent[i])
        ]

        # we only need the measures with ids that are also in options_dependent
        options_independent[i] = options_independent[i].loc[
            options_independent[i].ID.isin(options_dependent[i].ID.unique())
        ]

        # Now that we have split the measures we should avoid double counting of costs by correcting some of the cost values.
        # This concerns the startcosts for soil reinforcement

        # We get the min cost which is equal to the minimum costs for a soil reinforcement (which we assume has dimensions 0 m crest and 0 m berm)
        startcosts_soil = np.min(
            options[i][(options[i]["type"] == "Soil reinforcement")]["cost"]
        )

        # we subtract the startcosts for all soil reinforcements (including those with stability screens)
        # Note that this is not robust as it depends on the exact formulation of types in the options. We should ensure that these names do not change in the future
        # we need to distinguish between measures where cost is a float and where it is a list
        # TODO this can be a function
        float_costs = options_independent[i]["cost"].map(type) == float
        soil_reinforcements = options_independent[i]["type"].str.contains(
            "Soil reinforcement"
        )
        for idx, row in options_independent[i].iterrows():
            # subtract startcosts for all entries in options_independent where float_costs is true and the type contains soil reinforcement
            if float_costs[idx] & soil_reinforcements[idx]:
                options_independent[i].loc[idx, "cost"] = np.subtract(
                    options_independent[i].loc[idx, "cost"], startcosts_soil
                )[0]
            # if it is a soil reinforcement combined with others we need to modify the right value from the list of costs
            if (not float_costs[idx]) & soil_reinforcements[idx]:
                # break the type string at '+' and find the value that contains soil reinforcement
                for cost_index, measure_type in enumerate(
                    row["type"].item().split("+")
                ):
                    if "Soil reinforcement" in measure_type:
                        # get list of costs and subtract startcosts from the cost that contains soil reinforcement
                        cost_list = row["cost"].item().copy()
                        cost_list[cost_index] = np.subtract(
                            cost_list[cost_index], startcosts_soil
                        )
                        options_independent[i].loc[idx, "cost"] = [
                            [val] for val in cost_list
                        ]

        # Then we deal with the costs for a stability screen when combined with a berm, these are accounted for in the independent_measure costs so should be removed from the dependent measures
        cost_stability_screen = (
            np.min(
                options[i].loc[
                    options[i]["type"].str.fullmatch(
                        "Soil reinforcement with stability screen"
                    )
                ]["cost"]
            )
            - startcosts_soil
        )
        # Find all dependent measures that contain a stability screen
        stability_screens = options_dependent[i]["type"].str.contains(
            "Soil reinforcement with stability screen"
        )
        for idx, row in options_dependent[i].iterrows():
            # subtract cost_stability_screen from all entries in options_dependent where stability_screens is true
            if stability_screens[idx]:
                # break the type string at '+' and find the value that contains soil reinforcement
                for cost_index, measure_type in enumerate(
                    row["type"].item().split("+")
                ):
                    if "Soil reinforcement with stability screen" in measure_type:
                        if isinstance(row["cost"].item(), float):
                            options_dependent[i].loc[idx, "cost"] = np.subtract(
                                options_dependent[i].loc[idx, "cost"],
                                cost_stability_screen,
                            )[0]
                        else:
                            # get list of costs and subtract startcosts from the cost that contains soil reinforcement
                            cost_list = row["cost"].item()
                            cost_list[cost_index] = np.subtract(
                                cost_list[cost_index], cost_stability_screen
                            )
                            # pass cost_list back to the idx, "cost" column in options_dependent[i]
                            options_dependent[i].loc[idx, "cost"] = [
                                [val] for val in cost_list
                            ]

        # Find all dependent measures that contain a Vertical Geotextile or a Diaphragm wall
        vertical_geotextiles = options_dependent[i]["type"].str.contains(
            "Vertical Geotextile"
        )
        diaphragm_walls = options_dependent[i]["type"].str.contains("Diaphragm Wall")

        def set_cost_to_zero(options_set, bools, measure_string):
            for row_idx, option_row in options_set.iterrows():
                if bools[row_idx]:
                    # break the type string at '+' and find the value that is a vertical Geotextile
                    for cost_index, measure_type in enumerate(
                        option_row["type"].item().split("+")
                    ):
                        if measure_type == measure_string:
                            if isinstance(option_row["cost"].item(), float):
                                options_set.loc[row_idx, "cost"] = 0.0
                            else:
                                # get list of costs and set cost of geotextile to 0
                                cost_list = option_row["cost"].item().copy()
                                cost_list[cost_index] = 0.0
                                # pass cost_list back to the row_idx, "cost" column in options_dependent[i]
                                options_set.loc[row_idx, "cost"] = [
                                    [val] for val in cost_list
                                ]
            return options_set

        # set costs of vertical geotextiles & diaphragm walls to 0 in options_dependent
        if any(vertical_geotextiles):
            options_dependent[i] = set_cost_to_zero(
                options_dependent[i], vertical_geotextiles, "Vertical Geotextile"
            )

        if any(diaphragm_walls):
            options_dependent[i] = set_cost_to_zero(
                options_dependent[i], diaphragm_walls, "Diaphragm Wall"
            )

        revetments = options_independent[i]["type"].str.contains("Revetment")
        if any(revetments):
            options_independent[i] = set_cost_to_zero(
                options_independent[i], revetments, "Revetment"
            )

        options_independent[i] = options_independent[i].reset_index(drop=True)
        options_dependent[i] = options_dependent[i].reset_index(drop=True)

        # only keep reliability of relevant mechanisms in dictionary
        options_dependent[i].drop(
            get_dropped_dependent_options(available_mechanisms),
            axis=1,
            level=0,
            inplace=True,
        )
        options_independent[i].drop(
            get_dropped_independent_options(available_mechanisms),
            axis=1,
            level=0,
            inplace=True,
        )
    return options_dependent, options_independent

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
