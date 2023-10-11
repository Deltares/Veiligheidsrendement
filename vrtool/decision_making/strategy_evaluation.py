import copy
import itertools
import logging
import math

import numpy as np
import pandas as pd

from vrtool.common.enums import MechanismEnum
from vrtool.decision_making.solutions import Solutions
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject, calc_traject_prob
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf, pf_to_beta


# This script combines two sets of measures to a single option
def measure_combinations(
    combinables,
    partials,
    solutions: Solutions,
    indexCombined2single: list[int],
    splitparams=False,
):
    _combined_measures = pd.DataFrame(columns=combinables.columns)
    # all columns without a second index are attributes of the measure
    attribute_col_names = combinables.columns.get_level_values(0)[
        combinables.columns.get_level_values(1) == ""
    ].tolist()
    # years are those columns of level 2 with a second index that is not ''
    years = (
        combinables.columns.get_level_values(1)[
            combinables.columns.get_level_values(1) != ""
        ]
        .unique()
        .tolist()
    )
    # mechanisms are those columns of level 1 with a second index that is not ''
    mechanism_names = (
        combinables.columns.get_level_values(0)[
            combinables.columns.get_level_values(1) != ""
        ]
        .unique()
        .tolist()
    )

    # make dict with attribute_col_names as keys and empty lists as values
    attribute_col_dict = {col: [] for col in attribute_col_names}

    # make dict with mechanisms as keys, sub dicts of years and then empty lists as values
    mechanism_beta_dict = {
        mechanism_name: {year: [] for year in years}
        for mechanism_name in mechanism_names
    }
    count = 0
    # loop over partials
    for i, row1 in partials.iterrows():
        # combine with all combinables
        for j, row2 in combinables.iterrows():
            indexCombined2single.append([i, j])
            for col in attribute_col_names:
                if (
                    col == "ID"
                ):  # TODO maybe add type here as well and just concatenate the types as a string
                    attribute_value = f'{row1["ID"].values[0]}+{row2["ID"].values[0]}'
                elif col == "class":
                    attribute_value = "combined"
                else:
                    # for all other columns we combine the lists and make sure that it is not nested
                    combined_data = row1[col].tolist() + row2[col].tolist()
                    attribute_value = list(
                        itertools.chain.from_iterable(
                            itertools.repeat(x, 1)
                            if (isinstance(x, str))
                            or (isinstance(x, int))
                            or (isinstance(x, float))
                            else x
                            for x in combined_data
                        )
                    )
                    if (
                        col == "type"
                    ):  # if it is the type we make sure that it is a single string and store it as list of length 1
                        attribute_value = ["+".join(attribute_value)]
                    # drop all -999 values from attribute_value
                    attribute_value = [
                        x for x in attribute_value if x != -999 and x != -999.0
                    ]
                    if (
                        len(attribute_value) == 1
                    ):  # if there is only one value we take that value
                        attribute_value = attribute_value[0]
                    elif len(attribute_value) == 0:  # if there is no value we take -999
                        attribute_value = -999
                    else:
                        pass
                attribute_col_dict[col].append(attribute_value)

            # then we fill the mechanism_beta_dict we ignore Section as mechanism, we do that as a last step on the dataframe
            for mechanism_name in mechanism_beta_dict.keys():
                if mechanism_name == "Section":
                    continue
                else:
                    for year in mechanism_beta_dict[mechanism_name].keys():
                        if row1["type"].all() == "Vertical Geotextile":
                            vzg_idx = solutions.measure_table.loc[
                                solutions.measure_table["Name"]
                                == "Verticaal Zanddicht Geotextiel"
                            ].index.values[0]
                            Pf_VZG = solutions.measures[vzg_idx].parameters[
                                "Pf_solution"
                            ]
                            P_VZG = solutions.measures[vzg_idx].parameters["P_solution"]
                            pf_vzg = (1 - P_VZG) * Pf_VZG + P_VZG * beta_to_pf(
                                row2[(mechanism_name, year)]
                            )
                            mechanism_beta_dict[mechanism_name][year].append(
                                pf_to_beta(pf_vzg)
                            )
                        else:
                            mechanism_beta_dict[mechanism_name][year].append(
                                np.maximum(
                                    row1[mechanism_name, year],
                                    row2[mechanism_name, year],
                                )
                            )

            count += 1

    attribute_col_df = pd.DataFrame.from_dict(attribute_col_dict)
    attribute_col_df.columns = pd.MultiIndex.from_tuples(
        [(col, "") for col in attribute_col_df.columns]
    )
    mechanism_beta_df = (
        pd.DataFrame.from_dict(mechanism_beta_dict, orient="index").stack().to_frame()
    )
    mechanism_beta_df = pd.DataFrame(
        mechanism_beta_df[0].values.tolist(), index=mechanism_beta_df.index
    )
    mechanism_beta_df.index = pd.MultiIndex.from_tuples(mechanism_beta_df.index)
    _combined_measures = pd.concat(
        (attribute_col_df, mechanism_beta_df.transpose()), axis=1
    )
    for year in years:
        # compute the section beta
        betas_in_year = _combined_measures.loc[
            :,
            (
                _combined_measures.columns.get_level_values(0) != "Section",
                _combined_measures.columns.get_level_values(1) == 25,
            ),
        ]
        pf_in_year = beta_to_pf(betas_in_year)
        section_beta = pf_to_beta(1 - np.prod(1 - pf_in_year, axis=1))
        # add the section beta to the dataframe
        _combined_measures.loc[:, ("Section", year)] = section_beta
    return _combined_measures


def revetment_combinations(
    partials: pd.DataFrame,
    revetment_measures: pd.DataFrame,
    indexCombined2single: list[int],
) -> pd.DataFrame:
    """
    Combines the revetment measures based on the input arguments.

    Args:
        partials (pd.Dataframe): An object containing the measures to combine the revetment measures with.
        revetment_measures (pd.Dataframe): An object containing the revetment measures.

    Returns:
        pd.DataFrame: An object containing the combined revetment measures.
    """
    _combined_measures = pd.DataFrame(columns=revetment_measures.columns)
    # all columns without a second index are attributes of the measure
    attribute_col_names = revetment_measures.columns.get_level_values(0)[
        revetment_measures.columns.get_level_values(1) == ""
    ].tolist()
    # years are those columns of level 2 with a second index that is not ''
    years = (
        revetment_measures.columns.get_level_values(1)[
            revetment_measures.columns.get_level_values(1) != ""
        ]
        .unique()
        .tolist()
    )
    # mechanisms are those columns of level 1 with a second index that is not ''
    mechanism_names = (
        revetment_measures.columns.get_level_values(0)[
            revetment_measures.columns.get_level_values(1) != ""
        ]
        .unique()
        .tolist()
    )

    # make dict with attribute_col_names as keys and empty lists as values
    attribute_col_dict = {col: [] for col in attribute_col_names}

    # make dict with mechanisms as keys, sub dicts of years and then empty lists as values
    mechanism_beta_dict = {
        mechanism_name: {year: [] for year in years}
        for mechanism_name in mechanism_names
    }

    # loop over partials
    for i, row1 in partials.iterrows():
        # combine with all combinables (in this case revetment measures)
        for j, row2 in revetment_measures.iterrows():
            newIndex = [j]  # revetment is a single measure
            for k in indexCombined2single[i]:
                newIndex.append(
                    k
                )  # partial can be combined (TODO name partial is not correct)
            indexCombined2single.append(newIndex)
            for col in attribute_col_names:
                if (
                    col == "ID"
                ):  # TODO maybe add type here as well and just concatenate the types as a string
                    attribute_value = f'{row1["ID"].values[0]}+{row2["ID"].values[0]}'
                elif col == "class":
                    attribute_value = "combined"
                else:
                    # for all other columns we combine the lists and make sure that it is not nested
                    combined_data = row1[col].tolist() + row2[col].tolist()
                    attribute_value = list(
                        itertools.chain.from_iterable(
                            itertools.repeat(x, 1)
                            if (isinstance(x, str))
                            or (isinstance(x, int))
                            or (isinstance(x, float))
                            else x
                            for x in combined_data
                        )
                    )
                    if (
                        col == "type"
                    ):  # if it is the type we make sure that it is a single string and store it as list of length 1
                        attribute_value = ["+".join(attribute_value)]
                    # drop all -999 values from attribute_value
                    attribute_value = [
                        x for x in attribute_value if x != -999 and x != -999.0
                    ]
                    if (
                        len(attribute_value) == 1
                    ):  # if there is only one value we take that value
                        attribute_value = attribute_value[0]
                    elif len(attribute_value) == 0:  # if there is no value we take -999
                        attribute_value = -999
                    else:
                        pass
                attribute_col_dict[col].append(attribute_value)

            # then we fill the mechanism_beta_dict we ignore Section as mechanism, we do that as a last step on the dataframe
            for mechanism_name in mechanism_beta_dict.keys():
                if mechanism_name == "Section":
                    continue
                else:
                    for year in mechanism_beta_dict[mechanism_name].keys():
                        mechanism_beta_dict[mechanism_name][year].append(
                            np.maximum(
                                row1[mechanism_name, year], row2[mechanism_name, year]
                            )
                        )

    attribute_col_df = pd.DataFrame.from_dict(attribute_col_dict)
    attribute_col_df.columns = pd.MultiIndex.from_tuples(
        [(col, "") for col in attribute_col_df.columns]
    )
    mechanism_beta_df = (
        pd.DataFrame.from_dict(mechanism_beta_dict, orient="index").stack().to_frame()
    )
    mechanism_beta_df = pd.DataFrame(
        mechanism_beta_df[0].values.tolist(), index=mechanism_beta_df.index
    )
    mechanism_beta_df.index = pd.MultiIndex.from_tuples(mechanism_beta_df.index)
    _combined_measures = pd.concat(
        (attribute_col_df, mechanism_beta_df.transpose()), axis=1
    )
    for year in years:
        # compute the section beta
        betas_in_year = _combined_measures.loc[
            :,
            (
                _combined_measures.columns.get_level_values(0) != "Section",
                _combined_measures.columns.get_level_values(1) == year,
            ),
        ]
        pf_in_year = beta_to_pf(betas_in_year)
        section_beta = pf_to_beta(1 - np.prod(1 - pf_in_year, axis=1))
        # add the section beta to the dataframe
        _combined_measures.loc[:, ("Section", year)] = section_beta

    return _combined_measures


def make_traject_df(traject: DikeTraject, cols):
    # cols = cols[1:]
    sections = []

    for i in traject.sections:
        sections.append(i.name)

    mechanism_names = list(_mechanism.name for _mechanism in traject.mechanisms) + [
        "Section"
    ]
    df_index = pd.MultiIndex.from_product(
        [sections, mechanism_names], names=["name", "mechanism"]
    )
    _traject_probability = pd.DataFrame(columns=cols, index=df_index)

    for _section in traject.sections:
        for _mechanism_name in mechanism_names:
            if (
                _mechanism_name
                not in _section.section_reliability.SectionReliability.index
            ):
                # TODO (VRTOOL-187).
                # Should we inject nans?
                # Not all sections include revetment(s), therefore it's skipped.
                logging.warning(
                    "Section '{}' does not include data for mechanism '{}'.".format(
                        _section.name, _mechanism_name
                    )
                )
                continue
            _traject_probability.loc[(_section.name, _mechanism_name)] = list(
                _section.section_reliability.SectionReliability.loc[_mechanism_name]
            )

    return _traject_probability


# hereafter a bunch of functions to compute costs, risks and probabilities over time are defined:
def calc_tc(section_options, discount_rate: float, horizon=100):
    costs = section_options["cost"].values
    years = section_options["year"].values
    discountfactors = list(map(lambda x: 1 / (1 + discount_rate) ** np.array(x), years))
    TC = list(map(lambda c, r: c * r, costs, discountfactors))
    return np.array(list(map(lambda c: np.sum(c), TC)))


def calc_tr(
    section,
    section_options,
    base_traject,
    original_section,
    discount_rate: float,
    horizon=100,
    damage=1e9,
):
    # section: the section name
    # section_options: all options for the section
    # base_traject: traject probability with all implemented measures
    # takenmeasures: object with all measures taken
    # original section: series of probabilities of section, before taking a measure.
    if damage == 1e9:
        logging.warning("No damage defined.")

    TotalRisk = []
    dR = []
    mechs = np.unique(base_traject.index.get_level_values("mechanism").values)
    sections = np.unique(base_traject.index.get_level_values("name").values)
    section_idx = np.where(sections == section)[0]
    section_options_array = {}
    base_array = {}
    TotalRisk = []
    dR = []

    for i in mechs:
        base_array[i] = base_traject.xs(i, level=1).values.astype("float")
        if isinstance(section_options, pd.DataFrame):
            section_options_array[i] = section_options.xs(
                i, level=0, axis=1
            ).values.astype("float")
            range_idx = len(section_options_array[mechs[0]])

        if isinstance(section_options, pd.Series):
            section_options_array[i] = section_options.xs(i, level=0).values.astype(
                "float"
            )
            range_idx = 0

    if "section_options_array" in locals():
        base_risk = calc_life_cycle_risks(
            base_array,
            discount_rate,
            horizon,
            damage,
            datatype="Array",
            ts=base_traject.columns.values,
            mechs=mechs,
        )

        for i in range(range_idx):
            TR = calc_life_cycle_risks(
                base_array,
                discount_rate,
                horizon,
                damage,
                change=section_options_array,
                section=section_idx,
                datatype="Array",
                ts=base_traject.columns.values,
                mechs=mechs,
                option=i,
            )
            TotalRisk.append(TR)
            dR.append(base_risk - TR)
    else:
        base_risk = calc_life_cycle_risks(base_traject, discount_rate, horizon, damage)
        if isinstance(section_options, pd.DataFrame):
            for i, row in section_options.iterrows():
                TR = calc_life_cycle_risks(
                    base_traject,
                    discount_rate,
                    horizon,
                    damage,
                    change=row,
                    section=section,
                )
                TotalRisk.append(TR)
                dR.append(base_risk - TR)

        elif isinstance(section_options, pd.Series):
            TR = calc_life_cycle_risks(
                base_traject,
                discount_rate,
                horizon,
                damage,
                change=section_options,
                section=section,
            )
            TotalRisk.append(TR)
            dR.append(base_risk - TR)

    return base_risk, dR, TotalRisk


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


# this function changes the trajectprobability of a measure is implemented:
def implement_option(section, traject_probability, new_probability):
    mechs = np.unique(traject_probability.index.get_level_values("mechanism").values)
    # change trajectprobability by changing probability for each mechanism
    for i in mechs:
        traject_probability.loc[(section, i)] = new_probability[i]
    return traject_probability


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
                        cost_list = row["cost"].item()
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


def solve_mip(mip_model):

    MixedIntegerSolution = mip_model.solve()
    return MixedIntegerSolution


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
