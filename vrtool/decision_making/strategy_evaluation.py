import copy
import logging

import numpy as np
import pandas as pd

from vrtool.decision_making.solutions import Solutions
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject, calc_traject_prob
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf, pf_to_beta


# This script combines two sets of measures to a single option
def measure_combinations(
    combinables, partials, solutions: Solutions, splitparams=False
):
    _combined_measures = pd.DataFrame(columns=combinables.columns)

    # loop over partials
    for i, row1 in partials.iterrows():
        # combine with all combinables
        for j, row2 in combinables.iterrows():

            ID = "+".join((row1["ID"].values[0], row2["ID"].values[0]))
            types = [row1["type"].values[0], row2["type"].values[0]]
            year = [row1["year"].values[0], row2["year"].values[0]]
            if splitparams:
                params = [
                    row1["yes/no"].values[0],
                    row2["dcrest"].values[0],
                    row2["dberm"].values[0],
                    row2["beta_target"].values[0],
                    row2["transition_level"].values[0],
                ]
            else:
                params = [row1["params"].values[0], row2["params"].values[0]]
            Cost = [row1["cost"].values[0], row2["cost"].values[0]]
            # combine betas
            # take maximums of mechanisms except if it is about StabilityInner for partial Stability Screen
            betas = []
            years = []

            for ij in partials.columns:
                if ij[0] != "Section" and ij[1] != "":  # It is a beta value
                    # TODO make clean. Quick fix to fix incorrect treatment of vertical geotextile.
                    # VSG is idx in MeasureTable
                    if (row1["type"].values == "Vertical Geotextile") and (
                        ij[0] == "Piping"
                    ):
                        idx = solutions.measure_table.loc[
                            solutions.measure_table["Name"]
                            == "Verticaal Zanddicht Geotextiel"
                        ].index.values[0]
                        Pf_VSG = solutions.measures[idx].parameters["Pf_solution"]
                        P_VSG = solutions.measures[idx].parameters["P_solution"]
                        pf = (1 - P_VSG) * Pf_VSG + P_VSG * beta_to_pf(row2[ij])
                        beta = pf_to_beta(pf)
                    else:
                        beta = np.maximum(row1[ij], row2[ij])
                    years.append(ij[1])
                    betas.append(beta)

            # next update section probabilities
            for ij in partials.columns:
                if ij[0] == "Section":  # It is a beta value
                    # where year in years is the same as ij[1]
                    indices = [indices for indices, x in enumerate(years) if x == ij[1]]
                    ps = beta_to_pf(np.array(betas)[indices])
                    p = np.sum(ps)  # TODO replace with correct formula
                    betas.append(pf_to_beta(p))
                    # if ProbabilisticFunctions.pf_to_beta(p)-np.max([row1[ij],row2[ij]]) > 1e-8:
                    #     pass
            if splitparams:
                in1 = [
                    ID,
                    types,
                    "combined",
                    year,
                    params[0],
                    params[1],
                    params[2],
                    params[3],
                    params[4],
                    Cost,
                ]
            else:
                in1 = [ID, types, "combined", year, params, Cost]

            allin = pd.DataFrame([in1 + betas], columns=combinables.columns)
            _combined_measures = pd.concat((_combined_measures, allin))
    return _combined_measures


def make_traject_df(traject: DikeTraject, cols):
    # cols = cols[1:]
    sections = []

    for i in traject.sections:
        sections.append(i.name)

    mechanisms = list(traject.mechanism_names) + ["Section"]
    df_index = pd.MultiIndex.from_product(
        [sections, mechanisms], names=["name", "mechanism"]
    )
    _traject_probability = pd.DataFrame(columns=cols, index=df_index)

    for _section in traject.sections:
        for _mechanism_name in mechanisms:
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
    options: dict[str, pd.DataFrame], available_mechanism_names: list[str]
) -> tuple[list[dict[str, pd.DataFrame]], list[dict[str, pd.DataFrame]]]:
    """Splits the options for the measures.

    Args:
        options (_type_): The available options to split.
        available_mechanism_names (list[str]): The collection of the names of the available mechanisms for the evaluation.

    Returns:
        list[dict[str, pd.DataFrame]]: The collection of splitted options_dependent
        list[dict[str, pd.DataFrame]]: The collection of splitted options_independent
    """

    def get_dropped_dependent_options(
        available_mechanism_names: list[str],
    ) -> list[str]:
        options = []
        for available_mechanism_name in available_mechanism_names:
            if available_mechanism_name in ["StabilityInner", "Piping"]:
                options.append(available_mechanism_name)

        options.append("Section")
        return options

    def get_dropped_independent_options(
        available_mechanism_names: list[str],
    ) -> list[str]:
        options = []
        for available_mechanism_name in available_mechanism_names:
            if available_mechanism_name in ["Overflow", "Revetment"]:
                options.append(available_mechanism_name)

        options.append("Section")
        return options

    options_dependent = copy.deepcopy(options)
    options_independent = copy.deepcopy(options)
    for i in options:
        # filter all different measures for dependent
        options_dependent[i] = options_dependent[i].loc[
            options[i]["class"] != "combined"
        ]
        options_dependent[i] = options_dependent[i].loc[
            (options[i]["type"] == "Diaphragm Wall")
            | (options[i]["type"] == "Custom")
            | (options[i]["dberm"] == 0)
        ]

        # now we filter all independent measures
        # first all crest heights are thrown out
        options_independent[i] = options_independent[i].loc[
            (options_independent[i]["dcrest"] == 0.0)
            | (options_independent[i]["dcrest"] == -999)
            | (
                (options_independent[i]["class"] == "combined")
                & (options_independent[i]["dberm"] == 0)
            )
        ]
        # filter out revetments from all independent measures
        if "Revetment" in available_mechanism_names:
            for key in ["transition_level", "n_pf_stone"]:  # TODO check keys
                options_independent[i] = options_independent[i].loc[
                    (options_independent[i][key] == 0.0)
                    | (options_independent[i][key] == -999)
                    | (
                        (options_independent[i]["class"] == "combined")
                        & (options_independent[i][key] == 0)
                    )
                ]

        # subtract startcosts, only for dependent.
        startcosts = np.min(
            options_dependent[i][
                (options_dependent[i]["type"] == "Soil reinforcement")
            ]["cost"]
        )
        options_dependent[i]["cost"] = np.where(
            options_dependent[i]["type"] == "Soil reinforcement",
            np.subtract(options_dependent[i]["cost"], startcosts),
            options_dependent[i]["cost"],
        )

        # if an option has a stability screen, the costs for height are too high. This has to be adjusted. We do this
        # for all soil reinforcements. costs are not discounted yet, so we can disregard the year of the investment:
        for ij in np.unique(
            options_dependent[i].loc[
                options_dependent[i]["type"] == "Soil reinforcement"
            ]["dcrest"]
        ):
            options_dependent[i].loc[
                options_dependent[i]["dcrest"] == ij, "cost"
            ] = np.min(
                options_dependent[i].loc[options_dependent[i]["dcrest"] == ij]["cost"]
            )

        options_independent[i] = options_independent[i].reset_index(drop=True)
        options_dependent[i] = options_dependent[i].reset_index(drop=True)

        # loop for the independent stuff:
        newcosts = []
        for ij in options_independent[i].index:
            if (
                options_independent[i].iloc[ij]["type"].values[0]
                == "Soil reinforcement"
            ):
                newcosts.append(options_independent[i].iloc[ij]["cost"].values[0])
            elif options_independent[i].iloc[ij]["class"].values[0] == "combined":
                newcosts.append(
                    [
                        options_independent[i].iloc[ij]["cost"].values[0][0],
                        options_independent[i].iloc[ij]["cost"].values[0][1],
                    ]
                )
            else:
                newcosts.append(options_independent[i].iloc[ij]["cost"].values[0])
        options_independent[i]["cost"] = newcosts
        # only keep reliability of relevant mechanisms in dictionary
        options_dependent[i].drop(
            get_dropped_dependent_options(available_mechanism_names), axis=1, level=0
        )
        options_independent[i].drop(
            get_dropped_independent_options(available_mechanism_names), axis=1, level=0
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
    for i in config.mechanisms:
        if i == "Overflow":
            init_overflow_risk[n, :] = strategy.RiskOverflow[n, sh, :]
        elif i == "Revetment":
            init_revetment_risk[n, :] = strategy.RiskRevetment[n, sh, :]
        else:
            init_geo_risk[n, :] = strategy.RiskGeotechnical[n, sg, :]
    return init_overflow_risk, init_revetment_risk, init_geo_risk


def update_probability(init_probability, strategy, index):
    """index = [n,sh,sg]"""
    for i in init_probability:
        from scipy.stats import norm

        # plt.plot(-norm.ppf(init_probability[i][index[0],:]), 'r')
        if i == "Overflow":
            init_probability[i][index[0], :] = strategy.Pf[i][index[0], index[1], :]
        else:
            init_probability[i][index[0], :] = strategy.Pf[i][index[0], index[2], :]
        # plt.plot(-norm.ppf(init_probability[i][index[0],:]),'b')
        # plt.savefig('Beta ' + i + str(index) + '.png')
        # plt.close()
    return init_probability


def overflow_bundling(
    strategy,  #: StrategyBase nb: activating this type hint gives a circular dependency
    init_overflow_risk,
    existing_investment,
    life_cycle_cost,
    traject: DikeTraject,
):
    """
    Alternative routine that only uses the reliability to determine what measures are allowed.
    The logic of this version is that measures are not restricted by type,
    but that geotechnical reliability may not decrease compared to the already chosen option
    """

    # ensure that life_cycle_cost is not modified
    life_cycle_cost = copy.deepcopy(life_cycle_cost)
    # Step 1: fill an array of size (n,2) with sh and sg of existing investments per section in order to properly filter
    # the viable options per section
    existing_investments = np.zeros(
        (np.size(life_cycle_cost, axis=0), 2), dtype=np.int32
    )
    if len(existing_investment) > 0:
        for i in range(0, len(existing_investment)):
            existing_investments[existing_investment[i][0], 0] = existing_investment[i][
                1
            ]  # sh
            existing_investments[existing_investment[i][0], 1] = existing_investment[i][
                2
            ]  # sg

    # Step 2: for each section, determine the sorted_indices of the min to max LCC. Note that this could also be based on TC but the performance is good as is.
    # first make the proper arrays for sorted_indices (sh), corresponding sg indices and the LCC for each section.
    sorted_sh = np.full(tuple(life_cycle_cost.shape[0:2]), 999, dtype=int)
    LCC_values = np.zeros((life_cycle_cost.shape[0],))
    sg_indices = np.full(tuple(life_cycle_cost.shape[0:2]), 999, dtype=int)

    # loop over the sections
    for i in range(0, len(traject.sections)):
        # get all geotechnical options for this section:
        GeotechnicalOptions = strategy.options_geotechnical[traject.sections[i].name]
        HeightOptions = strategy.options_height[traject.sections[i].name]
        # if there is already an investment we ensure that the reliability for none of the mechanisms is lower than the current investment
        if any(existing_investments[i, :] > 0):
            # if there is a GeotechnicalOption in place, we need to filter the options based on the current investment
            if existing_investments[i, 1] > 0:
                investment_id = (
                    existing_investments[i, 1] - 1
                )  # note that matrix indices in existing_investments are always 1 higher than the investment id
                current_investment_geotechnical = GeotechnicalOptions.iloc[
                    investment_id
                ]
                current_investment_stability = current_investment_geotechnical[
                    "StabilityInner"
                ]
                current_investment_piping = current_investment_geotechnical["Piping"]
                # check if all rows in comparison only contain True values
                comparison_geotechnical = (
                    GeotechnicalOptions.StabilityInner >= current_investment_stability
                ) & (GeotechnicalOptions.Piping >= current_investment_piping)
                available_measures_geotechnical = comparison_geotechnical.all(
                    axis=1
                )  # df indexing, so a False should be added before
            else:
                available_measures_geotechnical = pd.Series(
                    np.ones(len(GeotechnicalOptions), dtype=bool)
                )

            # same for HeightOptions
            if existing_investments[i, 0] > 0:
                # exclude rows for height options that are not safer than current
                current_investment_overflow = HeightOptions.iloc[
                    existing_investments[i, 0] - 1
                ]["Overflow"]
                # TODO turn on revetment once the proper data is available.
                # current_investment_revetment = HeightOptions.iloc[existing_investments[i, 0] - 1]['Revetment']
                current_investment_revetment = HeightOptions.iloc[
                    existing_investments[i, 0] - 1
                ]["Overflow"]
                # check if all rows in comparison only contain True values
                comparison_height = (
                    HeightOptions.Overflow > current_investment_overflow
                )  # & (HeightOptions.Revetment >= current_investment_revetment)
                available_measures_height = comparison_height.any(axis=1)
            else:  # if there is no investment in height, all options are available
                available_measures_height = pd.Series(
                    np.ones(len(HeightOptions), dtype=bool)
                )

            # now replace the life_cycle_cost where available_measures_height is False with a very high value:
            # the reliability for overflow has to increase so we do not want to pick these measures.
            life_cycle_cost[
                i, available_measures_height[~available_measures_height].index + 1, :
            ] = 1e99

            # next we get the ids for the possible geotechnical measures
            ids = (
                available_measures_geotechnical[
                    available_measures_geotechnical
                ].index.values
                + 1
            )

            # we get a matrix with the LCC values, and get the order of sh measures:
            lcc_subset = life_cycle_cost[i, :, ids].T
            sh_order = np.argsort(np.min(lcc_subset, axis=1))
            sg_indices[i, :] = np.array(ids)[np.argmin(lcc_subset, axis=1)][sh_order]
            sorted_sh[i, :] = sh_order
            sorted_sh[i, :] = np.where(
                np.sort(np.min(lcc_subset, axis=1)) > 1e60, 999, sorted_sh[i, :]
            )
        elif np.max(existing_investments[i, :]) == 0:  # nothing has been invested yet
            sg_indices[i, :] = np.argmin(life_cycle_cost[i, :, :], axis=1)
            LCCs = np.min(life_cycle_cost[i, :, :], axis=1)
            sorted_sh[i, :] = np.argsort(LCCs)
            sorted_sh[i, :] = np.where(
                np.sort(LCCs) > 1e60, 999, sorted_sh[i, 0 : len(LCCs)]
            )
            sg_indices[i, 0 : len(LCCs)] = sg_indices[i, 0 : len(LCCs)][
                np.argsort(LCCs)
            ]
        else:
            logging.error(
                "Unknown measure type in overflow bundling (error can be removed?)"
            )
    new_overflow_risk = copy.deepcopy(init_overflow_risk)

    # Step 3: determine various bundles for overflow:

    # first initialize som values
    index_counter = np.zeros(
        (len(traject.sections),), dtype=np.int32
    )  # counter that keeps track of the next cheapest option for each section
    run_number = 0  # used for counting the loop
    counter_list = []  # used to store the bundle indices
    BC_list = []  # used to store BC for each bundle
    weak_list = []  # used to store index of weakest section

    # here we start the loop. Note that we rarely make it to run 100, for larger problems this limit might need to be increased
    while run_number < 100:
        # get weakest section
        ind_weakest = np.argmax(np.sum(new_overflow_risk, axis=1))

        # We should increase the measure at the weakest section, but only if we have not reached the end of the array yet:
        if sorted_sh.shape[1] - 1 > index_counter[ind_weakest]:
            index_counter[ind_weakest] += 1
            # take next step, exception if there is no valid measure. In that case exit the routine.
            if sorted_sh[ind_weakest, index_counter[ind_weakest]] == 999:
                logging.error(
                    "Bundle quit after {} steps, weakest section has no more available measures".format(
                        run_number
                    )
                )
                break
        else:
            logging.error(
                "Bundle quit after {} steps, weakest section has no more available measures".format(
                    run_number
                )
            )
            break

        # insert next cheapest measure from sorted list into overflow risk, then compute the LCC value and BC
        new_overflow_risk[ind_weakest, :] = strategy.RiskOverflow[
            ind_weakest, sorted_sh[ind_weakest, index_counter[ind_weakest]], :
        ]
        LCC_values[ind_weakest] = np.min(
            life_cycle_cost[
                ind_weakest,
                sorted_sh[ind_weakest, index_counter[ind_weakest]],
                sg_indices[ind_weakest, index_counter[ind_weakest]],
            ]
        )
        BC = (
            np.sum(np.max(init_overflow_risk, axis=0))
            - np.sum(np.max(new_overflow_risk, axis=0))
        ) / np.sum(LCC_values)
        # store results of step:
        if np.isnan(BC):
            BC_list.append(0.0)
        else:
            BC_list.append(BC)
        weak_list.append(ind_weakest)

        # store the bundle indices, do -1 as index_counter contains the NEXT step
        counter_list.append(copy.deepcopy(index_counter))

        # Strategy.get_measure_from_index((ind_weakest, sorted_sh[ind_weakest, index_counter[ind_weakest]],
        #                                  sg_indices[ind_weakest, index_counter[ind_weakest]]), print_measure=True)

        # in the next step, the next measure should be taken for this section
        run_number += 1

    # take the final index from the list, where BC is max
    if len(BC_list) > 0:
        ind = np.argwhere(BC_list == np.max(BC_list))[0][0]
        final_index = counter_list[ind]
        # convert measure_index to sh based on sorted_indices
        sg_index = np.zeros((len(traject.sections),))
        measure_index = np.zeros((np.size(life_cycle_cost, axis=0),), dtype=np.int32)
        for i in range(0, len(measure_index)):
            if final_index[i] != 0:  # a measure was taken
                measure_index[i] = sorted_sh[i, final_index[i]]
                sg_index[i] = sg_indices[i, final_index[i]]
            else:  # no measure was taken
                measure_index[i] = existing_investments[i, 0]
                sg_index[i] = existing_investments[i, 1]

        measure_index = (
            np.append(measure_index, sg_index)
            .reshape((2, len(traject.sections)))
            .T.astype(np.int32)
        )
        BC_out = np.max(BC_list)
    else:
        BC_out = 0
        measure_index = []
        logging.warning("No more measures for weakest overflow section")

    return measure_index, BC_out, BC_list
