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
    TrajectProbability = pd.DataFrame(columns=cols, index=df_index)

    for i in traject.sections:
        for j in mechanisms:
            TrajectProbability.loc[(i.name, j)] = list(
                i.section_reliability.SectionReliability.loc[j]
            )

    return TrajectProbability


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
        logging.warn("No damage defined.")

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
    init_overflow_risk, init_geo_risk, strategy, n, sh, sg, config: VrtoolConfig
):
    for i in config.mechanisms:
        if i == "Overflow":
            init_overflow_risk[n, :] = strategy.RiskOverflow[n, sh, :]
        else:
            init_geo_risk[n, :] = strategy.RiskGeotechnical[n, sg, :]
    return init_overflow_risk, init_geo_risk


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
    strategy,
    init_overflow_risk,
    existing_investment,
    life_cycle_cost,
    traject: DikeTraject,
):
    """"""
    """ Alternative routine that only uses the reliability to determine what measures are allowed.
     The logic of this version is that measures are not restricted by type, but that geotechnical reliability may not decrease compared to the already chosen option"""
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
    sorted_sh = np.empty(
        (np.size(life_cycle_cost, axis=0), np.size(life_cycle_cost, axis=1)),
        dtype=np.int32,
    )
    sorted_sh.fill(999)
    LCC_values = np.zeros((np.size(life_cycle_cost, axis=0),))
    sg_indices = np.empty(
        (np.size(life_cycle_cost, axis=0), np.size(life_cycle_cost, axis=1)),
        dtype=np.int32,
    )
    sg_indices.fill(999)

    # loop over the sections
    for i in range(0, len(traject.sections)):
        extra_type = None
        index_existing = 0  # value is only used in 1 of the branches of the if statement, otherwise should be 0.
        # get the indices where safety is equal to no measure for stabilityinner & piping
        # if there are investments this loop is needed to deal with the fact that it can be an integer or list.

        # get all geotechnical options for this section:
        GeotechnicalOptions = strategy.options_geotechnical[traject.sections[i].name]
        HeightOptions = strategy.options_height[traject.sections[i].name]
        #if there is already an investment
        if any(existing_investments[i, :] > 0):
            investment_id = existing_investments[i, 1] - 1
            current_investment_geotechnical = GeotechnicalOptions.iloc[investment_id]
            current_investment_stability = current_investment_geotechnical["StabilityInner"]
            current_investment_piping = current_investment_geotechnical["Piping"]
            # check if all rows in comparison only contain True values
            comparison_geotechnical = (GeotechnicalOptions.StabilityInner >= current_investment_stability) & (
                    GeotechnicalOptions.Piping >= current_investment_piping)
            available_measures_geotechnical = comparison_geotechnical.all(
                axis=1)  # df indexing, so a False should be added before

            if existing_investments[i, 0] > 0:
                # exclude rows for height options that are not safer than current
                current_investment_height = HeightOptions.iloc[existing_investments[i, 0] - 1]['Overflow']
                # check if all rows in comparison only contain True values #TODO extend with revetment
                comparison_height = HeightOptions.Overflow > current_investment_height
                available_measures_height = comparison_height.all(axis=1)
            else: # if there is no investment in height, all options are available
                available_measures_height = pd.Series(np.ones(len(HeightOptions), dtype=bool))

            # now replace the life_cycle_cost where available_measures_height is False with a very high value: the reliability for overflow has to increase.
            life_cycle_cost[i, available_measures_height[~available_measures_height].index + 1, :] = 1e99

            #next we get the ids for the possible geotechnical measures
            ids = available_measures_geotechnical[available_measures_geotechnical].index.values + 1

            #we get a matrix with the LCC values, and get the order of sh measures:
            lcc_subset = life_cycle_cost[i, :, ids].T
            sh_order = np.argsort(np.min(lcc_subset, axis=1))
            sg_indices[i, :] = np.array(ids)[np.argmin(lcc_subset, axis=1)][sh_order]
            sorted_sh[i, :] = sh_order
            sorted_sh[i, :] = np.where(
                np.sort(np.min(lcc_subset, axis=1)) > 1e60, 999, sorted_sh[i, :]
            )
        elif np.max(existing_investments[i, :]) == 0: # nothing has been invested yet
            sg_indices[i, :] = np.argmin(life_cycle_cost[i, :, :], axis=1)
            LCCs = np.min(life_cycle_cost[i, :, :], axis=1)
            sorted_sh[i, :] = np.argsort(LCCs)
            sorted_sh[i, :] = np.where(
                np.sort(LCCs) > 1e60, 999, sorted_sh[i, 0: len(LCCs)]
            )
            sg_indices[i, 0: len(LCCs)] = sg_indices[i, 0: len(LCCs)][
                np.argsort(LCCs)
            ]
        else:
            logging.error("Unknown measure type in overflow bundling (error can be removed?)")
    new_overflow_risk = copy.deepcopy(init_overflow_risk)
    # print('New:')
    # print(sg_indices)
    # print(sorted_sh)
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
                    "Bundle quit after {} steps, weakest section has no more available measures".format(run_number)
                )
                break
        else:
            logging.error("Bundle quit after {} steps, weakest section has no more available measures".format(run_number))
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
        logging.warn("No more measures for weakest overflow section")
    return measure_index, BC_out, BC_list

def old_overflow_bundling(
    strategy,
    init_overflow_risk,
    existing_investment,
    life_cycle_cost,
    traject: DikeTraject,
):
    """Routine for bundling several measures for overflow to prevent getting stuck if many overflow-dominated
    sections have about equal reliability. A bundle is a set of measures (typically crest heightenings) at different sections.
    This routine is needed for mechanisms where the system reliability is computed as a series system with fully correlated components."""

    """Step 1: fill an array of size (n,2) with sh and sg of existing investments per section in order to properly filter
     the viable options per section"""
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

    """Step 2: for each section, determine the sorted_indices of the min to max LCC. Note that this could also be based on TC but the performance is good as is.
    # first make the proper arrays for sorted_indices (sh), corresponding sg indices and the LCC for each section."""
    sorted_sh = np.empty(
        (np.size(life_cycle_cost, axis=0), np.size(life_cycle_cost, axis=1)),
        dtype=np.int32,
    )
    sorted_sh.fill(999)
    LCC_values = np.zeros((np.size(life_cycle_cost, axis=0),))
    sg_indices = np.empty(
        (np.size(life_cycle_cost, axis=0), np.size(life_cycle_cost, axis=1)),
        dtype=np.int32,
    )
    sg_indices.fill(999)

    # loop over the sections
    for i in range(0, len(traject.sections)):
        extra_type = None
        index_existing = 0  # value is only used in 1 of the branches of the if statement, otherwise should be 0.
        # get the indices where safety is equal to no measure for stabilityinner & piping
        # if there are investments this loop is needed to deal with the fact that it can be an integer or list.

        # get all geotechnical options for this section:
        GeotechnicalOptions = strategy.options_geotechnical[traject.sections[i].name]
        HeightOptions = strategy.options_height[traject.sections[i].name]
        if any(existing_investments[i, :] > 0):
            investment_id = existing_investments[i, 1] - 1
            if isinstance(
                GeotechnicalOptions.iloc[investment_id]["year"].values[0], list
            ):
                #if it is a combined measure (list of years).
                # take last: this should be the soil reinforcement.
                if (
                    GeotechnicalOptions.iloc[investment_id]["type"].values[0][0]
                    == "Soil reinforcement"
                ):
                    logging.warn(
                        "First combined measure is a soil reinforcement. This might not result in the intended behaviour"
                    )
                current_type = GeotechnicalOptions.iloc[investment_id]["type"].values[
                    0
                ][1]
                extra_type = GeotechnicalOptions.iloc[investment_id]["type"].values[0][
                    0
                ]
                current_class = GeotechnicalOptions.iloc[investment_id]["class"].values[
                    0
                ]
                year_of_investment = GeotechnicalOptions.iloc[investment_id][
                    "year"
                ].values[0][-1]
            else:
                year_of_investment = GeotechnicalOptions.iloc[investment_id][
                    "year"
                ].values[0]
                current_type = GeotechnicalOptions.iloc[investment_id]["type"].values[0]
                current_class = GeotechnicalOptions.iloc[investment_id]["class"].values[
                    0
                ]
        else:
            # no investments yet
            year_of_investment = np.int32(0)
            current_type = None
            current_class = None

        # no changes allowed for Diaphragm Wall and Custom.

        if current_type in ["Diaphragm Wall", "Custom"]:
            # get costs for measure (1e99)
            LCC = life_cycle_cost[
                i, existing_investments[i, 0], existing_investments[i, 1]
            ]
            # TCs = np.add(LCCs, np.sum(Strategy.RiskOverflow[i, existing_investments[i,0]:, :], axis=1))
            # fill indices
            sg_indices[i, :].fill(existing_investments[i, 1])
            sorted_sh[i, :].fill(999)
            # no further action needed.

        # Soil reinforcement can only remain of same class. For t=20 it can be moved forward in time:
        elif (
            current_type
            in ["Soil reinforcement", "Soil reinforcement with stability screen"]
        ) and (
            extra_type == None
        ):  # soil reinforcement with stability screen
            # if in t=0, only t=0. Otherwise also options for moving to t=0
            #

            if year_of_investment == 0:
                # do not change geotechnical measure:
                sg_indices[i, :].fill(existing_investments[i, 1])

                # find indices of sh with same measure
                subset = HeightOptions.loc[
                    (HeightOptions["type"] == "Soil reinforcement")
                    & (HeightOptions["class"] == current_class)
                ]
                sh_opts = (
                    subset.loc[subset["year"] == year_of_investment].index.values + 1
                )
                LCCs = life_cycle_cost[i, sh_opts, existing_investments[i, 1]]
                sorted_sh[i, 0 : len(LCCs)] = sh_opts[np.argsort(LCCs)]
                sorted_sh[i, 0 : len(LCCs)] = np.where(
                    np.sort(LCCs) > 1e60, 999, sorted_sh[i, 0 : len(LCCs)]
                )
            elif year_of_investment == 20:
                # allow both the berm at this and the other time slot
                current_berm = GeotechnicalOptions.iloc[investment_id].dberm.values[0]
                sh_opts = (
                    HeightOptions.loc[
                        (HeightOptions["type"] == "Soil reinforcement")
                        & (HeightOptions["class"] == current_class)
                    ].index.values
                    + 1
                )
                sg_opts = (
                    GeotechnicalOptions.loc[
                        (GeotechnicalOptions["type"] == "Soil reinforcement")
                        & (GeotechnicalOptions["class"] == current_class)
                        * (GeotechnicalOptions["dberm"] == current_berm)
                    ].index.values
                    + 1
                )
                LCCs = life_cycle_cost[i, sh_opts, :][:, sg_opts]
                # order = np.dstack(np.unravel_index(np.argsort(LCCs.ravel()), (LCCs.shape[0], LCCs.shape[1])))
                order = np.unravel_index(np.argsort(LCCs.ravel()), (LCCs.shape))
                orderedLCCs = LCCs[order[0], order[1]]
                orderedLCCs = orderedLCCs[orderedLCCs < 1e60]

                sg_indices[i, 0 : len(orderedLCCs)] = sg_opts[
                    order[1][0 : len(orderedLCCs)]
                ]
                sorted_sh[i, 0 : len(orderedLCCs)] = sh_opts[
                    order[0][0 : len(orderedLCCs)]
                ]
                sorted_sh[i, 0 : len(orderedLCCs)] = np.where(
                    orderedLCCs > 1e60, 999, sh_opts[order[0][0 : len(orderedLCCs)]]
                )

        # For a stability screen, we should check if it can be extended with a berm or crest. Note that not allowing this might result in a local optimum.
        elif current_type == "Stability Screen":
            # check if there are options with a full reinforcement with a Stability screen. This should be a soil reinforcement with the same beta for StabilityInner in the year of investment

            beta_investment_year = (
                GeotechnicalOptions["StabilityInner", year_of_investment]
                .loc[
                    GeotechnicalOptions["ID"]
                    == GeotechnicalOptions["ID"][investment_id]
                ]
                .values[0]
            )
            ID_allowed = (
                GeotechnicalOptions.loc[
                    GeotechnicalOptions["StabilityInner", year_of_investment]
                    == beta_investment_year
                ]
                .loc[GeotechnicalOptions["class"] == "full"]["ID"]
                .values
            )
            if len(ID_allowed) > 0:
                ID_allowed = np.unique(ID_allowed)
                if len(ID_allowed) > 1:
                    ids = []
                    for ID_all in ID_allowed:
                        ids.append(
                            np.argwhere(GeotechnicalOptions["ID"].values == ID_all)
                        )
                    ids = np.concatenate(ids)
                else:
                    ids = np.argwhere(GeotechnicalOptions["ID"].values == ID_allowed)
                # convert to matrix indexing:
                ids = np.add(ids.reshape((len(ids),)), 1)
                testLCC = life_cycle_cost[i, existing_investments[i, 0] :, ids].T
                LCCs = np.min(testLCC, axis=1)
                sg_indices[i, :] = np.array(ids)[np.argmin(testLCC, axis=1)]
                sorted_sh[i, :] = np.argsort(LCCs) + index_existing
                sorted_sh[i, :] = np.where(
                    np.sort(LCCs) > 1e60, 999, sorted_sh[i, 0 : len(LCCs)]
                )
        elif (current_type == "Vertical Geotextile") or (
            extra_type == "Vertical Geotextile"
        ):
            if extra_type == "Vertical Geotextile":
                # There is already a soil reinforcement, we should keep the berm
                current_berm = GeotechnicalOptions.iloc[
                    existing_investments[i][1] - 1
                ].dberm.values[0]
                subset = GeotechnicalOptions.loc[
                    (GeotechnicalOptions["dberm"] == current_berm)
                    & (GeotechnicalOptions["class"] == "combined")
                ]
                if (
                    GeotechnicalOptions.iloc[
                        existing_investments[i][1] - 1
                    ].year.values[0][1]
                    == 0
                ):
                    # note not entirely robust
                    # remove the investment in year 20, we cannot postpone the existing berm.
                    for count, row in subset.iterrows():
                        years = row["year"].values[0]
                        if 20 in years:
                            subset = subset.drop(axis=0, index=row.name)

            else:
                # find the options in GeotechnicalOptions where VZG is combined with no berm
                subset = GeotechnicalOptions.loc[
                    (GeotechnicalOptions["dberm"] == 0.0)
                    & (GeotechnicalOptions["class"] == "combined")
                ]

            for count, row in subset.iterrows():
                types = row["type"].values[0]
                if not "Vertical Geotextile" in types:
                    subset = subset.drop(axis=0, index=row.name)
                else:
                    pass
            # matrix indices:
            ids = subset.index.values + 1
            #alternative
            # get LCC with correct geotechnical measure:
            LCC_1 = life_cycle_cost[i, :, ids].T

            # sort sg indices
            # sg_indices[i,0:len(ids)] = np.array(ids)[np.argsort(np.min(LCC_1,axis=0))]
            sh_order = np.argsort(np.min(LCC_1, axis=1))
            sg_indices[i, :] = np.array(ids)[np.argmin(LCC_1, axis=1)][sh_order]
            # find columnwise minimum of LCC_1, this is sh
            sorted_sh[i, :] = sh_order
            sorted_sh[i, :] = np.where(
                np.sort(np.min(LCC_1, axis=1)) > 1e60, 999, sorted_sh[i, :]
            )

        elif current_type == None:
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
            logging.error("This one is not covered yet!")

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
                    "Bundle quit, weakest section has no more available measures"
                )
                break
        else:
            logging.error("Bundle quit, weakest section has no more available measures")
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
        logging.warn("No more measures for weakest overflow section")

    return measure_index, BC_out, BC_list
