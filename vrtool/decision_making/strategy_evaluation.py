import copy
import logging

import numpy as np
import pandas as pd
import itertools

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

def revetment_combinations(partials, combinables):
    _combined_measures = pd.DataFrame(columns=combinables.columns)
    #all columns without a second index are attributes of the measure
    attribute_col_names = combinables.columns.get_level_values(0)[combinables.columns.get_level_values(1)==''].tolist()
    #years are those columns of level 2 with a second index that is not ''
    years = combinables.columns.get_level_values(1)[combinables.columns.get_level_values(1)!=''].unique().tolist()
    #mechanisms are those columns of level 1 with a second index that is not ''
    mechanisms = combinables.columns.get_level_values(0)[combinables.columns.get_level_values(1)!=''].unique().tolist()

    #make dict with attribute_col_names as keys and empty lists as values
    attribute_col_dict = {col:[] for col in attribute_col_names}

    #make dict with mechanisms as keys, sub dicts of years and then empty lists as values
    mechanism_beta_dict = {mechanism:{year:[] for year in years} for mechanism in mechanisms}
    count = 0
    # loop over partials
    for i, row1 in partials.iterrows():
        # combine with all combinables (in this case revetment measures)
        for j, row2 in combinables.iterrows():

            for col in attribute_col_names:
                if col == 'ID': #TODO maybe add type here as well and just concatenate the types as a string
                    attribute_value = f'{row1["ID"].values[0]}+{row2["ID"].values[0]}'
                elif col == 'class':
                    attribute_value = "combined"
                else:
                    #for all other columns we combine the lists and make sure that it is not nested
                    combined_data = row1[col].tolist() + row2[col].tolist()
                    attribute_value = list(itertools.chain.from_iterable(
                        itertools.repeat(x, 1) if (isinstance(x, str)) or (isinstance(x, int)) or (isinstance(x, float)) else x for x in
                        combined_data))
                    #drop all -999 values from attribute_value
                    attribute_value = [x for x in attribute_value if x != -999 and x != -999.0]
                    if len(attribute_value) == 1: #if there is only one value we take that value
                        attribute_value = attribute_value[0]
                attribute_col_dict[col].append(attribute_value)

            #then we fill the mechanism_beta_dict we ignore Section as mechanism, we do that as a last step on the dataframe
            for mechanism in mechanism_beta_dict.keys():
                if mechanism == "Section":
                    continue
                else:
                    for year in mechanism_beta_dict[mechanism].keys():
                        mechanism_beta_dict[mechanism][year].append(np.maximum(row1[mechanism,year],row2[mechanism,year]))

            count+=1

    attribute_col_df = pd.DataFrame.from_dict(attribute_col_dict)
    attribute_col_df.columns = pd.MultiIndex.from_tuples([(col,"") for col in attribute_col_df.columns])
    mechanism_beta_df = pd.DataFrame.from_dict(mechanism_beta_dict, orient="index").stack().to_frame()
    mechanism_beta_df = pd.DataFrame(mechanism_beta_df[0].values.tolist(), index=mechanism_beta_df.index)
    mechanism_beta_df.index = pd.MultiIndex.from_tuples(mechanism_beta_df.index)
    _combined_measures = pd.concat((attribute_col_df,mechanism_beta_df.transpose()),axis=1)
    for year in years:
        #compute the section beta
        betas_in_year = _combined_measures.loc[:,(_combined_measures.columns.get_level_values(0) != 'Section', _combined_measures.columns.get_level_values(1)==25)]
        pf_in_year = beta_to_pf(betas_in_year)
        section_beta = pf_to_beta(1 - np.prod(1-pf_in_year, axis=1))
        #add the section beta to the dataframe
        _combined_measures.loc[:,("Section",year)] = section_beta

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
            for key in ["transition_level", "beta_target"]:
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
        if i in ["Overflow","Revetment"]:
            init_probability[i][index[0], :] = strategy.Pf[i][index[0], index[1], :]
        else:
            init_probability[i][index[0], :] = strategy.Pf[i][index[0], index[2], :]
        # plt.plot(-norm.ppf(init_probability[i][index[0],:]),'b')
        # plt.savefig('Beta ' + i + str(index) + '.png')
        # plt.close()
    return init_probability
