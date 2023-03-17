import copy
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d

from vrtool.decision_making.solutions import Solutions
from vrtool.decision_making.strategy_evaluation import (
    calc_tc,
    measure_combinations,
    split_options,
)
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import (
    DikeTraject,
    calc_traject_prob,
    get_section_length_in_traject,
    plot_settings,
)
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf, pf_to_beta


class StrategyBase:
    """This defines a Strategy object, which can be allowed to evaluate a set of solutions/measures. There are currently 3 types:
    Greedy: a greedy optimization method is used here
    TargetReliability: a cross-sectional optimization in line with OI2014
    MixedInteger: a Mixed Integer optimization. Note that this has exponential runtime for large systems so it should not be used for more than approximately 13 sections.
    Note that this is the main class. Each type has a different subclass"""

    def __init__(self, type, config: VrtoolConfig):
        self.type = type
        self.r = config.discount_rate

        self.config = config
        self.OI_year = config.OI_year
        self.OI_horizon = config.OI_horizon
        self.mechanisms = config.mechanisms
        self.T = config.T
        self.beta_cost_settings = config.beta_cost_settings
        self.beta_or_prob = config.beta_or_prob
        self.LE_in_section = config.LE_in_section

    @staticmethod
    def get_measure_table(
        solutions_dict: dict, language: str, abbrev: bool
    ) -> pd.DataFrame:
        _overall_mt = pd.DataFrame([], columns=["ID", "Name"])
        for i in solutions_dict:
            _overall_mt = pd.concat([_overall_mt, solutions_dict[i].measure_table])
        _overall_mt: Union[pd.DataFrame, None, pd.Series] = _overall_mt.drop_duplicates(
            subset="ID"
        )
        _name_key = "Name"

        def _replace_with_abbreviations(str_to_replace: str) -> str:
            return (
                str_to_replace.replace("Grondversterking binnenwaarts", "Soil based")
                .replace("Grondversterking met stabiliteitsscherm", "Soil based + SS")
                .replace("Verticaal Zanddicht Geotextiel", "VSG")
                .replace("Zelfkerende constructie", "DW")
                .replace("Stabiliteitsscherm", "SS")
            )

        def _replace_without_abbreviations(str_to_replace: str) -> str:
            return (
                str_to_replace.replace("Grondversterking binnenwaarts", "Soil based")
                .replace(
                    "Grondversterking met stabiliteitsscherm",
                    "Soil inward + Stability Screen",
                )
                .replace(
                    "Verticaal Zanddicht Geotextiel", "Vertical Sandtight Geotextile"
                )
                .replace("Zelfkerende constructie", "Diaphragm Wall")
                .replace("Stabiliteitsscherm", "Stability Screen")
            )

        if (
            np.max(_overall_mt[_name_key].str.find("Grondversterking").values) > -1
        ) and (language == "EN"):
            if abbrev:
                _overall_mt[_name_key] = _replace_with_abbreviations(
                    _overall_mt[_name_key].str
                )
            else:
                _overall_mt[_name_key] = _replace_without_abbreviations(
                    _overall_mt[_name_key].str
                )
        return _overall_mt

    def get_measure_from_index(self, index, section_order=False, print_measure=False):
        """ "Converts an index (n,sh,sg) to a printout of the measure data"""
        if not section_order:
            print(
                "Warning: deriving section order from unordered dictionary. Might be wrong"
            )
            section_order = list(self.options_height.keys())
        section = section_order[index[0]]
        if index[1] > 0:
            sh = self.options_height[section_order[index[0]]].iloc[index[1] - 1]
        else:
            sh = "No measure"
            if index[2] > 0:
                sg = self.options_geotechnical[section_order[index[0]]].iloc[
                    index[2] - 1
                ]
                if not (
                    (sg.type.values == "Stability Screen")
                    or (sg.type.values == "Vertical Geotextile")
                ):
                    raise Exception("Illegal combination")
            else:
                sg = "No measure"
                print("SECTION {}".format(section))
                print(
                    "No measures are taken at this section: sh and sg are: {} and {}".format(
                        sh, sg
                    )
                )
                return (section, sh, sg)
        if index[2] > 0:
            sg = self.options_geotechnical[section_order[index[0]]].iloc[index[2] - 1]

        if print_measure:
            print("SECTION {}".format(section))
            if isinstance(sh, str):
                print("There is no measure for height")
            else:
                print(
                    "The measure for height is a {} in year {} with dcrest={} meters of the class {}.".format(
                        sh["type"].values[0],
                        sh["year"].values[0],
                        sh["dcrest"].values[0],
                        sh["class"].values[0],
                    )
                )
            if (
                (sg.type.values == "Vertical Geotextile")
                or (sg.type.values == "Diaphragm Wall")
                or (sg.type.values == "Stability Screen")
                or (sg.type.values == "Custom")
            ):
                print(" The geotechnical measure is a {}".format(sg.type.values[0]))
            elif isinstance(sg.type.values[0], list):  # VZG+Soil
                print(
                    " The geotechnical measure is a {} in year {} with a {} with dberm = {} in year {}".format(
                        sg.type.values[0][0],
                        sg.year.values[0][0],
                        sg.type.values[0][1],
                        sg.dberm.values[0],
                        sg.year.values[0][1],
                    )
                )
            elif sg.type.values == "Soil reinforcement":
                print(
                    " The geotechnical measure is a {} in year {} of class {} with dberm = {}".format(
                        sg.type.values[0],
                        sg.year.values[0],
                        sg["class"].values[0],
                        sg.dberm.values[0],
                    )
                )

        return (section, sh, sg)

    def combine(
        self,
        traject: DikeTraject,
        solutions_dict: Dict[str, Solutions],
        filtering="off",
        splitparams=False,
    ):
        # This routine combines 'combinable' solutions to options with two measures (e.g. VZG + 10 meter berm)
        self.options = {}

        cols = list(
            solutions_dict[list(solutions_dict.keys())[0]]
            .MeasureData["Section"]
            .columns.values
        )

        # measures at t=0 (2025) and t=20 (2045)
        # for i in range(0, len(traject.sections)):
        for i, section in enumerate(traject.sections):

            # Step 1: combine measures with partial measures
            combinables = solutions_dict[section.name].MeasureData.loc[
                solutions_dict[section.name].MeasureData["class"] == "combinable"
            ]
            partials = solutions_dict[section.name].MeasureData.loc[
                solutions_dict[section.name].MeasureData["class"] == "partial"
            ]
            if self.__class__.__name__ == "TargetReliabilityStrategy":
                combinables = combinables.loc[
                    solutions_dict[section.name].MeasureData["year"] == self.OI_year
                ]
                partials = partials.loc[
                    solutions_dict[section.name].MeasureData["year"] == self.OI_year
                ]

            combinedmeasures = measure_combinations(
                combinables,
                partials,
                solutions_dict[section.name],
                splitparams=splitparams,
            )
            # make sure combinable, mechanism and year are in the MeasureData dataframe
            # make a strategies dataframe where all combinable measures are combined with partial measures for each timestep
            # if there is a measureid that is not known yet, add it to the measure table

            existingIDs = solutions_dict[section.name].measure_table["ID"].values
            IDs = np.unique(combinedmeasures["ID"].values)
            if len(IDs) > 0:
                for ij in IDs:
                    if ij not in existingIDs:
                        indexes = ij.split("+")
                        name = (
                            solutions_dict[section.name]
                            .measure_table.loc[
                                solutions_dict[traject.sections[i].name].measure_table[
                                    "ID"
                                ]
                                == indexes[0]
                            ]["Name"]
                            .values[0]
                            + "+"
                            + solutions_dict[section.name]
                            .measure_table.loc[
                                solutions_dict[traject.sections[i].name].measure_table[
                                    "ID"
                                ]
                                == indexes[1]
                            ]["Name"]
                            .values[0]
                        )
                        solutions_dict[section.name].measure_table.loc[
                            len(solutions_dict[traject.sections[i].name].measure_table)
                            + 1
                        ] = name
                        solutions_dict[section.name].measure_table.loc[
                            len(solutions_dict[traject.sections[i].name].measure_table)
                        ]["ID"] = ij

            StrategyData = copy.deepcopy(solutions_dict[section.name].MeasureData)
            if self.__class__.__name__ == "TargetReliabilityStrategy":
                StrategyData = StrategyData.loc[StrategyData["year"] == self.OI_year]

            StrategyData = pd.concat((StrategyData, combinedmeasures))
            if filtering == "on":
                StrategyData = copy.deepcopy(StrategyData)
                StrategyData = StrategyData.reset_index(drop=True)
                LCC = calc_tc(StrategyData)
                ind = np.argsort(LCC)
                LCC_sort = LCC[ind]
                StrategyData = StrategyData.iloc[ind]
                beta_max = StrategyData["Section"].ix[0].values
                indexes = []

                for j in StrategyData.index:
                    if np.any(beta_max < StrategyData["Section"].ix[j].values - 0.01):
                        # measure has sense at some point in time
                        beta_max = np.maximum(
                            beta_max, StrategyData["Section"].ix[j].values - 0.01
                        )
                        indexes.append(i)
                    else:
                        # inefficient measure
                        pass

                StrategyData = StrategyData.ix[indexes]
                StrategyData = StrategyData.sort_index()

            self.options[section.name] = StrategyData.reset_index(drop=True)

    def evaluate(
        self,
        traject: DikeTraject,
        solutions_dict: Dict[str, Solutions],
        OI_horizon=50,
        splitparams=False,
        setting="fast",
    ):
        raise Exception(
            "General strategy can not be evaluated. Please make an object of the desired subclass (GreedyStrategy/MixedIntegerStrategy/TargetReliabilityStrategy"
        )

    def make_optimization_input(self, traject: DikeTraject):
        """This subroutine organizes the input into an optimization problem such that it can be accessed by the evaluation algorithm"""

        # TODO Currently incorrectly combined measures with sh = 0.5 crest and sg 0.5 crest + geotextile have not cost 1e99. However they
        #  do have costs higher than the correct option (sh=0m, sg=0.5+VZG) so they will never be selected. This
        #  should be fixed though

        self.options_height, self.options_geotechnical = split_options(self.options)

        N = len(self.options)  # Number of dike sections
        T = np.max(self.T)  # Number of time steps
        Sh = 0
        Sg = 0
        # #Number of strategies (maximum for all dike sections), for geotechnical and height
        Sh = np.max(
            [
                np.max([Sh, np.max(len(self.options_height[i]))])
                for i in self.options_height.keys()
            ]
        )
        Sg = np.max(
            [
                np.max([Sg, np.max(len(self.options_geotechnical[i]))])
                for i in self.options_geotechnical.keys()
            ]
        )

        # probabilities [N,S,T]
        self.Pf = {}
        for i in self.mechanisms:
            if i == "Overflow":
                self.Pf[i] = np.full((N, Sh + 1, T), 1.0)
            else:
                self.Pf[i] = np.full((N, Sg + 1, T), 1.0)

        # fill values
        # TODO Think about the initial condition and whether this should be added separately or teh 0,
        #  0 soil reinforcement also suffices.
        keys = list(self.options.keys())

        # get all probabilities. Interpolate on beta per section, then combine p_f
        betas = {}
        for n in range(0, N):
            for i in self.mechanisms:
                len_beta1 = traject.sections[
                    n
                ].section_reliability.SectionReliability.shape[1]
                beta1 = (
                    traject.sections[n]
                    .section_reliability.SectionReliability.loc[i]
                    .values.reshape((len_beta1, 1))
                    .T
                )  # Initial
                # condition with no measure
                if i == "Overflow":
                    beta2 = self.options_height[keys[n]][i]
                    # All solutions
                else:
                    beta2 = self.options_geotechnical[keys[n]][i]  # All solutions
                betas[i] = np.concatenate((beta1, beta2), axis=0)
                if np.shape(betas[i])[1] != T:
                    betas[i] = interp1d(self.T, betas[i])(np.arange(0, T, 1))
                self.Pf[i][n, 0 : np.size(betas[i], 0), :] = beta_to_pf(betas[i])

        # Costs of options [N,Sh,Sg]
        self.LCCOption = np.full((N, Sh + 1, Sg + 1), 1e99)
        for n in range(0, len(keys)):
            self.LCCOption[n, 0, 0] = 0.0
            LCC_sh = calc_tc(self.options_height[keys[n]])
            LCC_sg = calc_tc(self.options_geotechnical[keys[n]])
            # LCC_tot = calcTC(self.options[keys[n]])
            for sh in range(0, len(self.options_height[keys[n]])):
                # if it is a full type, it should only be combined with another full of the same type
                if self.options_height[keys[n]].iloc[sh]["class"].values[0] == "full":
                    full = True
                else:
                    full = False
                # if (self.options_height[keys[n]].iloc[sh]['type'].values[0] == 'Diaphragm Wall') | (
                #         self.options_height[keys[n]].iloc[sh]['type'].values[0] == 'Stability Screen'):
                #     full_structure = True
                # else:
                #     full_structure = False
                for sg in range(0, len(self.options_geotechnical[keys[n]])):  # Sg):
                    # if sh is a diaphragm wall, only a diaphragm wall can be taken for sg
                    if full:
                        # if the type is different it is not a possibility:
                        if (
                            self.options_geotechnical[keys[n]]
                            .iloc[sg]["type"]
                            .values[0]
                            != self.options_height[keys[n]].iloc[sh]["type"].values[0]
                        ) or (
                            self.options_geotechnical[keys[n]]
                            .iloc[sg]["year"]
                            .values[0]
                            != self.options_height[keys[n]].iloc[sh]["year"].values[0]
                        ):
                            pass  # do not change value, impossible combination (keep at 1e99)

                        else:
                            # if the type is a soil reinforcement, the year has to be the same
                            if (
                                self.options_geotechnical[keys[n]]
                                .iloc[sg]["type"]
                                .values[0]
                                == "Soil reinforcement"
                            ):
                                if (
                                    self.options_geotechnical[keys[n]]
                                    .iloc[sg]["year"]
                                    .values[0]
                                    == self.options_height[keys[n]]
                                    .iloc[sh]["year"]
                                    .values[0]
                                ) & (
                                    self.options_geotechnical[keys[n]]
                                    .iloc[sg]["class"]
                                    .values[0]
                                    == "full"
                                ):
                                    self.LCCOption[n, sh + 1, sg + 1] = (
                                        LCC_sh[sh] + LCC_sg[sg]
                                    )  # only use the costs once
                                else:
                                    pass  # not allowed
                            else:  # Diaphragm wall
                                self.LCCOption[n, sh + 1, sg + 1] = LCC_sh[
                                    sh
                                ]  # only use the costs once
                    # if sg is a plain geotextile or stability screen, it can only be combined with no measure for height, otherwise it
                    # would be a combined measure
                    elif (
                        self.options_geotechnical[keys[n]].iloc[sg]["type"].values[0]
                        == "Vertical Geotextile"
                    ) or (
                        self.options_geotechnical[keys[n]].iloc[sg]["type"].values[0]
                        == "Stability Screen"
                    ):
                        # can only be combined with no measure for height
                        self.LCCOption[n, 0, sg + 1] = LCC_sg[sg]
                    # if sg is a combined measure the soil reinforcement timing should be aligned:
                    elif (
                        self.options_geotechnical[keys[n]].iloc[sg]["class"].values[0]
                        == "combined"
                    ):
                        # check if the time of the soil reinforcement part is the same as for the height one
                        # first extract the index of the soil reinforcement
                        index = np.argwhere(
                            np.array(
                                self.options_geotechnical[keys[n]]
                                .iloc[sg]["type"]
                                .values[0]
                            )
                            == "Soil reinforcement"
                        )[0][0]
                        if (
                            self.options_geotechnical[keys[n]]
                            .iloc[sg]["year"]
                            .values[0][index]
                            == self.options_height[keys[n]].iloc[sh]["year"].values[0]
                        ):
                            if (
                                self.options_geotechnical[keys[n]]
                                .iloc[sg]["dcrest"]
                                .values
                                > 0.0
                            ):
                                if (
                                    self.options_geotechnical[keys[n]]
                                    .iloc[sg]["dcrest"]
                                    .values
                                    == self.options_height[keys[n]]
                                    .iloc[sh]["dcrest"]
                                    .values
                                ):
                                    self.LCCOption[n, sh + 1, sg + 1] = LCC_sg[
                                        sg
                                    ]  # only use the costs once
                                else:
                                    self.LCCOption[n, sh + 1, sg + 1] = 1e99
                            else:
                                self.LCCOption[n, sh + 1, sg + 1] = (
                                    LCC_sh[sh] + LCC_sg[sg]
                                )  # only use the costs once
                        else:
                            pass  # dont change, impossible combi
                        # if sg is combinable, it should only be combined with sh that have the same year
                    elif (
                        self.options_geotechnical[keys[n]].iloc[sg]["class"].values[0]
                        == "combinable"
                    ):
                        if (
                            self.options_geotechnical[keys[n]]
                            .iloc[sg]["year"]
                            .values[0]
                            == self.options_height[keys[n]].iloc[sh]["year"].values[0]
                        ):
                            self.LCCOption[n, sh + 1, sg + 1] = (
                                LCC_sh[sh] + LCC_sg[sg]
                            )  # only use the costs once
                        else:
                            pass
                    elif (
                        self.options_geotechnical[keys[n]].iloc[sg]["class"].values[0]
                        == "full"
                    ):
                        pass  # not allowed as the sh is not 'full'
                    else:
                        # if sg is a diaphragm wall cost should be only accounted for once:
                        if (
                            self.options_geotechnical[keys[n]]
                            .iloc[sg]["type"]
                            .values[0]
                            != "Diaphragm Wall"
                        ):
                            self.LCCOption[n, sh + 1, sg + 1] = (
                                LCC_sh[sh] + LCC_sg[sg]
                            )  # only use the costs once
                        else:
                            pass

        # Decision Variables for executed options [N,Sh] & [N,Sg]
        self.Cint_h = np.zeros((N, Sh))
        self.Cint_g = np.zeros((N, Sg))

        # Decision Variable for weakest overflow section with dims [N,Sh]
        self.Dint = np.zeros((N, Sh))

        # add discounted damage [T,]
        self.D = np.array(
            traject.general_info["FloodDamage"]
            * (1 / ((1 + VrtoolConfig.discount_rate) ** np.arange(0, T, 1)))
        )

        # expected damage for overflow and for piping & slope stability
        # self.RiskGeotechnical = np.zeros((N,Sg+1,T))
        self.RiskGeotechnical = (
            1 - np.multiply(1 - self.Pf["StabilityInner"], 1 - self.Pf["Piping"])
        ) * np.tile(self.D.T, (N, Sg + 1, 1))

        self.RiskOverflow = self.Pf["Overflow"] * np.tile(
            self.D.T, (N, Sh + 1, 1)
        )  # np.zeros((N,Sh+1,T))
        # add a few general parameters
        self.opt_parameters = {"N": N, "T": T, "Sg": Sg + 1, "Sh": Sh + 1}

    def make_solution(self, csv_path, step=False, type="Final"):
        """This is a routine to write the results for different types of solutions. It provides a dataframe with for each section the final measure.
        There are 3 types:
        FinalSolution: which is the result in the last step of the optimization
        OptimalSolution: the result with the lowest total cost
        SatisfiedStandardSolution: the result at which the reliability requirement is met.
        Note that if type is not Final the step parameter has to be defined."""

        if (type != "Final") and not step:
            raise Exception(
                "Error: input for make solution is inconsistent. If type is not Final, step should be provided"
            )

        if step:
            AllMeasures = copy.deepcopy(self.TakenMeasures.iloc[0:step])
        else:
            AllMeasures = copy.deepcopy(self.TakenMeasures)
        # sections = np.unique(AllMeasures['Section'][1:])
        sections = list(self.options.keys())
        Solution = pd.DataFrame(columns=AllMeasures.columns)
        for section in sections:
            lines = AllMeasures.loc[AllMeasures["Section"] == section]
            if len(lines) > 1:
                lcctot = np.sum(lines["LCC"])
                lines.loc[lines.index.values[-1], "LCC"] = lcctot
                lines.loc[lines.index.values[-1], "BC"] = np.nan
                Solution = pd.concat([Solution, lines[-1:]])
            elif len(lines) == 0:
                lines = pd.DataFrame(
                    np.array(
                        [
                            np.nan,
                            section,
                            0,
                            "No Measure",
                            -999.0,
                            -999.0,
                            -999.0,
                            -999.0,
                            0.0,
                        ]
                    ).reshape(1, len(Solution.columns)),
                    columns=Solution.columns,
                )
                Solution = pd.concat([Solution, lines])
            else:
                Solution = pd.concat([Solution, lines])
                Solution.iloc[-1:]["BC"] = np.nan
        Solution = Solution.drop(columns=["option_index", "BC"])
        colorder = ["ID", "Section", "LCC", "name", "yes/no", "dcrest", "dberm"]
        Solution = Solution[colorder]
        names = []
        for i in Solution["name"]:
            names.append(i[0])
        Solution["name"] = names
        if type == "Final":
            self.FinalSolution = Solution
            self.FinalSolution.to_csv(csv_path)
        elif type == "Optimal":
            self.OptimalSolution = Solution
            self.OptimalSolution.to_csv(csv_path)
            self.OptimalStep = step - 1
        elif type == "SatisfiedStandard":
            self.SatisfiedStandardSolution = Solution
            self.SatisfiedStandardSolution.to_csv(csv_path)

    def plot_beta_time(self, traject: DikeTraject, typ="single", path=None):
        """This routine plots the reliability in time for each step in the optimization. Mainly for debugging purposes."""
        horizon = np.max(self.T)

        step = 0
        beta_t = []
        plt.figure(100)
        for i in self.Probabilities:
            step += 1
            beta_t0, p = calc_traject_prob(i, horizon=horizon)
            beta_t.append(beta_t0)
            t = range(2025, 2025 + horizon)
            plt.plot(t, beta_t0, label=self.type + " stap " + str(step))
        if typ == "single":
            plt.plot(
                [2025, 2025 + horizon],
                [
                    pf_to_beta(traject.general_info["Pmax"]),
                    pf_to_beta(traject.general_info["Pmax"]),
                ],
                "k--",
                label="Norm",
            )
            plt.xlabel("Time")
            plt.ylabel(r"$\beta$")
            plt.legend()
            plt.savefig(
                path.joinpath("figures", "BetaInTime" + self.type + ".png"),
                bbox_inches="tight",
            )
            plt.close(100)
        else:
            pass

    def plot_beta_costs(
        self,
        traject: DikeTraject,
        save_dir,
        fig_id,
        series_name=None,
        MeasureTable=None,
        t=0,
        cost_type="LCC",
        last=False,
        horizon=100,
        markersize=10,
        symbolsections=False,
        color="r",
        linestyle="-",
        final_step=False,
        final_measure_symbols=False,
        solutiontype=False,
    ):
        """Script to plot the costs versus the reliability in time. Different measures are indicated by different markers. There is a bunch of options that might have to be cleaned up, but this will be done later"""
        # TODO Evaluate options for plotBetaCosts and eliminate obsolete options/put in general config.
        if series_name == None:
            series_name = self.type
        if self.beta_cost_settings["symbols"]:
            symbols = [
                "*",
                "o",
                "^",
                "s",
                "p",
                "X",
                "d",
                "h",
                ">",
                ".",
                "<",
                "v",
                "3",
                "P",
                "D",
            ]
            MeasureTable = MeasureTable.assign(symbol=symbols[0 : len(MeasureTable)])
        else:
            symbols = None

        if solutiontype == "OptimalSolution":
            final_solution_index = list(self.OptimalSolution.index)
            markersize2 = self.beta_cost_settings["markersize"] / 2
        elif solutiontype == "SatisfiedStandard":
            final_solution_index = list(self.SatisfiedStandardSolution.index)
            markersize2 = self.beta_cost_settings["markersize"] / 2
        else:
            final_solution_index = list(self.TakenMeasures.index)
            markersize2 = self.beta_cost_settings["markersize"]

        if "years" not in locals():
            years = traject.sections[
                0
            ].section_reliability.SectionReliability.columns.values.astype("float")
            horizon = np.max(years)
        if not final_step:
            final_step = self.TakenMeasures["Section"].size
        step = 0
        betas = []
        pfs = []

        for i in self.Probabilities[0:final_step]:
            step += 1
            beta_t0, p_t = calc_traject_prob(i, horizon=horizon)
            betas.append(beta_t0[t])
            pfs.append(p_t[t])

        # plot beta vs costs
        x = 0
        Costs = []
        if cost_type == "LCC":
            costname = "LCC"
            for i in range(final_step):
                if not np.isnan(self.TakenMeasures["LCC"].iloc[i]):
                    x += self.TakenMeasures["LCC"].iloc[i]
                else:
                    pass
                Costs.append(x)

        elif cost_type == "Initial":
            costname = "Investment cost"

            for i in range(final_step):
                if i > 0:
                    years = (
                        self.options[self.TakenMeasures.iloc[i]["Section"]]
                        .iloc[self.TakenMeasures.iloc[i]["option_index"]]["year"]
                        .values[0]
                    )
                    if isinstance(years, list):
                        for ij in range(len(years)):
                            if years[ij] == 0:
                                x += (
                                    self.options[self.TakenMeasures.iloc[i]["Section"]]
                                    .iloc[self.TakenMeasures.iloc[i]["option_index"]][
                                        "cost"
                                    ]
                                    .values[0][ij]
                                )

                    else:  # isinstance(years, np.integer):
                        if years > 0:
                            pass
                        else:
                            x += self.TakenMeasures["LCC"].iloc[i]

                    Costs.append(x)
                else:
                    Costs.append(x)

        Costs = np.divide(Costs, 1e6)
        if self.beta_or_prob == "beta":
            rel_unit = r"$\beta$"
            data_to_plot = betas
            interval = 0.07
        elif self.beta_or_prob == "prob":
            data_to_plot = pfs
            rel_unit = r"$P_f$"
            interval = 2
        plt.figure(fig_id)
        if self.beta_cost_settings["symbols"]:
            plt.plot(
                Costs,
                data_to_plot,
                label=series_name,
                color=color,
                linestyle=linestyle,
                zorder=1,
            )
        else:
            plt.plot(
                Costs,
                data_to_plot,
                label=series_name,
                color=color,
                linestyle=linestyle,
                markerstyle="o",
            )

        if self.beta_cost_settings["symbols"]:
            if self.beta_or_prob == "beta":
                base = np.max(data_to_plot) + interval
                ycoord = np.array(
                    [base, base + interval, base + 2 * interval, base + 3 * interval]
                )
            elif self.beta_or_prob == "prob":
                base = np.min(data_to_plot) / interval
                ycoord = np.array(
                    [
                        base,
                        base / interval,
                        base / (2 * interval),
                        base / (3 * interval),
                    ]
                )
            ycoords = np.tile(ycoord, np.int(np.ceil(len(Costs) / len(ycoord))))

            for i in range(len(Costs)):
                line = self.TakenMeasures.iloc[i]
                if line["option_index"] != None:
                    if isinstance(line["ID"], list):
                        line["ID"] = "+".join(line["ID"])
                    if Costs[i] > Costs[i - 1]:
                        if final_measure_symbols and i not in final_solution_index:
                            marker = markersize2
                            edgecolor = "gray"
                        else:
                            marker = self.beta_cost_settings["markersize"]
                            edgecolor = "k"
                        if self.beta_or_prob == "beta":
                            plt.scatter(
                                Costs[i],
                                betas[i],
                                s=marker,
                                marker=MeasureTable.loc[
                                    MeasureTable["ID"] == line["ID"]
                                ]["symbol"].values[0],
                                label=MeasureTable.loc[
                                    MeasureTable["ID"] == line["ID"]
                                ]["Name"].values[0],
                                color=color,
                                edgecolors=edgecolor,
                                linewidths=0.5,
                                zorder=2,
                            )
                            if symbolsections:
                                plt.vlines(
                                    Costs[i],
                                    betas[i] + 0.05,
                                    ycoords[i] - 0.05,
                                    colors="tab:gray",
                                    linestyles=":",
                                    zorder=1,
                                )
                        elif self.beta_or_prob == "prob":
                            plt.scatter(
                                Costs[i],
                                pfs[i],
                                s=marker,
                                marker=MeasureTable.loc[
                                    MeasureTable["ID"] == line["ID"]
                                ]["symbol"].values[0],
                                label=MeasureTable.loc[
                                    MeasureTable["ID"] == line["ID"]
                                ]["Name"].values[0],
                                color=color,
                                edgecolors=edgecolor,
                                linewidths=0.5,
                                zorder=2,
                            )
                            if symbolsections:
                                plt.vlines(
                                    Costs[i],
                                    pfs[i],
                                    ycoords[i],
                                    colors="tab:gray",
                                    linestyles=":",
                                    zorder=1,
                                )
                    if symbolsections:
                        plt.text(
                            Costs[i],
                            ycoords[i],
                            line["Section"][-2:],
                            fontdict={"size": 8},
                            color=color,
                            horizontalalignment="center",
                            zorder=3,
                        )

        if last:
            axes = plt.gca()
            xmax = np.max([axes.get_xlim()[1], np.max(Costs)])
            ceiling = np.ceil(np.max([xmax, np.max(Costs)]) / 10) * 10
            if self.beta_or_prob == "beta":
                plt.plot(
                    [0, ceiling],
                    [
                        pf_to_beta(traject.general_info["Pmax"]),
                        pf_to_beta(traject.general_info["Pmax"]),
                    ],
                    "k--",
                    label="Safety standard",
                )
                plt.ylabel(r"$\beta$")

            if self.beta_or_prob == "prob":
                plt.plot(
                    [0, ceiling],
                    [traject.general_info["Pmax"], traject.general_info["Pmax"]],
                    "k--",
                    label="Safety standard",
                )
                plt.ylabel(r"$P_f$")
                axes.set_yscale("log")
                axes.invert_yaxis()
            plt.xlabel(costname + " in M€")
            plt.xticks(np.arange(0, ceiling + 1, 25))
            axes.set_xlim(left=0, right=ceiling)
            plt.grid()

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc=4, fontsize="x-small")
            leg = plt.gca().get_legend()
            for i in range(3, len(leg.legendHandles)):
                leg.legendHandles[i].set_color(color)
            # plt.legend(loc=5)
            plt.title(costname + " versus " + rel_unit + " in year " + str(t + 2025))

            # plt.title(r'Relation between ' + rel_unit + ' and investment costs in M€')
            plt.title("Priority order of investments")
        data = pd.DataFrame(
            np.array([Costs.T, np.array(betas)]).T, columns=["Cost", "beta"]
        )
        if cost_type == "LCC":
            data.to_csv(
                save_dir.joinpath(
                    "Beta vs "
                    + cost_type
                    + "_"
                    + series_name
                    + "_t"
                    + str(t + 2025)
                    + ".csv"
                )
            )
        if cost_type == "Initial":
            data.to_csv(
                save_dir.joinpath(
                    "Beta vs "
                    + cost_type
                    + "_"
                    + series_name
                    + "_t"
                    + str(t + 2025)
                    + ".csv"
                )
            )

    def plot_investment_limit(
        self,
        traject: DikeTraject,
        investmentlimit=False,
        step2=False,
        path=None,
        figure_size=(6, 4),
        years=[0],
        language="NL",
        flip=False,
        alpha=0.3,
    ):
        """This is a variant of plotAssessment that plots the reliability for all measures up to a certain investment limit"""
        # TODO Integrate with other, similar functions
        # all settings, similar to plotAssessment

        ##Settings part:
        plot_settings()
        # English or Dutch labels and titles
        if language == "NL":
            label_xlabel = "Dijkvakken"
            label_ylabel = r"Betrouwbaarheidsindex $\beta$ [-/jaar]"
            label_target = "Doelbetrouwbaarheid"
            labels_xticks = []
            for i in traject.sections:
                labels_xticks.append(i.name)
        elif language == "EN":
            label_xlabel = "Dike sections"
            label_ylabel = r"Reliability index $\beta$ [-/year]"
            label_target = "Target reliability"
            labels_xticks = []
            for i in traject.sections:
                labels_xticks.append("S" + i[-2:])
        color = ["r", "g", "b", "k"]

        cumlength, xticks1, middles = get_section_length_in_traject(
            traject.probabilities["Length"]
            .loc[traject.probabilities.index.get_level_values(1) == "Overflow"]
            .values
        )

        if not investmentlimit:
            # plot the probabilities for i-1 with alpha 0.3
            for i in range(1, len(self.TakenMeasures)):
                col = 0
                mech = 0
                line1 = {}
                line2 = {}
                mid = {}
                fig, ax = plt.subplots(figsize=figure_size)
                for j in self.mechanisms:
                    plotdata1 = copy.deepcopy(
                        self.Probabilities[i - 1][years[0]].xs(
                            j, level="mechanism", axis=0
                        )
                    ).values
                    plotdata2 = copy.deepcopy(
                        self.Probabilities[i][years[0]].xs(j, level="mechanism", axis=0)
                    ).values
                    ydata1 = copy.deepcopy(plotdata1)  # oude sterkte
                    ydata2 = copy.deepcopy(plotdata2)  # nieuwe sterkte
                    for ij in range(0, len(plotdata1)):
                        ydata1 = np.insert(ydata1, ij * 2, plotdata1[ij])
                        ydata2 = np.insert(ydata2, ij * 2, plotdata2[ij])
                    (line1[mech],) = ax.plot(
                        xticks1, ydata1, color=color[col], linestyle="-", alpha=alpha
                    )
                    (line2[mech],) = ax.plot(
                        xticks1, ydata2, color=color[col], linestyle="-", label=j
                    )
                    (mid[mech],) = ax.plot(
                        middles, plotdata2, color=color[col], linestyle="", marker="o"
                    )
                    col += 1
                    mech += 1
                for ik in cumlength:
                    ax.axvline(x=ik, color="k", linestyle=":", alpha=0.5)
                ax.plot(
                    [0, max(cumlength)],
                    [
                        pf_to_beta(traject.general_info["Pmax"]),
                        pf_to_beta(traject.general_info["Pmax"]),
                    ],
                    "k--",
                    label=label_target,
                    linewidth=1,
                )
                ax.legend(loc=1)
                ax.set_xlabel(label_xlabel)
                ax.set_ylabel(label_ylabel)
                ax.set_xticks(middles)
                ax.set_xticklabels(labels_xticks)
                ax.tick_params(axis="x", rotation=90)
                ax.set_xlim([0, max(cumlength)])
                ax.tick_params(axis="both", bottom=False)
                ax.set_ylim([1.5, 8.5])
                ax.grid(axis="y")
                if flip:
                    ax.invert_xaxis()
                plt.savefig(
                    path.joinpath(
                        str(2025 + years[0])
                        + "_Step="
                        + str(i - 1)
                        + " to "
                        + str(i)
                        + ".png"
                    ),
                    dpi=300,
                    bbox_inches="tight",
                    format="png",
                )
                plt.close()
        else:
            time2 = np.max(
                np.argwhere(np.nancumsum(self.TakenMeasures["LCC"]) < investmentlimit)
            )
            for i in range(0, 2):
                if i == 0:
                    step1 = 0
                    step2 = time2  # ; print(step1); print(step2)
                if i == 1:
                    step1 = time2
                    step2 = -1  # ; print(step1); print(step2)
                col = 0
                mech = 0
                line1 = {}
                line2 = {}
                mid = {}
                fig, ax = plt.subplots(figsize=figure_size)
                for j in self.mechanisms:
                    plotdata1 = copy.deepcopy(
                        self.Probabilities[step1][years[0]].xs(
                            j, level="mechanism", axis=0
                        )
                    ).values
                    plotdata2 = copy.deepcopy(
                        self.Probabilities[step2][years[0]].xs(
                            j, level="mechanism", axis=0
                        )
                    ).values
                    ydata1 = copy.deepcopy(plotdata1)  # oude sterkte
                    ydata2 = copy.deepcopy(plotdata2)  # nieuwe sterkte
                    for ij in range(0, len(plotdata1)):
                        ydata1 = np.insert(ydata1, ij * 2, plotdata1[ij])
                        ydata2 = np.insert(ydata2, ij * 2, plotdata2[ij])
                    (line1[mech],) = ax.plot(
                        xticks1, ydata1, color=color[col], linestyle="-", alpha=alpha
                    )
                    (line2[mech],) = ax.plot(
                        xticks1, ydata2, color=color[col], linestyle="-", label=j
                    )
                    (mid[mech],) = ax.plot(
                        middles, plotdata2, color=color[col], linestyle="", marker="o"
                    )
                    col += 1
                    mech += 1
                for ik in cumlength:
                    ax.axvline(x=ik, color="k", linestyle=":", alpha=0.5)
                ax.plot(
                    [0, max(cumlength)],
                    [
                        pf_to_beta(traject.general_info["Pmax"]),
                        pf_to_beta(traject.general_info["Pmax"]),
                    ],
                    "k--",
                    label=label_target,
                    linewidth=1,
                )
                ax.legend(loc=1)
                ax.set_xlabel(label_xlabel)
                ax.set_ylabel(label_ylabel)
                ax.set_xticks(middles)
                ax.set_xticklabels(labels_xticks)
                ax.tick_params(axis="x", rotation=90)
                ax.set_xlim([0, max(cumlength)])
                ax.tick_params(axis="both", bottom=False)
                ax.set_ylim([1.5, 8.5])
                ax.grid(axis="y")
                if flip:
                    ax.invert_xaxis()
                if i == 0:
                    plt.savefig(
                        path.joinpath(
                            str(2025 + years[0])
                            + "_Begin to "
                            + str(int(investmentlimit))
                            + ".png"
                        ),
                        dpi=300,
                        bbox_inches="tight",
                        format="png",
                    )
                if i == 1:
                    plt.savefig(
                        path.joinpath(
                            str(2025 + years[0])
                            + "_"
                            + str(int(investmentlimit))
                            + " to end.png"
                        ),
                        dpi=300,
                        bbox_inches="tight",
                        format="png",
                    )
                plt.close()

    def write_reliability_to_csv(self, input_path: Path, type):
        """Routine to write all the reliability indices in a step of the algorithm to a csv file"""
        # with open(path + '\\ReliabilityLog_' + type + '.csv', 'w') as f:
        for i in range(len(self.Probabilities)):
            name = input_path.joinpath(
                "ReliabilityLog_" + type + "_Step" + str(i) + ".csv"
            )
            self.Probabilities[i].to_csv(path_or_buf=name, header=True)

    @abstractmethod
    def determine_risk_cost_curve(self, traject: DikeTraject, input_path: Path = None):
        raise NotImplementedError("Expected concrete definition in inherited class.")

    def get_safety_standard_step(self, Ptarget, t=50):
        """Get the index of the measure where the traject probability in year t is higher than the requirement"""
        for i in range(0, len(self.Probabilities)):
            beta_traj, Pf_traj = calc_traject_prob(self.Probabilities[i], ts=t)
            if Pf_traj < Ptarget:
                self.SafetyStandardStep = i
                print("found step {} with {:.2f}".format(i, beta_traj[0]))
                break

            if i == len(self.Probabilities) - 1:
                self.SafetyStandardStep = i
                print("Warning: safety standard not met. Using final step for plotting")

    def plot_measures(
        self,
        traject: DikeTraject,
        input_path: Path,
        fig_size=(12, 4),
        crestscale=25.0,
        creststep=0.5,
        show_xticks=True,
        flip=False,
        title_in=False,
        greedymode="Optimal",
        colors=False,
    ):
        """This routine plots the measures for different solution options. Extension of plotAssessment.
        We might need to generalize the labeling, as this is now only in English."""

        # TODO check labeling
        # set the lengths of the sections
        cumlength, xticks1, middles = get_section_length_in_traject(
            traject.probabilities["Length"]
            .loc[traject.probabilities.index.get_level_values(1) == "Overflow"]
            .values
        )
        if colors:
            color = sns.cubehelix_palette(**colors)
        else:
            color = sns.cubehelix_palette(
                n_colors=5, start=1.9, rot=1, gamma=1.5, hue=1.0, light=0.8, dark=0.3
            )
        markers = ["o", "v", "d", "*"]
        # fig, ax = plt.subplots(figsize=fig_size)
        fig, (ax, ax1) = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=fig_size,
            sharey="row",
            gridspec_kw={
                "width_ratios": [20, 1],
                "wspace": 0.08,
                "left": 0.03,
                "right": 0.98,
            },
        )
        col = 0
        # make berm and crest
        lines = {}
        types = ["dcrest", "dberm"]
        if self.__class__.__name__ == "GreedyStrategy":
            if greedymode == "Optimal":
                Solution = copy.deepcopy(self.OptimalSolution)
            elif greedymode == "SatisfiedStandard":
                self.get_safety_standard_step(traject.general_info["Pmax"])
                self.make_solution(
                    input_path.joinpath("SatisfiedStandardGreedy.csv"),
                    step=self.SafetyStandardStep + 1,
                    type="SatisfiedStandard",
                )
                Solution = self.SatisfiedStandardSolution

        else:
            Solution = copy.deepcopy(self.FinalSolution)
        # Solution['dcrest'].iloc[0]=0.5; print('careful: test line included')
        for i in types:
            data = Solution[i].values
            data[np.where(data.astype(np.float32) < -900)] = 0.01
            if i == "dcrest":
                if np.nanmax(data) > 0.2:
                    data = np.multiply(data, -crestscale)
            ydata = copy.deepcopy(data)
            for ij in range(0, len(data)):
                ydata = np.insert(ydata, ij * 2, data[ij])
            # lines[col], = ax.plot(xticks1, ydata, color=color[col], linestyle='-', alpha=1, label=i[1:])
            lines[col] = ax.fill_between(
                xticks1,
                0,
                np.array(ydata, dtype=np.float),
                color=color[col],
                linestyle="-",
                alpha=1,
                label=i[1:],
            )
            col += 1
        # additional measures
        SS = []
        VSG = []
        DW = []
        Customs = []
        T2045 = []
        T2045_y1 = []
        T2045_y2 = []
        for i in range(0, len(Solution["name"])):
            if "tabiliteitsscherm" in Solution["name"].iloc[i]:
                SS.append(middles[i])
            elif "Zanddicht" in Solution["name"].iloc[i]:
                VSG.append(middles[i])
            elif "Zelfkere" in Solution["name"].iloc[i]:
                DW.append(middles[i])
            elif "Grondversterking" in Solution["name"].iloc[i]:
                pass
            elif np.float32(Solution["LCC"].iloc[i]) > 0.0:
                Customs.append(middles[i])
            if "2045" in Solution["name"].iloc[i]:
                T2045.append(i)
                T2045_y1.append(Solution["dcrest"].iloc[i] * -(crestscale))
                T2045_y2.append(Solution["dberm"].iloc[i])

                # WARNING: only for soil!
        # (140: stability screen; 160: VSG; 180: DW) Possibly enter these at 0
        measures = {}
        # add thick zero line
        ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=1)

        if len(SS) > 0:
            measures["SS"] = ax.plot(
                SS,
                np.ones((len(SS), 1)) * 0,
                color=color[col],
                linestyle="",
                marker=markers[0],
                label="SS",
            )
        if len(VSG) > 0:
            measures["VSG"] = ax.plot(
                VSG,
                np.ones((len(VSG), 1)) * 0,
                color=color[col + 1],
                linestyle="",
                marker=markers[1],
                label="VZG",
            )
        if len(DW) > 0:
            measures["DW"] = ax.plot(
                DW,
                np.ones((len(DW), 1)) * 0,
                color=color[col + 2],
                linestyle="",
                marker=markers[2],
                label="DW",
            )
        print(col)
        if len(Customs) > 0:
            measures["Customs"] = ax.plot(
                Customs,
                np.ones((len(Customs), 1)) * 0,
                color=color[col + 3],
                linestyle="",
                marker=markers[3],
                label="Custom",
            )
        if len(T2045) > 0:
            # dummy for label
            ax.plot([-99, -98], [0, 0], color="black", linestyle=":", label="2045")
            for i in range(0, len(T2045)):
                ax.plot(
                    [cumlength[T2045[i]], cumlength[T2045[i] + 1]],
                    [T2045_y1[i], T2045_y1[i]],
                    color="black",
                    linestyle=":",
                )
                ax.plot(
                    [cumlength[T2045[i]], cumlength[T2045[i] + 1]],
                    [T2045_y2[i], T2045_y2[i]],
                    color="black",
                    linestyle=":",
                )
        for i in cumlength:
            ax.axvline(x=i, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.set_xlim(left=0, right=np.max(cumlength))
        bermticks = np.arange(0, 51, 10)
        crestticks = np.arange(-crestscale * 2, 0, creststep * crestscale)
        # otherticks = np.arange(140,181,20)
        allticks = np.concatenate((crestticks, bermticks))  # , otherticks))
        bermticklabels = bermticks.astype(np.int64).astype(str)
        crestticklabels = np.abs(np.divide(crestticks, -crestscale)).astype(str)
        # otherlabels = np.array(['SS', 'VSG', 'DW'])
        allticklabels = np.concatenate(
            (crestticklabels, bermticklabels)
        )  # ,otherlabels))
        ax.set_yticks(allticks)
        ax.set_yticklabels(allticklabels, fontsize="x-small")
        ax.set_ylim(top=np.max(allticks), bottom=np.min(allticks))
        if show_xticks:
            labels_xticks = []
            for i in traject.sections:
                labels_xticks.append(i.name)
            ax.set_xticks(middles)
            ax.set_xticklabels(labels_xticks)
            ax.tick_params(axis="x", rotation=90)
        else:
            ax.set_xticklabels("")
        ax.tick_params(axis="both", bottom=False)

        ax.grid(axis="y", linewidth=0.5, color="gray", alpha=0.5)
        ax.invert_yaxis()
        if flip:
            ax.invert_xaxis()
        ax.text(-0.035, 0.7, "Kruin in m", rotation=90, transform=ax.transAxes)
        ax.text(-0.035, 0.1, "Berm in m", rotation=90, transform=ax.transAxes)
        # ax.text(-0.035, -0.02, 'Structural', rotation=90, transform=ax.transAxes)
        ax.legend(bbox_to_anchor=(1.01, 0.85))
        if title_in:
            ax.set_title(title_in)
        ax1.axis("off")
        plt.savefig(
            input_path.joinpath(self.__class__.__name__ + "_measures.png"),
            dpi=300,
            bbox_inches="tight",
            format="png",
        )
