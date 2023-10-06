import copy
import logging
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d

from vrtool.decision_making.solutions import Solutions
from vrtool.decision_making.strategy_evaluation import (
    calc_tc,
    measure_combinations,
    revetment_combinations,
    split_options,
)
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.dike_traject import DikeTraject, calc_traject_prob
from vrtool.probabilistic_tools.combin_functions import CombinFunctions
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf, pf_to_beta


class StrategyBase:
    """This defines a Strategy object, which can be allowed to evaluate a set of solutions/measures. There are currently 3 types:
    Greedy: a greedy optimization method is used here
    TargetReliability: a cross-sectional optimization in line with OI2014
    MixedInteger: a Mixed Integer optimization. Note that this has exponential runtime for large systems so it should not be used for more than approximately 13 sections.
    Note that this is the main class. Each type has a different subclass"""

    def __init__(self, type, config: VrtoolConfig):
        self.type = type
        self.discount_rate = config.discount_rate

        self.config = config
        self.OI_year = config.OI_year
        self.OI_horizon = config.OI_horizon
        self.mechanisms = config.mechanisms
        self.T = config.T
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
            logging.warning(
                "Deriving section order from unordered dictionary. Might be wrong"
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
                logging.info("SECTION {}".format(section))
                logging.info(
                    "No measures are taken at this section: sh and sg are: {} and {}".format(
                        sh, sg
                    )
                )
                return (section, sh, sg)
        if index[2] > 0:
            sg = self.options_geotechnical[section_order[index[0]]].iloc[index[2] - 1]

        if print_measure:
            logging.info("SECTION {}".format(section))
            if isinstance(sh, str):
                logging.info("There is no measure for height")
            else:
                logging.info(
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
                logging.info(
                    " The geotechnical measure is a {}".format(sg.type.values[0])
                )
            elif isinstance(sg.type.values[0], list):  # VZG+Soil
                logging.info(
                    " The geotechnical measure is a {} in year {} with a {} with dberm = {} in year {}".format(
                        sg.type.values[0][0],
                        sg.year.values[0][0],
                        sg.type.values[0][1],
                        sg.dberm.values[0],
                        sg.year.values[0][1],
                    )
                )
            elif sg.type.values == "Soil reinforcement":
                logging.info(
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
        self.indexCombined2single = {}

        # measures at t=0 (2025) and t=20 (2045)
        # for i in range(0, len(traject.sections)):
        for i, section in enumerate(traject.sections):
            combinedmeasures = self._step1combine(solutions_dict, section, splitparams)

            StrategyData = copy.deepcopy(solutions_dict[section.name].MeasureData)
            if self.__class__.__name__ == "TargetReliabilityStrategy":
                StrategyData = StrategyData.loc[StrategyData["year"] == self.OI_year]

            StrategyData = pd.concat((StrategyData, combinedmeasures))
            if filtering == "on":
                StrategyData = copy.deepcopy(StrategyData)
                StrategyData = StrategyData.reset_index(drop=True)
                LCC = calc_tc(StrategyData, self.discount_rate)
                ind = np.argsort(LCC)
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

    def _step1combine(
        self,
        solutions_dict: dict[str, Solutions],
        section: DikeSection,
        splitparams: bool,
    ) -> pd.DataFrame:
        """
        Combines the measures based on the input arguments.

        Args:
            solutions_dict (dict[str, Solutions]): The dictionary containing the solutions for each section.
            section (DikeSection): The dike section to combine the measures for.
            splitparams (bool): Indicator whether the parameters should be split.

        Returns:
            pd.DataFrame: An object that contains all information of the combined measures.
        """
        # split different measure types:
        available_measure_classes = (
            solutions_dict[section.name].MeasureData["class"].unique().tolist()
        )
        measures_per_class = {
            measure_class: solutions_dict[section.name].MeasureData.loc[
                solutions_dict[section.name].MeasureData["class"] == measure_class
            ]
            for measure_class in available_measure_classes
        }

        if self.__class__.__name__ == "TargetReliabilityStrategy":
            # only consider measures at the OI_year
            measures_per_class = {
                measure_class: measures_per_class[measure_class].loc[
                    measures_per_class[measure_class]["year"] == self.OI_year
                ]
                for measure_class in available_measure_classes
            }

        self.indexCombined2single[section.name] = [
            [i] for i in range(len(solutions_dict[section.name].MeasureData))
        ]

        combinedmeasures = measure_combinations(
            measures_per_class["combinable"],
            measures_per_class["partial"],
            solutions_dict[section.name],
            self.indexCombined2single[section.name],
            splitparams=splitparams,
        )

        if "revetment" in measures_per_class:
            combinedmeasures_with_revetment = revetment_combinations(
                combinedmeasures,
                measures_per_class["revetment"],
                self.indexCombined2single[section.name],
            )
            # combine solutions_dict[section.name].MeasureData with revetments
            base_measures_with_revetment = revetment_combinations(
                solutions_dict[section.name].MeasureData.loc[
                    solutions_dict[section.name].MeasureData["class"] != "revetment"
                ],
                measures_per_class["revetment"],
                self.indexCombined2single[section.name],
            )
            combinedmeasures = pd.concat(
                [base_measures_with_revetment, combinedmeasures_with_revetment]
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
                    # concatenate names with + sign based on solutions_dict using list comprehension
                    name = "+".join(
                        solutions_dict[section.name].measure_table[
                            (
                                solutions_dict[section.name]
                                .measure_table["ID"]
                                .isin(indexes)
                            )
                        ]["Name"]
                    )
                    solutions_dict[section.name].measure_table = pd.concat(
                        [
                            solutions_dict[section.name].measure_table,
                            pd.DataFrame([[ij, name]], columns=["ID", "Name"]),
                        ]
                    )
        return combinedmeasures

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

        def get_independent_probability_of_failure(
            probability_of_failure_lookup: dict[str, np.array]
        ) -> np.array:
            return CombinFunctions.combine_probabilities(
                probability_of_failure_lookup, ("StabilityInner", "Piping")
            )

        # TODO Currently incorrectly combined measures with sh = 0.5 crest and sg 0.5 crest + geotextile have not cost 1e99. However they
        #  do have costs higher than the correct option (sh=0m, sg=0.5+VZG) so they will never be selected. This
        #  should be fixed though

        self.options_height, self.options_geotechnical = split_options(
            self.options, traject.mechanism_names
        )

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
        for _mechanism_name in self.mechanisms:
            if _mechanism_name == "Overflow":
                self.Pf[_mechanism_name] = np.full((N, Sh + 1, T), 1.0)
            elif _mechanism_name == "Revetment":
                self.Pf[_mechanism_name] = np.full((N, Sh + 1, T), 1.0e-18)
            else:
                self.Pf[_mechanism_name] = np.full((N, Sg + 1, T), 1.0)

        # fill values
        # TODO Think about the initial condition and whether this should be added separately or teh 0,
        #  0 soil reinforcement also suffices.
        section_keys = list(self.options.keys())

        # get all probabilities. Interpolate on beta per section, then combine p_f
        betas = {}
        for n in range(0, N):
            for _mechanism_name in traject.sections[n].mechanism_data:
                len_beta1 = traject.sections[
                    n
                ].section_reliability.SectionReliability.shape[1]

                beta1 = (
                    traject.sections[n]
                    .section_reliability.SectionReliability.loc[_mechanism_name]
                    .values.reshape((len_beta1, 1))
                    .T
                )
                # Initial
                # condition with no measure
                if _mechanism_name == "Overflow" or _mechanism_name == "Revetment":
                    beta2 = self.options_height[section_keys[n]][_mechanism_name]
                    # All solutions
                else:
                    beta2 = self.options_geotechnical[section_keys[n]][
                        _mechanism_name
                    ]  # All solutions
                betas[_mechanism_name] = np.concatenate((beta1, beta2), axis=0)
                if np.shape(betas[_mechanism_name])[1] != T:
                    betas[_mechanism_name] = interp1d(self.T, betas[_mechanism_name])(
                        np.arange(0, T, 1)
                    )
                self.Pf[_mechanism_name][
                    n, 0 : np.size(betas[_mechanism_name], 0), :
                ] = beta_to_pf(betas[_mechanism_name])

        # Costs of options [N,Sh,Sg]
        self.LCCOption = np.full((N, Sh + 1, Sg + 1), 1e99)
        for n in range(0, len(section_keys)):
            self.LCCOption[n, 0, 0] = 0.0
            LCC_sh = calc_tc(self.options_height[section_keys[n]], self.discount_rate)
            LCC_sg = calc_tc(
                self.options_geotechnical[section_keys[n]], self.discount_rate
            )
            # LCC_tot = calcTC(self.options[keys[n]])
            # we get the unique ids of the options in the height and geotechnical measures
            section_sg_ids = self.options_geotechnical[section_keys[n]].ID.unique()
            section_sh_ids = self.options_height[section_keys[n]].ID.unique()
            for sh_id in section_sh_ids:
                sh_indices = (
                    self.options_height[section_keys[n]]
                    .index[self.options_height[section_keys[n]].ID == sh_id]
                    .tolist()
                )

                # we get the indices of sg_id in the options_geotechnical df
                sg_indices = (
                    self.options_geotechnical[section_keys[n]]
                    .index[self.options_geotechnical[section_keys[n]].ID == sh_id]
                    .tolist()
                )
                # combined LCC array for sh_indices and sg_indices
                LCC_combined = np.add(
                    np.tile(LCC_sh[sh_indices], (len(sg_indices), 1)),
                    np.tile(LCC_sg[sg_indices], (len(sh_indices), 1)).T,
                )
                # fill self.LCCOption[n, sh_indices, sg_indices]
                # we do it using a loop, as masking is not working properly
                # Loop through the indices and update values
                for i, sh_idx in enumerate(sh_indices):
                    for j, sg_idx in enumerate(sg_indices):
                        self.LCCOption[n, sh_idx + 1, sg_idx + 1] = LCC_combined[j, i]

        # Decision Variables for executed options [N,Sh] & [N,Sg]
        self.Cint_h = np.zeros((N, Sh))
        self.Cint_g = np.zeros((N, Sg))

        # Decision Variable for weakest overflow section with dims [N,Sh]
        self.Dint = np.zeros((N, Sh))

        # add discounted damage [T,]
        self.D = np.array(
            traject.general_info.FloodDamage
            * (1 / ((1 + self.discount_rate) ** np.arange(0, T, 1)))
        )

        # expected damage for overflow and for piping & slope stability
        # self.RiskGeotechnical = np.zeros((N,Sg+1,T))
        self.RiskGeotechnical = get_independent_probability_of_failure(
            self.Pf
        ) * np.tile(self.D.T, (N, Sg + 1, 1))

        self.RiskOverflow = self.Pf["Overflow"] * np.tile(self.D.T, (N, Sh + 1, 1))

        self.RiskRevetment = []
        if "Revetment" in self.mechanisms:
            self.RiskRevetment = self.Pf["Revetment"] * np.tile(
                self.D.T, (N, Sh + 1, 1)
            )
        else:
            self.RiskRevetment = np.zeros((N, Sh + 1, T))

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
        Solution = Solution.drop(columns=["option_index", "BC"])

        for section in sections:
            lines = AllMeasures.loc[AllMeasures["Section"] == section].drop(
                columns=["option_index", "BC"]
            )
            if len(lines) > 1:
                lcctot = np.sum(lines["LCC"])
                lines.loc[lines.index.values[-1], "LCC"] = lcctot
                Solution = pd.concat([Solution, lines[-1:]])
            elif len(lines) == 0:
                lines = pd.DataFrame(
                    np.array(
                        [
                            section,
                            0,
                            0,
                            "No measure",
                            "no",
                            0.0,
                            0.0,
                            -999.0,
                            -999.0,
                        ]
                    ).reshape(1, len(Solution.columns)),
                    columns=Solution.columns,
                )
                Solution = pd.concat([Solution, lines])
            else:
                Solution = pd.concat([Solution, lines])
        colorder = [
            "ID",
            "Section",
            "LCC",
            "name",
            "yes/no",
            "dcrest",
            "dberm",
            "transition_level",
            "beta_target",
        ]
        Solution = Solution[colorder]
        for count, row in Solution.iterrows():
            if isinstance(row["name"], np.ndarray) and any(row["name"]):  # clean output
                Solution.loc[count, "name"] = row["name"][0]

        # TODO: writing to csv is obsolete; use results in the database
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

    def write_reliability_to_csv(
        self, input_path: Path, type: str, time_stamps=[0, 25, 50]
    ) -> None:
        """Routine to write all the reliability indices in a step of the algorithm to a csv file

        Args:
            input_path (Path)        : path to input folder
            type (str)               : strategy type
            time_stamps (list float) : list of years
        """
        # with open(path + '\\ReliabilityLog_' + type + '.csv', 'w') as f:
        total_reliability = np.zeros((len(self.Probabilities), len(time_stamps)))
        for i in range(len(self.Probabilities)):
            name = input_path.joinpath(
                "ReliabilityLog_" + type + "_Step" + str(i) + ".csv"
            )
            self.Probabilities[i].to_csv(path_or_buf=name, header=True)
            beta_t, p_t = calc_traject_prob(self.Probabilities[i], ts=time_stamps)
            total_reliability[i, :] = beta_t
        reliability_df = pd.DataFrame(total_reliability, columns=time_stamps)
        reliability_df.to_csv(
            path_or_buf=input_path.joinpath("TrajectReliabilityInTime.csv"), header=True
        )

    @abstractmethod
    def determine_risk_cost_curve(self, flood_damage: float, output_path: Path):
        raise NotImplementedError("Expected concrete definition in inherited class.")
