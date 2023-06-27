import copy
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from vrtool.decision_making.solutions import Solutions
from vrtool.decision_making.strategies.mixed_integer_strategy import (
    MixedIntegerStrategy,
)
from vrtool.decision_making.strategies.strategy_base import StrategyBase
from vrtool.decision_making.strategy_evaluation import calc_tc, split_options
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf


class ParetoFrontierStrategy(StrategyBase):
    """This is a subclass for generating a ParetoFrontier based on Mixed Integer evaluations with a budget limit."""

    def evaluate(
        self,
        traject: DikeTraject,
        solutions_dict: Dict[str, Solutions],
        lcc_list: bool = False,
        strategy_path=False,
    ):
        _mip_object = MixedIntegerStrategy("MIPObject")
        _mip_object.combine(traject, solutions_dict, splitparams=True)
        _mip_object.make_optimization_input(traject)

        MIPObjects = []
        MIPModels = []
        MIPResults = []
        LCC = []
        TR = []
        TC = []
        ObjectiveValue = []
        for j in range(0, len(lcc_list)):
            MIPObjects.append(copy.deepcopy(_mip_object))
            MIPModels.append(
                MIPObjects[-1].create_optimization_model(BudgetLimit=lcc_list[j])
            )
            MIPModels[-1].solve()
            MIPResult = {}
            MIPResult["Values"] = MIPModels[-1].solution.get_values()
            MIPResult["Names"] = MIPModels[-1].variables.get_names()
            MIPResult["ObjectiveValue"] = MIPModels[-1].solution.get_objective_value()
            MIPResult["Status"] = MIPModels[-1].solution.get_status_string()
            MIPResults.append(MIPResult)
            MIPObjects[-1].readResults(
                MIPResults[-1],
                MeasureTable=self.get_measure_table(solutions_dict, "NL", False),
            )
            MIPObjects[-1].TakenMeasures.to_csv(
                strategy_path.joinpath(
                    "Pareto_LCC=" + str(np.int32(lcc_list[j])) + ".csv"
                )
            )
            LCC.append(MIPObjects[-1].results["LCC"])
            TR.append(
                MIPObjects[-1].results["GeoRisk"]
                + MIPObjects[-1].results["OverflowRisk"]
            )
            TC.append(MIPObjects[-1].results["TC"])
            ObjectiveValue.append(MIPResult["ObjectiveValue"])
            logging.info(
                MIPModels[-1].solution.status[MIPModels[-1].solution.get_status()]
            )

        self.costs = pd.DataFrame(
            np.array([LCC, TR, TC]).T, columns=["LCC", "TR", "TC"]
        )

    def filter(self, traject: DikeTraject, type: str = "ParetoPerSection"):
        """This is an optional routine that can be used to filter measures per section.
        It is based on defining a Pareto front, its main idea is that you throw out measures that have a certain reliability but are more costly than other measures that provide the same reliability."""
        self.options_height, self.options_geotechnical = split_options(
            self.options, traject.mechanism_names
        )
        if type == "ParetoPerSection":
            damage = traject.general_info.FloodDamage
            r = self.r
            horizon = np.max(self.T)
            self.options_g_filtered = copy.deepcopy(self.options_geotechnical)

            # we filter the options for each section, such that only interesting ones remain

            # filter only geotechnical
            for i in self.options_g_filtered.keys():

                # indexes part 1: only the pareto front for stability and piping
                _calculated_lcc = calc_tc(
                    self.options_g_filtered[i], self.discount_rate
                )

                tgrid = self.options_g_filtered[i]["StabilityInner"].columns.values
                pf_SI = beta_to_pf(self.options_g_filtered[i]["StabilityInner"])
                pf_pip = beta_to_pf(self.options_g_filtered[i]["Piping"])

                pftot1 = interp1d(tgrid, np.add(pf_SI, pf_pip))
                risk1 = np.sum(
                    pftot1(np.arange(0, horizon, 1))
                    * (damage / (1 + r) ** np.arange(0, horizon, 1)),
                    axis=1,
                )
                paretolcc, paretorisk, index1 = self.pareto_frontier(
                    _calculated_lcc, risk1, None, False, False
                )
                index = index1

                self.options_g_filtered[i] = self.options_g_filtered[i].iloc[index]
                self.options_g_filtered[i]["LCC"] = _calculated_lcc[index]
                self.options_g_filtered[i] = self.options_g_filtered[i].reset_index(
                    drop=True
                )
                logging.info(
                    "For dike section "
                    + i
                    + " reduced size from "
                    + str(len(_calculated_lcc))
                    + " to "
                    + str(len(index))
                )

            # swap filtered and original measures:
            self.options_old_geotechnical = copy.deepcopy(self.options_geotechnical)
            self.options_geotechnical_filtered = copy.deepcopy(self.options_g_filtered)
            del self.options_g_filtered

        @staticmethod
        def pareto_frontier(
            x_array: np.ndarray,
            y_array: np.ndarray,
            input_path: Path,
            max_x: bool,
            max_y: bool,
        ) -> tuple[list, list, list]:
            if input_path:
                x_array = np.array([])  # LCC
                y_array = np.array([])  # TR
                # read info from path
                for _input_file in input_path.iterdir():
                    if _input_file.is_file() and "ParetoResults" in _input_file.stem:
                        data = pd.read_csv(_input_file)
                        x_array = np.concatenate((x_array, data["LCC"].values))
                        y_array = np.concatenate((y_array, data["TR"].values))
            elif not x_array.size or not y_array.size:
                raise ValueError("No input provided")

            myList = sorted(
                [[x_array[i], y_array[i]] for i in range(len(x_array))], reverse=max_x
            )
            index_order = np.argsort(x_array)
            p_front = [myList[0]]
            index = [index_order[0]]
            count = 1
            for pair in myList[1:]:
                if max_y:
                    if pair[1] >= p_front[-1][1]:
                        p_front.append(pair)
                        index.append(index_order[count])
                else:
                    if pair[1] < p_front[-1][1]:
                        p_front.append(pair)
                        index.append(index_order[count])
                    elif pair[1] == p_front[-1][1]:
                        if pair[0] < p_front[-1][0]:
                            p_front.append(pair)
                            index.append(index_order[count])
                count += 1
            p_frontX = [pair[0] for pair in p_front]
            p_frontY = [pair[1] for pair in p_front]
            return p_frontX, p_frontY, index
