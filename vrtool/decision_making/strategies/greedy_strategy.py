import copy
import logging
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from vrtool.decision_making.solutions import Solutions
from vrtool.decision_making.strategies.strategy_base import StrategyBase
from vrtool.decision_making.strategy_evaluation import (
    calc_life_cycle_risks,
    evaluate_risk,
    overflow_bundling,
    old_overflow_bundling,
    update_probability,
)
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.probabilistic_tools.probabilistic_functions import pf_to_beta


class GreedyStrategy(StrategyBase):
    def evaluate(
        self,
        traject: DikeTraject,
        solutions_dict: Dict[str, Solutions],
        splitparams=False,
        setting="fast",
        BCstop=0.1,
        max_count=150,
        f_cautious=2,
    ):
        """This is the main routine for a greedy evaluation of all solutions."""
        # TODO put settings in config
        self.make_optimization_input(traject)
        start = time.time()
        # set start values:
        self.Cint_g[:, 0] = 1
        self.Cint_h[:, 0] = 1

        init_probability = {}
        init_overflow_risk = np.empty(
            (self.opt_parameters["N"], self.opt_parameters["T"])
        )
        init_geotechnical_risk = np.empty(
            (self.opt_parameters["N"], self.opt_parameters["T"])
        )
        for m in self.mechanisms:
            init_probability[m] = np.empty(
                (self.opt_parameters["N"], self.opt_parameters["T"])
            )
            for n in range(0, self.opt_parameters["N"]):
                init_probability[m][n, :] = self.Pf[m][n, 0, :]
                if m == "Overflow":
                    init_overflow_risk[n, :] = self.RiskOverflow[n, 0, :]
                else:
                    init_geotechnical_risk[n, :] = self.RiskGeotechnical[n, 0, :]

        count = 0
        measure_list = []
        Probabilities = []
        Probabilities.append(copy.deepcopy(init_probability))
        risk_per_step = []
        cost_per_step = []
        cost_per_step.append(0)
        # TODo add existing investments
        SpentMoney = np.zeros([self.opt_parameters["N"]])
        InitialCostMatrix = copy.deepcopy(self.LCCOption)
        BC_list = []
        Measures_per_section = np.zeros((self.opt_parameters["N"], 2), dtype=np.int32)
        while count < max_count:
            init_risk = np.sum(np.max(init_overflow_risk, axis=0)) + np.sum(
                init_geotechnical_risk
            )
            risk_per_step.append(init_risk)
            cost_per_step.append(np.sum(SpentMoney))
            # first we compute the BC-ratio for each combination of Sh, Sg, for each section
            LifeCycleCost = np.full(
                [
                    self.opt_parameters["N"],
                    self.opt_parameters["Sh"],
                    self.opt_parameters["Sg"],
                ],
                1e99,
            )
            TotalRisk = np.full(
                [
                    self.opt_parameters["N"],
                    self.opt_parameters["Sh"],
                    self.opt_parameters["Sg"],
                ],
                init_risk,
            )
            for n in range(0, self.opt_parameters["N"]):
                # for each section, start from index 1 to prevent putting inf in top left cell
                for sg in range(1, self.opt_parameters["Sg"]):
                    for sh in range(0, self.opt_parameters["Sh"]):
                        if self.LCCOption[n, sh, sg] < 1e20:
                            LifeCycleCost[n, sh, sg] = copy.deepcopy(
                                np.subtract(self.LCCOption[n, sh, sg], SpentMoney[n])
                            )
                            new_overflow_risk, new_geotechnical_risk = evaluate_risk(
                                copy.deepcopy(init_overflow_risk),
                                copy.deepcopy(init_geotechnical_risk),
                                self,
                                n,
                                sh,
                                sg,
                                self.config,
                            )
                            TotalRisk[n, sh, sg] = copy.deepcopy(
                                np.sum(np.max(new_overflow_risk, axis=0))
                                + np.sum(new_geotechnical_risk)
                            )
                        else:
                            pass
            # do not go back:
            LifeCycleCost = np.where(LifeCycleCost <= 0, 1e99, LifeCycleCost)
            dR = np.subtract(init_risk, TotalRisk)
            BC = np.divide(dR, LifeCycleCost)  # risk reduction/cost [n,sh,sg]
            TC = np.add(LifeCycleCost, TotalRisk)
            # determine the BC of the most favourable option for height
            overflow_bundle_index, BC_bundle = overflow_bundling(
                copy.deepcopy(self), copy.deepcopy(init_overflow_risk), copy.deepcopy(measure_list), copy.deepcopy(LifeCycleCost), copy.deepcopy(traject)
            )

            # compute additional measures where we combine overflow measures, here we optimize a package, purely based
            # on overflow, and compute a general BC ratio that is a factor (factor cautious) higher than the max BC.
            # then in the selection of the measure we make a if-elif split with either the normal routine or an
            # 'overflow bundle'
            if np.isnan(np.max(BC)):
                ids = np.argwhere(np.isnan(BC))
                for i in range(0, ids.shape[0]):
                    error_measure = self.get_measure_from_index(ids[i, :])
                    logging.error(error_measure)
                    # TODO think about a more sophisticated error catch here, as currently tracking the error is extremely difficult.
                raise ValueError("nan value encountered in BC-ratio")
            if (np.max(BC) > BCstop) or (BC_bundle > BCstop):
                if np.max(BC) >= BC_bundle:
                    # find the best combination
                    Index_Best = np.unravel_index(np.argmax(BC), BC.shape)

                    if setting == "robust":
                        measure_list.append(Index_Best)
                        # update init_probability
                        init_probability = update_probability(
                            init_probability, self, Index_Best
                        )

                    elif (setting == "fast") or (setting == "cautious"):
                        BC_sections = np.empty((self.opt_parameters["N"]))
                        # find best measure for each section
                        for n in range(0, self.opt_parameters["N"]):
                            BC_sections[n] = np.max(BC[n, :, :])
                        if len(BC_sections) > 2:
                            BC_second = -np.partition(-BC_sections, 2)[1]
                        else:
                            BC_second = np.min(BC_sections)

                        if setting == "fast":
                            indices = np.argwhere(
                                BC[Index_Best[0]] - np.max([BC_second, 1]) > 0
                            )
                        elif setting == "cautious":
                            indices = np.argwhere(
                                np.divide(BC[Index_Best[0]], np.max([BC_second, 1]))
                                > f_cautious
                            )
                        # a bit more cautious
                        if indices.shape[0] > 1:
                            # take the investment that has the lowest total cost:

                            fast_measure = indices[
                                np.argmin(
                                    TC[Index_Best[0]][(indices[:, 0], indices[:, 1])]
                                )
                            ]
                            Index_Best = (
                                Index_Best[0],
                                fast_measure[0],
                                fast_measure[1],
                            )
                            measure_list.append(Index_Best)
                        else:
                            measure_list.append(Index_Best)
                    BC_list.append(BC[Index_Best])
                    init_probability = update_probability(
                        init_probability, self, Index_Best
                    )
                    init_geotechnical_risk[Index_Best[0], :] = copy.deepcopy(
                        self.RiskGeotechnical[Index_Best[0], Index_Best[2], :]
                    )

                    init_overflow_risk[Index_Best[0], :] = copy.deepcopy(
                        self.RiskOverflow[Index_Best[0], Index_Best[1], :]
                    )

                    # TODO update risks
                    SpentMoney[Index_Best[0]] += copy.deepcopy(
                        LifeCycleCost[Index_Best]
                    )
                    self.LCCOption[Index_Best] = 1e99
                    Measures_per_section[Index_Best[0], 0] = Index_Best[1]
                    Measures_per_section[Index_Best[0], 1] = Index_Best[2]
                    Probabilities.append(copy.deepcopy(init_probability))
                    logging.info("Single measure in step " + str(count))
                elif BC_bundle > np.max(BC):
                    for j in range(0, self.opt_parameters["N"]):
                        if overflow_bundle_index[j, 0] != Measures_per_section[j, 0]:
                            IndexMeasure = (
                                j,
                                overflow_bundle_index[j, 0],
                                overflow_bundle_index[j, 1],
                            )

                            measure_list.append(IndexMeasure)
                            BC_list.append(BC_bundle)
                            init_probability = update_probability(
                                init_probability, self, IndexMeasure
                            )
                            init_overflow_risk[IndexMeasure[0], :] = copy.deepcopy(
                                self.RiskOverflow[IndexMeasure[0], IndexMeasure[1], :]
                            )
                            SpentMoney[IndexMeasure[0]] += copy.deepcopy(
                                LifeCycleCost[IndexMeasure]
                            )
                            self.LCCOption[IndexMeasure] = 1e99
                            Measures_per_section[IndexMeasure[0], 0] = IndexMeasure[1]
                            # no update of geotechnical risk needed
                            Probabilities.append(copy.deepcopy(init_probability))
                    # add the height measures in separate entries in the measure list

                    # write them to the measure_list
                    logging.info("Bundled measures in step " + str(count))

            else:  # stop the search
                break
            count += 1
            if count == max_count:
                pass
                # Probabilities.append(copy.deepcopy(init_probability))
        # pd.DataFrame([risk_per_step,cost_per_step]).to_csv('GreedyResults_per_step.csv') #useful for debugging
        logging.info("Elapsed time for greedy algorithm: " + str(time.time() - start))
        self.LCCOption = copy.deepcopy(InitialCostMatrix)
        # #make dump
        # import shelve
        #
        # filename = config.directory.joinpath('FinalGreedyResult.out')
        # # make shelf
        # my_shelf = shelve.open(str(filename), 'n')
        # my_shelf['Strategy'] = locals()['self']
        # my_shelf['solutions'] = locals()['solutions']
        # my_shelf['measure_list'] = locals()['measure_list']
        # my_shelf['BC_list'] = locals()['BC_list']
        # my_shelf['Probabilities'] = locals()['Probabilities']
        #
        # my_shelf.close()



        self.write_greedy_results(
            traject, solutions_dict, measure_list, BC_list, Probabilities
        )

    def write_greedy_results(
        self,
        traject: DikeTraject,
        solutions_dict: Dict[str, Solutions],
        measure_list,
        BC,
        Probabilities,
    ):
        """This writes the results of a step to a list of dataframes for all steps."""
        # TODO We need to think about how to include outward reinforcement here. Can we formulate outward reinforcement as a 'dberm'?
        TakenMeasuresHeaders = [
            "Section",
            "option_index",
            "LCC",
            "BC",
            "ID",
            "name",
            "yes/no",
            "dcrest",
            "dberm",
        ]
        sections = []
        LCC = []
        LCC2 = []
        LCC_invested = np.zeros((len(traject.sections)))
        ID = []
        dcrest = []
        dberm = []
        yes_no = []
        option_index = []
        names = []
        # write the first line:
        sections.append("")
        LCC.append(0)
        ID.append("")
        dcrest.append("")
        dberm.append("")
        yes_no.append("")
        option_index.append("")
        names.append("")
        BC.insert(0, 0)
        self.MeasureIndices = pd.DataFrame(measure_list)
        for i in measure_list:
            sections.append(traject.sections[i[0]].name)
            LCC.append(
                np.subtract(self.LCCOption[i], LCC_invested[i[0]])
            )  # add costs and subtract the money already
            LCC2.append(self.LCCOption[i])  # add costs
            # spent
            LCC_invested[i[0]] += np.subtract(self.LCCOption[i], LCC_invested[i[0]])

            # get the ids
            ID1 = (
                self.options_geotechnical[traject.sections[i[0]].name]
                .iloc[i[2] - 1]["ID"]
                .values[0]
            )
            if "+" in ID1:
                ID_relevant = ID1[-1]
            else:
                ID_relevant = ID1
            if i[1] != 0:
                ID2 = (
                    self.options_height[traject.sections[i[0]].name]
                    .iloc[i[1] - 1]["ID"]
                    .values[0]
                )
                if ID_relevant == ID2:
                    if (
                        self.options_height[traject.sections[i[0]].name]
                        .iloc[i[1] - 1]["dcrest"]
                        .values[0]
                        == 0.0
                    ) and (
                        self.options_geotechnical[traject.sections[i[0]].name]
                        .iloc[i[2] - 1]["dberm"]
                        .values[0]
                        == 0.0
                    ):
                        ID.append(ID1[0])  # TODO Fixen
                    else:
                        ID.append(ID1)
                else:
                    logging.info(i)
                    logging.info(
                        self.options_geotechnical[traject.sections[i[0]].name].iloc[
                            i[2] - 1
                        ]
                    )
                    logging.info(
                        self.options_height[traject.sections[i[0]].name].iloc[i[1] - 1]
                    )
                    raise ValueError("warning, conflicting IDs found for measures")
            else:
                ID2 = ""
                ID.append(ID1)

            # get the parameters
            dcrest.append(
                self.options_height[traject.sections[i[0]].name]
                .iloc[i[1] - 1]["dcrest"]
                .values[0]
            )
            dberm.append(
                self.options_geotechnical[traject.sections[i[0]].name]
                .iloc[i[2] - 1]["dberm"]
                .values[0]
            )
            yes_no.append(
                self.options_geotechnical[traject.sections[i[0]].name]
                .iloc[i[2] - 1]["yes/no"]
                .values[0]
            )

            # get the option_index
            option_df = self.options[traject.sections[i[0]].name].loc[
                self.options[traject.sections[i[0]].name]["ID"] == ID[-1]
            ]
            if len(option_df) > 1:
                option_index.append(
                    self.options[traject.sections[i[0]].name]
                    .loc[self.options[traject.sections[i[0]].name]["ID"] == ID[-1]]
                    .loc[
                        self.options[traject.sections[i[0]].name]["dcrest"]
                        == dcrest[-1]
                    ]
                    .loc[
                        self.options[traject.sections[i[0]].name]["dberm"] == dberm[-1]
                    ]
                    .loc[
                        self.options[traject.sections[i[0]].name]["yes/no"]
                        == yes_no[-1]
                    ]
                    .index.values[0]
                )
            else:  # partial measure with no parameter variations
                option_index.append(
                    self.options[traject.sections[i[0]].name]
                    .loc[self.options[traject.sections[i[0]].name]["ID"] == ID[-1]]
                    .index.values[0]
                )
            # get the name
            names.append(
                solutions_dict[traject.sections[i[0]].name]
                .measure_table.loc[
                    solutions_dict[traject.sections[i[0]].name].measure_table["ID"]
                    == ID[-1]
                ]["Name"]
                .values[0][0]
            )
        self.TakenMeasures = pd.DataFrame(
            list(
                zip(sections, option_index, LCC, BC, ID, names, yes_no, dcrest, dberm)
            ),
            columns=TakenMeasuresHeaders,
        )

        # writing the probabilities to self.Probabilities
        tgrid = copy.deepcopy(self.T)
        # make sure it doesnt exceed the data:
        tgrid[-1] = np.size(Probabilities[0]["Overflow"], axis=1) - 1
        probabilities_columns = ["name", "mechanism"] + tgrid
        count = 0
        self.Probabilities = []
        for i in Probabilities:
            name = []
            mech = []
            probs = []
            for n in range(0, self.opt_parameters["N"]):
                for m in self.mechanisms:
                    name.append(traject.sections[n].name)
                    mech.append(m)
                    probs.append(i[m][n, np.array(tgrid)])
                    pass
                name.append(traject.sections[n].name)
                mech.append("Section")
                probs.append(np.sum(probs[-3:], axis=0))
            betas = np.array(pf_to_beta(probs))
            leftpart = pd.DataFrame(
                list(zip(name, mech)), columns=probabilities_columns[0:2]
            )
            rightpart = pd.DataFrame(betas, columns=tgrid)
            combined = pd.concat((leftpart, rightpart), axis=1)
            combined = combined.set_index(["name", "mechanism"])
            self.Probabilities.append(combined)

    def determine_risk_cost_curve(self, flood_damage: float, output_path: Path):
        """Determines risk-cost curve for greedy approach. Can be used to compare with a Pareto Frontier."""
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)

        if not hasattr(self, "TakenMeasures"):
            raise TypeError("TakenMeasures not found")
        costs = {}
        costs["TR"] = []
        # if (self.type == 'Greedy') or (self.type == 'TC'): #do a loop

        costs["LCC"] = np.cumsum(self.TakenMeasures["LCC"].values)
        count = 0
        for i in self.Probabilities:
            if output_path:
                costs["TR"].append(
                    calc_life_cycle_risks(
                        i,
                        self.discount_rate,
                        np.max(self.T),
                        flood_damage,
                        dumpPt=output_path.joinpath(
                            "Greedy_step_" + str(count) + ".csv"
                        ),
                    )
                )
            else:
                costs["TR"].append(
                    calc_life_cycle_risks(
                        i,
                        self.discount_rate,
                        np.max(self.T),
                        flood_damage,
                    )
                )
            count += 1
        costs["TC"] = np.add(costs["TR"], costs["LCC"])
        costs["TC_min"] = np.argmin(costs["TC"])

        return costs
