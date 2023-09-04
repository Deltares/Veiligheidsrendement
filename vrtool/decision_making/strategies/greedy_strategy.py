import copy
import logging
import time
from pathlib import Path
from typing import Dict

import numpy
import numpy as np
import pandas as pd

from vrtool.decision_making.solutions import Solutions
from vrtool.decision_making.strategies.strategy_base import StrategyBase
from vrtool.decision_making.strategy_evaluation import (
    calc_life_cycle_risks,
    evaluate_risk,
    update_probability,
)
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.probabilistic_tools.probabilistic_functions import pf_to_beta


class GreedyStrategy(StrategyBase):
    def bundling_output(
        self, BC_list, counter_list, sh_array, sg_array, existing_investments
    ):
        no_of_sections = sh_array.shape[0]
        maximum_BC_index = np.array(BC_list).argmax()
        optimal_counter_combination = counter_list[maximum_BC_index]
        # convert measure_index to sh based on sorted_indices
        sg_index = np.zeros((no_of_sections,))
        measure_index = np.zeros((no_of_sections,), dtype=np.int32)
        for i in range(0, no_of_sections):
            if optimal_counter_combination[i] != 0:  # a measure was taken
                measure_index[i] = sh_array[i, optimal_counter_combination[i]]
                sg_index[i] = sg_array[i, optimal_counter_combination[i]]
            else:  # no measure was taken
                measure_index[i] = existing_investments[i, 0]
                sg_index[i] = existing_investments[i, 1]

        measure_index = (
            np.append(measure_index, sg_index)
            .reshape((2, no_of_sections))
            .T.astype(np.int32)
        )
        BC_out = np.max(BC_list)

        return BC_out, measure_index

    def bundling_loop(
        self,
        initial_mechanism_risk: numpy.ndarray,
        life_cycle_cost: numpy.ndarray,
        sh_array: numpy.ndarray,
        sg_array: numpy.ndarray,
        mechanism: str,
        n_runs: int = 100,
    ):
        """
        Bundles measures for dependent mechanisms (overflow or revetment) and returns the optimal bundling combination.
        It only looks at risk reduction for dependent mechanism considered, not for the other mechanisms.

        Args:
            initial_mechanism_risk: Initial risk of the dependent mechanism.
            life_cycle_cost: Life cycle cost of the possible measures for each section.
            sh_array: Array containing the indices of the possible measures for dependent mechanisms for each section. These indices are sorted from cheapest to most expensive.
            sg_array: Array containing the indices of the possible measures for independent mechanisms for each section. These indices correspond to the cheapest available sg combination for the corresponding sh.
                        Note that sh_array and sg_array have the same dimensions.
            mechanism: Array containing the mechanism type for each section (Overflow or Revetment).
            n_runs: Number of runs for the bundling loop. Default = 100, but typically less is sufficient.
        """

        # first initialize some relevant arrays and values for the loop for bundling measures
        number_of_sections = sh_array.shape[0]
        number_of_available_height_measures = sh_array.shape[1] - 1
        LCC_values = np.zeros((number_of_sections,))  # total LCC spent for each section
        index_counter = np.zeros(
            (number_of_sections,), dtype=np.int32
        )  # counter that keeps track of the next cheapest option for each section

        run_number = 0  # used for counting the loop
        counter_list = []  # used to store the bundle indices
        BC_list = []  # used to store BC for each bundle
        highest_risk_section_indices = []  # used to store index of weakest section
        new_mechanism_risk = copy.deepcopy(
            initial_mechanism_risk
        )  # initialize overflow risk
        # here we start the loop. Note that we rarely make it to run 100, for larger problems this limit might need to be increased
        while run_number < n_runs:
            # get weakest section

            ind_highest_risk = np.argmax(np.sum(new_mechanism_risk, axis=1))

            # We should increase the measure at the weakest section, but only if we have not reached the end of the array yet:
            if number_of_available_height_measures > index_counter[ind_highest_risk]:
                index_counter[ind_highest_risk] += 1
                # take next step, exception if there is no valid measure. In that case exit the routine.
                if sh_array[ind_highest_risk, index_counter[ind_highest_risk]] == 999:
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

            # insert next cheapest measure from sorted list into mechanism_risk, then compute the LCC value and BC
            if mechanism == "Overflow":
                new_mechanism_risk[ind_highest_risk, :] = self.RiskOverflow[
                    ind_highest_risk,
                    sh_array[ind_highest_risk, index_counter[ind_highest_risk]],
                    :,
                ]
            elif mechanism == "Revetment":
                new_mechanism_risk[ind_highest_risk, :] = self.RiskRevetment[
                    ind_highest_risk,
                    sh_array[ind_highest_risk, index_counter[ind_highest_risk]],
                    :,
                ]
            else:
                raise ValueError("Mechanism {} not recognized".format(mechanism))

            LCC_values[ind_highest_risk] = np.min(
                life_cycle_cost[
                    ind_highest_risk,
                    sh_array[ind_highest_risk, index_counter[ind_highest_risk]],
                    sg_array[ind_highest_risk, index_counter[ind_highest_risk]],
                ]
            )
            BC = (
                np.sum(np.max(initial_mechanism_risk, axis=0))
                - np.sum(np.max(new_mechanism_risk, axis=0))
            ) / np.sum(LCC_values)
            # store results of step:
            if np.isnan(BC):
                BC_list.append(0.0)
            else:
                BC_list.append(BC)
            highest_risk_section_indices.append(ind_highest_risk)

            counter_list.append(copy.deepcopy(index_counter))

            # in the next step, the next measure should be taken for this section
            run_number += 1
        return BC_list, counter_list, highest_risk_section_indices

    def get_sg_sh_indices(
        self,
        section_no: int,
        life_cycle_cost: np.array,
        existing_investments: np.array,
        mechanism: str,
        dim_sh: int,
        traject: DikeTraject,
    ):
        """Subroutine for overflow bundling that gets the correct indices for sh and sg for measures at a given section_no"""
        # make arrays for section
        sh_section_sorted = np.full((1, dim_sh), 999, dtype=int)
        sg_section = np.full((1, dim_sh), 999, dtype=int)

        GeotechnicalOptions = self.options_geotechnical[
            traject.sections[section_no].name
        ]
        HeightOptions = self.options_height[traject.sections[section_no].name]
        # if there is already an investment we ensure that the reliability for none of the mechanisms is lower than the current investment
        if any(existing_investments[section_no, :] > 0):
            # if there is a GeotechnicalOption in place, we need to filter the options based on the current investment
            if existing_investments[section_no, 1] > 0:
                investment_id = (
                    existing_investments[section_no, 1] - 1
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
            if existing_investments[section_no, 0] > 0:
                # exclude rows for height options that are not safer than current
                current_investment_overflow = HeightOptions.iloc[
                    existing_investments[section_no, 0] - 1
                ]["Overflow"]
                # TODO turn on revetment once the proper data is available.
                # current_investment_revetment = HeightOptions.iloc[existing_investments[i, 0] - 1]['Revetment']
                current_investment_revetment = HeightOptions.iloc[
                    existing_investments[section_no, 0] - 1
                ]["Overflow"]
                # check if all rows in comparison only contain True values
                if mechanism == "Overflow":
                    comparison_height = (
                        HeightOptions.Overflow >= current_investment_overflow
                    )  # & (HeightOptions.Revetment >= current_investment_revetment)
                    # comparison_height = (HeightOptions.Overflow > current_investment_overflow) #& (HeightOptions.Revetment >= current_investment_revetment)
                elif mechanism == "Revetment":
                    comparison_height = (
                        HeightOptions.Overflow >= current_investment_overflow
                    )  # & (HeightOptions.Revetment > current_investment_revetment)
                else:
                    raise Exception("Unknown mechanism in overflow bundling")

                # available_measures_height = comparison_height.any(axis=1)
                available_measures_height = comparison_height.all(axis=1)
            else:  # if there is no investment in height, all options are available
                available_measures_height = pd.Series(
                    np.ones(len(HeightOptions), dtype=bool)
                )

            # now replace the life_cycle_cost where available_measures_height is False with a very high value:
            # the reliability for overflow/revetment has to increase so we do not want to pick these measures.
            life_cycle_cost[
                section_no,
                available_measures_height[~available_measures_height].index + 1,
                :,
            ] = 1e99

            # next we get the ids for the possible geotechnical measures
            ids = (
                available_measures_geotechnical[
                    available_measures_geotechnical
                ].index.values
                + 1
            )

            # we get a matrix with the LCC values, and get the order of sh measures:
            lcc_subset = life_cycle_cost[section_no, :, ids].T
            sh_order = np.argsort(np.min(lcc_subset, axis=1))
            sg_section[0, :] = np.array(ids)[np.argmin(lcc_subset, axis=1)][sh_order]
            sh_section_sorted[0, :] = sh_order
            sh_section_sorted[0, :] = np.where(
                np.sort(np.min(lcc_subset, axis=1)) > 1e60, 999, sh_section_sorted[0, :]
            )
        elif (
            np.max(existing_investments[section_no, :]) == 0
        ):  # nothing has been invested yet
            sg_section[0, :] = np.argmin(life_cycle_cost[section_no, :, :], axis=1)
            LCCs = np.min(life_cycle_cost[section_no, :, :], axis=1)
            sh_section_sorted[0, :] = np.argsort(LCCs)
            sh_section_sorted[0, :] = np.where(
                np.sort(LCCs) > 1e60, 999, sh_section_sorted[0, 0 : len(LCCs)]
            )
            sg_section[0, 0 : len(LCCs)] = sg_section[0, 0 : len(LCCs)][
                np.argsort(LCCs)
            ]
        else:
            logging.error(
                "Unknown measure type in overflow bundling (error can be removed?)"
            )

        return sh_section_sorted, sg_section

    def bundling_of_measures(
        self,
        mechanism: str,
        init_mechanism_risk: np.array,
        existing_investment: list,
        life_cycle_cost: np.array,
        traject: DikeTraject,
    ):
        """This function bundles the measures for which sections are dependent. It can be used for overflow and revetment"""
        life_cycle_cost = copy.deepcopy(life_cycle_cost)

        number_of_sections = np.size(life_cycle_cost, axis=0)
        # first we determine the existing investments and make a n,2 array for options for dependent and independent mechanisms
        existing_investments = np.zeros(
            (np.size(life_cycle_cost, axis=0), 2), dtype=np.int32
        )

        if len(existing_investment) > 0:
            for i in range(0, len(existing_investment)):
                existing_investments[
                    existing_investment[i][0], 0
                ] = existing_investment[i][
                    1
                ]  # sh
                existing_investments[
                    existing_investment[i][0], 1
                ] = existing_investment[i][
                    2
                ]  # sg

        # prepare arrays
        sorted_sh = np.full(tuple(life_cycle_cost.shape[0:2]), 999, dtype=int)
        LCC_values = np.zeros((life_cycle_cost.shape[0],))
        sg_indices = np.full(tuple(life_cycle_cost.shape[0:2]), 999, dtype=int)

        # then we loop over sections to get indices of those measures that are available
        for i in range(0, number_of_sections):
            sorted_sh[i, :], sg_indices[i, :] = self.get_sg_sh_indices(
                i,
                life_cycle_cost,
                existing_investments,
                mechanism,
                life_cycle_cost.shape[1],
                traject,
            )

        # then we bundle the measures by getting the BC for the mechanism under consideration
        BC_list, counter_list, weak_list = self.bundling_loop(
            init_mechanism_risk,
            life_cycle_cost,
            sorted_sh,
            sg_indices,
            mechanism,
            n_runs=100,
        )

        # and we generate the required output
        if len(BC_list) > 0:
            BC_out, measure_index = self.bundling_output(
                BC_list, counter_list, sorted_sh, sg_indices, existing_investments
            )
            return measure_index, BC_out
        else:
            return [], 0

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
        init_revetment_risk = np.empty(
            (self.opt_parameters["N"], self.opt_parameters["T"])
        )
        init_independent_risk = np.empty(
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
                elif m == "Revetment":
                    init_revetment_risk[n, :] = self.RiskRevetment[n, 0, :]
                else:
                    init_independent_risk[n, :] = self.RiskGeotechnical[n, 0, :]

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
            init_risk = (
                np.sum(np.max(init_overflow_risk, axis=0))
                + np.sum(np.max(init_revetment_risk, axis=0))
                + np.sum(init_independent_risk)
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
                            (
                                new_overflow_risk,
                                new_revetment_risk,
                                new_geotechnical_risk,
                            ) = evaluate_risk(
                                copy.deepcopy(init_overflow_risk),
                                copy.deepcopy(init_revetment_risk),
                                copy.deepcopy(init_independent_risk),
                                self,
                                n,
                                sh,
                                sg,
                                self.config,
                            )
                            TotalRisk[n, sh, sg] = copy.deepcopy(
                                np.sum(np.max(new_overflow_risk, axis=0))
                                + np.sum(np.max(new_revetment_risk, axis=0))
                                + np.sum(new_geotechnical_risk)
                            )
                        else:
                            pass
            # do not go back:
            LifeCycleCost = np.where(LifeCycleCost <= 0, 1e99, LifeCycleCost)
            dR = np.subtract(init_risk, TotalRisk)
            BC = np.divide(dR, LifeCycleCost)  # risk reduction/cost [n,sh,sg]
            TC = np.add(LifeCycleCost, TotalRisk)

            # compute additional measures where we combine overflow/revetment measures, here we optimize a package, purely based
            # on overflow/revetment, and compute a BC ratio for a combination of measures at different sections.

            # for overflow:
            BC_bundleOverflow = 0
            (overflow_bundle_index, BC_bundleOverflow) = self.bundling_of_measures(
                "Overflow",
                copy.deepcopy(init_overflow_risk),
                copy.deepcopy(measure_list),
                copy.deepcopy(LifeCycleCost),
                copy.deepcopy(traject),
            )
            # for revetment:
            BC_bundleRevetment = 0.0
            if "Revetment" in self.mechanisms:
                (
                    revetment_bundle_index,
                    BC_bundleRevetment,
                ) = self.bundling_of_measures(
                    "Revetment",
                    copy.deepcopy(init_revetment_risk),
                    copy.deepcopy(measure_list),
                    copy.deepcopy(LifeCycleCost),
                    copy.deepcopy(traject),
                )

            # then in the selection of the measure we make a if-elif split with either the normal routine or an
            # 'overflow bundle'
            if np.isnan(np.max(BC)):
                ids = np.argwhere(np.isnan(BC))
                for i in range(0, ids.shape[0]):
                    error_measure = self.get_measure_from_index(ids[i, :])
                    logging.error(error_measure)
                    # TODO think about a more sophisticated error catch here, as currently tracking the error is extremely difficult.
                raise ValueError("nan value encountered in BC-ratio")
            if (
                (np.max(BC) > BCstop)
                or (BC_bundleOverflow > BCstop)
                or (BC_bundleRevetment > BCstop)
            ):
                if np.max(BC) >= BC_bundleOverflow or np.max(BC) >= BC_bundleRevetment:
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
                    init_independent_risk[Index_Best[0], :] = copy.deepcopy(
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
                elif BC_bundleOverflow > np.max(BC):
                    for j in range(0, self.opt_parameters["N"]):
                        if overflow_bundle_index[j, 0] != Measures_per_section[j, 0]:
                            IndexMeasure = (
                                j,
                                overflow_bundle_index[j, 0],
                                overflow_bundle_index[j, 1],
                            )

                            measure_list.append(IndexMeasure)
                            BC_list.append(BC_bundleOverflow)
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
                elif BC_bundleRevetment > np.max(BC):
                    for j in range(0, self.opt_parameters["N"]):
                        if revetment_bundle_index[j, 0] != Measures_per_section[j, 0]:
                            IndexMeasure = (
                                j,
                                revetment_bundle_index[j, 0],
                                revetment_bundle_index[j, 1],
                            )

                            measure_list.append(IndexMeasure)
                            BC_list.append(BC_bundleRevetment)
                            init_probability = update_probability(
                                init_probability, self, IndexMeasure
                            )
                            init_revetment_risk[IndexMeasure[0], :] = copy.deepcopy(
                                self.RiskRevetment[IndexMeasure[0], IndexMeasure[1], :]
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
            "beta_target",
            "transition_level",
            "dberm",
        ]
        sections = []
        LCC = []
        LCC2 = []
        LCC_invested = np.zeros((len(traject.sections)))
        ID = []
        dcrest = []
        dberm = []
        beta_target = []
        transition_level = []
        yes_no = []
        option_index = []
        names = []
        # write the first line:
        sections.append("")
        LCC.append(0)
        ID.append("")
        dcrest.append("")
        beta_target.append("")
        transition_level.append("")
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
                    if (self._heightMeasureIsZero(traject, i)) and (
                        self._geotechnicalMeasureIsZero(traject, i)
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
                    raise ValueError(
                        "warning, conflicting IDs found for measures, ID_relevant: '{}' ID2: '{}'".format(
                            ID_relevant, ID2
                        )
                    )
            else:
                ID2 = ""
                ID.append(ID1)

            # get the parameters
            dcrest.append(
                self.options_height[traject.sections[i[0]].name]
                .iloc[i[1] - 1]["dcrest"]
                .values[0]
            )
            beta_target.append(
                self.options_height[traject.sections[i[0]].name]
                .iloc[i[1] - 1]["beta_target"]
                .values[0]
            )
            transition_level.append(
                self.options_height[traject.sections[i[0]].name]
                .iloc[i[1] - 1]["transition_level"]
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
                        self.options[traject.sections[i[0]].name]["beta_target"]
                        == beta_target[-1]
                    ]
                    .loc[
                        self.options[traject.sections[i[0]].name]["transition_level"]
                        == transition_level[-1]
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
                zip(
                    sections,
                    option_index,
                    LCC,
                    BC,
                    ID,
                    names,
                    yes_no,
                    dcrest,
                    beta_target,
                    transition_level,
                    dberm,
                )
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

    def _heightMeasureIsZero(self, traject, i) -> bool:
        """
        Helper function for write_greedy_results
        """
        options = self.options_height[traject.sections[i[0]].name].iloc[i[1] - 1]
        dcrest = options["dcrest"].values[0]
        transition_level = options["transition_level"].values[0]
        beta_target = options["beta_target"].values[0]
        return dcrest == 0.0 and transition_level == -999.0 and beta_target == -999.0

    def _geotechnicalMeasureIsZero(self, traject, i) -> bool:
        """
        Helper function for write_greedy_results
        """
        options = self.options_geotechnical[traject.sections[i[0]].name].iloc[i[2] - 1]
        dberm = options["dberm"].values[0]
        return dberm == 0.0

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
