import copy
import logging
import time

import numpy as np

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.strategies.strategy_protocol import StrategyProtocol
from vrtool.decision_making.strategy_evaluation import evaluate_risk, update_probability
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.optimization.strategy_input.strategy_input import StrategyInput


class GreedyStrategy(StrategyProtocol):
    design_method: str

    def __init__(self, strategy_input: StrategyInput, config: VrtoolConfig) -> None:
        self.design_method = strategy_input.design_method
        self.options = strategy_input.options
        self.options_geotechnical = strategy_input.options_geotechnical
        self.options_height = strategy_input.options_height
        self.sections = strategy_input.sections

        self.opt_parameters = strategy_input.opt_parameters
        self.Pf = strategy_input.Pf
        self.LCCOption = strategy_input.LCCOption
        self.Cint_h = strategy_input.Cint_h
        self.Cint_g = strategy_input.Cint_g
        self.D = strategy_input.D
        self.Dint = strategy_input.Dint
        self.RiskGeotechnical = strategy_input.RiskGeotechnical
        self.RiskOverflow = strategy_input.RiskOverflow
        self.RiskRevetment = strategy_input.RiskRevetment

        self.config = config
        self.OI_horizon = config.OI_horizon
        self.mechanisms = config.mechanisms
        self._time_periods = config.T
        self.LE_in_section = config.LE_in_section

        self.measures_taken = []
        self.total_risk_per_step = []
        self.probabilities_per_step = []

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
            if optimal_counter_combination[i] >= 0:  # a measure was taken
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
        initial_mechanism_risk: np.ndarray,
        life_cycle_cost: np.ndarray,
        sh_array: np.ndarray,
        sg_array: np.ndarray,
        mechanism: MechanismEnum,
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
        index_counter = np.full(
            (number_of_sections,), -1, dtype=np.int32
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
                # take next step, exception if there is no valid measure. In that case exit the routine.
                if (
                    sh_array[ind_highest_risk, index_counter[ind_highest_risk] + 1]
                    == 999
                ):
                    logging.debug(
                        "Bundle quit after {} steps, weakest section has no more available measures".format(
                            run_number
                        )
                    )
                    break
                else:
                    index_counter[ind_highest_risk] += 1
            else:
                logging.debug(
                    "Bundle quit after {} steps, weakest section has no more available measures".format(
                        run_number
                    )
                )
                break

            # insert next cheapest measure from sorted list into mechanism_risk, then compute the LCC value and BC
            if mechanism == MechanismEnum.OVERFLOW:
                new_mechanism_risk[ind_highest_risk, :] = self.RiskOverflow[
                    ind_highest_risk,
                    sh_array[ind_highest_risk, index_counter[ind_highest_risk]],
                    :,
                ]
            elif mechanism == MechanismEnum.REVETMENT:
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
        mechanism: MechanismEnum,
        dim_sh: int,
    ):
        """Subroutine for overflow bundling that gets the correct indices for sh and sg for measures at a given section_no"""
        # make arrays for section
        sh_section_sorted = np.full((1, dim_sh), 999, dtype=int)
        sg_section = np.full((1, dim_sh), 999, dtype=int)

        # if there is already an investment we ensure that the reliability for none of the mechanisms is lower than the current investment
        if any(existing_investments[section_no, :] > 0):
            # if there is a GeotechnicalOption in place, we need to filter the options based on the current investment
            if existing_investments[section_no, 1] > 0:
                # note that matrix indices in existing_investments are always 1 higher than the investment id
                investment_id_sg = existing_investments[section_no, 1] - 1

                current_pf_stability = (
                    self.sections[section_no]
                    .sg_combinations[investment_id_sg]
                    .mechanism_year_collection.get_probabilities(
                        MechanismEnum.STABILITY_INNER,
                        np.arange(
                            0, self.Pf[MechanismEnum.STABILITY_INNER.name].shape[2]
                        ),
                    )
                )

                current_pf_piping = (
                    self.sections[section_no]
                    .sg_combinations[investment_id_sg]
                    .mechanism_year_collection.get_probabilities(
                        MechanismEnum.PIPING,
                        np.arange(0, self.Pf[MechanismEnum.PIPING.name].shape[2]),
                    )
                )

                # measure_pf_stability
                measure_pfs_stability = self.Pf[MechanismEnum.STABILITY_INNER.name][
                    section_no, :, :
                ]
                # measure_pf_piping
                measure_pfs_piping = self.Pf[MechanismEnum.PIPING.name][
                    section_no, :, :
                ]

                # get indices for rows in measure_pfs where measure_pf_stability and measure_pf_piping are smaller or equal to  current_pf_stability and current_pf_piping by comparing the numpy array
                comparison_geotechnical = np.argwhere(
                    np.all(measure_pfs_stability <= current_pf_stability, axis=1)
                    & np.all(measure_pfs_piping <= current_pf_piping, axis=1)
                ).flatten()
                # make a mask where geotechnical options are available
                available_measures_geotechnical = np.zeros(
                    len(measure_pfs_stability), dtype=bool
                )
                available_measures_geotechnical[comparison_geotechnical] = True

            else:
                # all available
                available_measures_geotechnical = np.ones(
                    len(measure_pfs_stability), dtype=bool
                )

            # same for HeightOptions
            if existing_investments[section_no, 0] > 0:
                current_pf = {}
                measure_pfs = {}
                investment_id_sh = existing_investments[section_no, 0] - 1
                # exclude rows for height options that are not safer than current.
                # Overflow must be present. Revetment is optional.
                current_pf[MechanismEnum.OVERFLOW] = (
                    self.sections[section_no]
                    .sh_combinations[investment_id_sh]
                    .mechanism_year_collection.get_probabilities(
                        MechanismEnum.OVERFLOW,
                        np.arange(0, self.Pf[MechanismEnum.OVERFLOW.name].shape[2]),
                    )
                )
                if (
                    MechanismEnum.REVETMENT
                    in self.sections[section_no].initial_assessment.get_mechanisms()
                ):
                    current_pf[MechanismEnum.REVETMENT] = (
                        self.sections[section_no]
                        .sh_combinations[investment_id_sh]
                        .mechanism_year_collection.get_probabilities(
                            MechanismEnum.REVETMENT,
                            np.arange(
                                0, self.Pf[MechanismEnum.REVETMENT.name].shape[2]
                            ),
                        )
                    )
                else:
                    try:  # case where some sections have revetments and some don't. Then we need to have 0's in the current_pf
                        current_pf[MechanismEnum.REVETMENT] = np.zeros(
                            self.Pf[MechanismEnum.REVETMENT.name].shape[2]
                        )
                    except:  # case where no revetment is present at any section
                        pass

                # check if all rows in comparison only contain True values
                if mechanism == MechanismEnum.OVERFLOW:
                    measure_pfs[MechanismEnum.OVERFLOW] = self.Pf[
                        MechanismEnum.OVERFLOW.name
                    ][section_no, :, :]
                    # get indices for rows in measure_pfs where all measure_pfs are greater than current_pf by comparing the numpy array
                    if (
                        MechanismEnum.REVETMENT
                        in self.sections[section_no].initial_assessment.get_mechanisms()
                    ):
                        # take combination of REVETMENT and OVERFLOW
                        measure_pfs[MechanismEnum.REVETMENT] = self.Pf[
                            MechanismEnum.REVETMENT.name
                        ][section_no, :, :]
                        comparison_height = np.where(
                            np.all(
                                measure_pfs[MechanismEnum.OVERFLOW]
                                <= current_pf[MechanismEnum.OVERFLOW],
                                axis=1,
                            )
                            & np.all(
                                measure_pfs[MechanismEnum.REVETMENT]
                                < current_pf[MechanismEnum.REVETMENT],
                                axis=1,
                            )
                        )
                    else:
                        # only look at OVERFLOW
                        comparison_height = np.where(
                            np.all(
                                measure_pfs[MechanismEnum.OVERFLOW]
                                < current_pf[MechanismEnum.OVERFLOW],
                                axis=1,
                            )
                        )

                elif mechanism == MechanismEnum.REVETMENT:
                    # overflow must be present so slightly different
                    measure_pfs[MechanismEnum.OVERFLOW] = self.Pf[
                        MechanismEnum.OVERFLOW.name
                    ][section_no, :, :]
                    measure_pfs[MechanismEnum.REVETMENT] = self.Pf[
                        MechanismEnum.OVERFLOW.name
                    ][section_no, :, :]
                    # get indices for rows in measure_pfs where all measure_pfs are greater than current_pf by comparing the numpy array
                    comparison_height = np.where(
                        np.all(
                            measure_pfs[MechanismEnum.OVERFLOW]
                            <= current_pf[MechanismEnum.OVERFLOW],
                            axis=1,
                        )
                        & np.all(
                            measure_pfs[MechanismEnum.REVETMENT]
                            < current_pf[MechanismEnum.REVETMENT],
                            axis=1,
                        )
                    )
                else:
                    raise Exception("Unknown mechanism in overflow bundling")

                available_measures_height = comparison_height
                # now replace the life_cycle_cost for measures that are not in comparison_height with a very high value:
                # the reliability for overflow/revetment has to increase so we do not want to pick these measures.
                mask = np.ones(life_cycle_cost.shape[1], dtype=bool)
                mask[available_measures_height] = False
            else:  # if there is no investment in height, all options are available
                mask = np.zeros(life_cycle_cost.shape[1], dtype=bool)

            # now replace the life_cycle_cost where available_measures_height is False with a very high value:
            # the reliability for overflow/revetment has to increase so we do not want to pick these measures.
            # get index of available_measures_height where `True`
            life_cycle_cost[
                section_no,
                mask,
                :,
            ] = 1e99

            # we get a matrix with the LCC values, and get the order of sh measures:
            lcc_subset = life_cycle_cost[section_no, :, comparison_geotechnical].T
            sh_order = np.argsort(np.min(lcc_subset, axis=1))
            sg_section[0, :] = comparison_geotechnical[np.argmin(lcc_subset, axis=1)][
                sh_order
            ]
            sh_section_sorted[0, :] = sh_order
            sh_section_sorted[0, :] = np.where(
                np.sort(np.min(lcc_subset, axis=1)) > 1e60, 999, sh_section_sorted[0, :]
            )
        elif (
            np.max(existing_investments[section_no, :]) == 0
        ):  # nothing has been invested yet
            # filter based on current reliability for Overflow or Revetment to make sure only improvements are included in the list
            if mechanism == MechanismEnum.OVERFLOW:
                # Overflow is always present for a section.
                current_pf_overflow = self.sections[
                    section_no
                ].initial_assessment.get_probabilities(
                    MechanismEnum.OVERFLOW,
                    np.arange(0, self.Pf[MechanismEnum.OVERFLOW.name].shape[2]),
                )
                measure_pfs = self.Pf[MechanismEnum.OVERFLOW.name][section_no, :, :]
                # get indices for rows in measure_pfs where all measure_pfs are greater than current_reliability_overflow by comparing the numpy array
                comparison_height = np.where(
                    np.all(measure_pfs < current_pf_overflow, axis=1)
                )

            elif mechanism == MechanismEnum.REVETMENT:
                try:  # if Revetment has been computed, get it from the assessment:
                    current_pf_overflow = self.sections[
                        section_no
                    ].initial_assessment.get_probabilities(
                        MechanismEnum.REVETMENT,
                        np.arange(0, self.Pf[MechanismEnum.OVERFLOW.name].shape[2]),
                    )
                    measure_pfs = self.Pf[MechanismEnum.REVETMENT.name][
                        section_no, :, :
                    ]
                    comparison_height = np.where(
                        np.all(measure_pfs < current_pf_overflow, axis=1)
                    )
                except:
                    # all measures are available
                    comparison_height = np.arange(
                        0, len(self.Pf[MechanismEnum.REVETMENT.name][section_no, :, :])
                    )

            else:
                raise Exception("Unknown mechanism in overflow bundling")

            available_measures_height = comparison_height

            # now replace the life_cycle_cost for measures that are not in comparison_height with a very high value:
            # the reliability for overflow/revetment has to increase so we do not want to pick these measures.
            mask = np.ones(life_cycle_cost.shape[1], dtype=bool)
            mask[available_measures_height] = False
            life_cycle_cost[section_no, mask, :] = 1e99

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
        mechanism: MechanismEnum,
        init_mechanism_risk: np.array,
        existing_investment: list,
        life_cycle_cost: np.array,
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
        setting: str = "fast",
        BCstop: float = 0.1,
        max_count: int = 600,
        f_cautious: float = 1.5,
    ):
        """This is the main routine for a greedy evaluation of all solutions."""

        # TODO put settings in config
        def calculate_total_risk(
            overflow_risk: np.ndarray,
            revetment_risk: np.ndarray,
            independent_risk: np.ndarray,
        ):
            return (
                np.sum(np.max(overflow_risk, axis=0))
                + np.sum(np.max(revetment_risk, axis=0))
                + np.sum(independent_risk)
            )

        start = time.time()
        # set start values:
        self.Cint_g[:, 0] = 1
        self.Cint_h[:, 0] = 1

        init_probability = {}
        init_overflow_risk = np.zeros(
            (self.opt_parameters["N"], self.opt_parameters["T"])
        )
        init_revetment_risk = np.zeros(
            (self.opt_parameters["N"], self.opt_parameters["T"])
        )
        init_independent_risk = np.zeros(
            (self.opt_parameters["N"], self.opt_parameters["T"])
        )
        for mechanism in self.mechanisms:
            init_probability[mechanism.name] = np.empty(
                (self.opt_parameters["N"], self.opt_parameters["T"])
            )
            if mechanism.name not in self.Pf:
                continue
            for n in range(0, self.opt_parameters["N"]):
                init_probability[mechanism.name][n, :] = self.Pf[mechanism.name][
                    n, 0, :
                ]
                if mechanism == MechanismEnum.OVERFLOW:
                    init_overflow_risk[n, :] = self.RiskOverflow[n, 0, :]
                elif mechanism == MechanismEnum.REVETMENT:
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
        TR_list = [
            calculate_total_risk(
                init_overflow_risk, init_revetment_risk, init_independent_risk
            )
        ]  # list to store the total risk for each step
        Measures_per_section = np.zeros((self.opt_parameters["N"], 2), dtype=np.int32)
        while count < max_count:
            init_risk = calculate_total_risk(
                init_overflow_risk, init_revetment_risk, init_independent_risk
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
                                new_independent_risk,
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
                                calculate_total_risk(
                                    new_overflow_risk,
                                    new_revetment_risk,
                                    new_independent_risk,
                                )
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
                MechanismEnum.OVERFLOW,
                copy.deepcopy(init_overflow_risk),
                copy.deepcopy(measure_list),
                copy.deepcopy(LifeCycleCost),
            )
            # for revetment:
            BC_bundleRevetment = 0.0
            if MechanismEnum.REVETMENT in self.mechanisms:
                (
                    revetment_bundle_index,
                    BC_bundleRevetment,
                ) = self.bundling_of_measures(
                    MechanismEnum.REVETMENT,
                    copy.deepcopy(init_revetment_risk),
                    copy.deepcopy(measure_list),
                    copy.deepcopy(LifeCycleCost),
                )

            # then in the selection of the measure we make a if-elif split with either the normal routine or an
            # 'overflow bundle'
            if np.isnan(np.max(BC)):
                ids = np.argwhere(np.isnan(BC))
                logging.error(
                    "NaN gevonden in matrix met kosten-batenratio. Uitvoer voor betreffende maatregel wordt gegenereerd."
                )
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
                if np.max(BC) >= BC_bundleOverflow and np.max(BC) >= BC_bundleRevetment:
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

                    init_revetment_risk[Index_Best[0], :] = copy.deepcopy(
                        self.RiskRevetment[Index_Best[0], Index_Best[1], :]
                    )

                    # TODO update risks
                    SpentMoney[Index_Best[0]] += copy.deepcopy(
                        LifeCycleCost[Index_Best]
                    )
                    self.LCCOption[Index_Best] = 1e99
                    Measures_per_section[Index_Best[0], 0] = Index_Best[1]
                    Measures_per_section[Index_Best[0], 1] = Index_Best[2]
                    Probabilities.append(copy.deepcopy(init_probability))
                    TR_list.append(
                        calculate_total_risk(
                            init_overflow_risk,
                            init_revetment_risk,
                            init_independent_risk,
                        )
                    )
                    logging.info(
                        "Enkele maatregel in optimalisatiestap {} (BC-ratio = {:.2f})".format(
                            count, BC[Index_Best]
                        )
                    )
                elif BC_bundleOverflow > BC_bundleRevetment:
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
                            init_independent_risk[IndexMeasure[0], :] = copy.deepcopy(
                                self.RiskGeotechnical[
                                    IndexMeasure[0], IndexMeasure[2], :
                                ]
                            )
                            init_overflow_risk[IndexMeasure[0], :] = copy.deepcopy(
                                self.RiskOverflow[IndexMeasure[0], IndexMeasure[1], :]
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
                            TR_list.append(
                                calculate_total_risk(
                                    init_overflow_risk,
                                    init_revetment_risk,
                                    init_independent_risk,
                                )
                            )
                    logging.info(
                        "Gebundelde maatregelen voor overslag in optimalisatiestap {} (BC-ratio = {:.2f})".format(
                            count, BC_bundleOverflow
                        )
                    )
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
                            init_independent_risk[IndexMeasure[0], :] = copy.deepcopy(
                                self.RiskGeotechnical[
                                    IndexMeasure[0], IndexMeasure[2], :
                                ]
                            )
                            init_overflow_risk[IndexMeasure[0], :] = copy.deepcopy(
                                self.RiskOverflow[IndexMeasure[0], IndexMeasure[1], :]
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
                            TR_list.append(
                                calculate_total_risk(
                                    init_overflow_risk,
                                    init_revetment_risk,
                                    init_independent_risk,
                                )
                            )
                    # add the height measures in separate entries in the measure list

                    # write them to the measure_list
                    logging.info(
                        "Gebundelde maatregelen voor bekleding in optimalisatiestap {} (BC-ratio = {:.2f})".format(
                            count, BC_bundleRevetment
                        )
                    )

            else:  # stop the search
                break
            count += 1
            if count == max_count:
                pass
                # Probabilities.append(copy.deepcopy(init_probability))
        # pd.DataFrame([risk_per_step,cost_per_step]).to_csv('GreedyResults_per_step.csv') #useful for debugging
        logging.info(
            "Totale rekentijd voor veiligheidsrendementoptimalisatie {:.2f} seconden".format(
                time.time() - start
            )
        )
        self.LCCOption = copy.deepcopy(InitialCostMatrix)
        self.measures_taken = measure_list
        self.total_risk_per_step = TR_list
        self.probabilities_per_step = Probabilities
