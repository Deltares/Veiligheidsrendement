import copy
import logging
import time

import numpy as np

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.strategies.strategy_protocol import StrategyProtocol
from vrtool.decision_making.traject_risk import TrajectRisk
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.optimization.strategy_input.strategy_input import StrategyInput


class GreedyStrategy(StrategyProtocol):
    design_method: str

    def __init__(self, strategy_input: StrategyInput, config: VrtoolConfig) -> None:
        self.design_method = strategy_input.design_method
        self.sections = strategy_input.sections

        self.LCCOption = strategy_input.LCCOption

        self.traject_risk = TrajectRisk(strategy_input.Pf, strategy_input.D)

        self.config = config
        self.OI_horizon = config.OI_horizon
        self.mechanisms = config.mechanisms
        self.time_periods = config.T
        self.LE_in_section = config.LE_in_section

        self.measures_taken = []
        self.total_risk_per_step = []
        self.probabilities_per_step = []
        self.selected_aggregated_measures = []

    def bundling_output(
        self,
        BC_list: list[float],
        counter_list: np.ndarray,
        sh_array: np.ndarray,
        sg_array: np.ndarray,
        existing_investments: np.ndarray,
    ) -> tuple[float, np.ndarray]:
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

        return np.max(BC_list), measure_index

    def bundling_loop(
        self,
        initial_mechanism_risk: np.ndarray,
        life_cycle_cost: np.ndarray,
        sh_array: np.ndarray,
        sg_array: np.ndarray,
        mechanism: MechanismEnum,
        n_runs: int = 100,
    ) -> tuple[list[float], list[np.ndarray]]:
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

        # counter that keeps track of the next cheapest option for each section
        index_counter = np.full((number_of_sections,), -1, dtype=np.int32)

        counter_list = []  # used to store the bundle indices
        BC_list = []  # used to store BC for each bundle

        # initialize overflow risk
        new_mechanism_risk = np.copy(initial_mechanism_risk)

        # here we start the loop. Note that we rarely make it to run 100, for larger problems this limit might need to be increased
        for _run_number in range(0, n_runs):
            # get weakest section

            ind_highest_risk = np.argmax(np.sum(new_mechanism_risk, axis=1))

            # We should increase the measure at the weakest section, but only if we have not reached the end of the array yet:
            if number_of_available_height_measures <= index_counter[
                ind_highest_risk
            ] or (
                sh_array[ind_highest_risk, index_counter[ind_highest_risk] + 1] == 999
            ):
                # take next step, exception if there is no valid measure. In that case exit the routine.
                logging.debug(
                    "Bundle quit after {} steps, weakest section has no more available measures".format(
                        _run_number
                    )
                )
                break

            index_counter[ind_highest_risk] += 1
            # insert next cheapest measure from sorted list into mechanism_risk, then compute the LCC value and BC
            _measure = (
                ind_highest_risk,
                sh_array[ind_highest_risk, index_counter[ind_highest_risk]],
                sg_array[ind_highest_risk, index_counter[ind_highest_risk]],
            )
            if mechanism == MechanismEnum.OVERFLOW:
                new_mechanism_risk[
                    ind_highest_risk, :
                ] = self.traject_risk.get_measure_risk(_measure, MechanismEnum.OVERFLOW)
            elif mechanism == MechanismEnum.REVETMENT:
                new_mechanism_risk[
                    ind_highest_risk, :
                ] = self.traject_risk.get_measure_risk(
                    _measure, MechanismEnum.REVETMENT
                )
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

            counter_list.append(np.copy(index_counter))

        return BC_list, counter_list

    def get_sg_sh_indices(
        self,
        section_no: int,
        life_cycle_cost: np.array,
        existing_investments: np.array,
        mechanism: MechanismEnum,
        dim_sh: int,
    ) -> tuple[np.ndarray, np.ndarray]:
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
                        np.arange(0, self.traject_risk.num_years),
                    )
                )

                current_pf_piping = (
                    self.sections[section_no]
                    .sg_combinations[investment_id_sg]
                    .mechanism_year_collection.get_probabilities(
                        MechanismEnum.PIPING,
                        np.arange(0, self.traject_risk.num_years),
                    )
                )

                # measure_pf_stability
                measure_pfs_stability = self.traject_risk.get_section_probabilities(
                    section_no, MechanismEnum.STABILITY_INNER
                )
                # measure_pf_piping
                measure_pfs_piping = self.traject_risk.get_section_probabilities(
                    section_no, MechanismEnum.PIPING
                )

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
                        np.arange(0, self.traject_risk.num_years),
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
                            np.arange(0, self.traject_risk.num_years),
                        )
                    )
                else:
                    try:  # case where some sections have revetments and some don't. Then we need to have 0's in the current_pf
                        current_pf[MechanismEnum.REVETMENT] = np.zeros(
                            self.traject_risk.num_years
                        )
                    except:  # case where no revetment is present at any section
                        pass

                # check if all rows in comparison only contain True values
                if mechanism == MechanismEnum.OVERFLOW:
                    measure_pfs[
                        MechanismEnum.OVERFLOW
                    ] = self.traject_risk.get_section_probabilities(
                        section_no, MechanismEnum.OVERFLOW
                    )
                    # get indices for rows in measure_pfs where all measure_pfs are greater than current_pf by comparing the numpy array
                    if (
                        MechanismEnum.REVETMENT
                        in self.sections[section_no].initial_assessment.get_mechanisms()
                    ):
                        # take combination of REVETMENT and OVERFLOW
                        measure_pfs[
                            MechanismEnum.REVETMENT
                        ] = self.traject_risk.get_section_probabilities(
                            section_no, MechanismEnum.REVETMENT
                        )
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
                    measure_pfs[
                        MechanismEnum.OVERFLOW
                    ] = self.traject_risk.get_section_probabilities(
                        section_no, MechanismEnum.OVERFLOW
                    )
                    measure_pfs[
                        MechanismEnum.REVETMENT
                    ] = self.traject_risk.get_section_probabilities(
                        section_no, MechanismEnum.OVERFLOW
                    )
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
                    np.arange(0, self.traject_risk.num_years),
                )
                measure_pfs = self.traject_risk.get_section_probabilities(
                    section_no, MechanismEnum.OVERFLOW
                )
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
                        np.arange(0, self.traject_risk.num_years),
                    )
                    measure_pfs = self.traject_risk.get_section_probabilities(
                        section_no, MechanismEnum.REVETMENT
                    )
                    comparison_height = np.where(
                        np.all(measure_pfs < current_pf_overflow, axis=1)
                    )
                except:
                    # all measures are available
                    comparison_height = np.arange(0, self.traject_risk.num_sh_measures)

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

    def _bundling_of_measures(
        self,
        mechanism: MechanismEnum,
        init_mechanism_risk: np.array,
        existing_investment_list: list,
        life_cycle_cost: np.array,
    ):
        """This function bundles the measures for which sections are dependent. It can be used for overflow and revetment"""
        _life_cycle_cost = np.copy(life_cycle_cost)
        number_of_sections = np.size(_life_cycle_cost, axis=0)

        # first we determine the existing investments and make a n,2 array for options for dependent and independent mechanisms
        _calculated_investments = np.zeros(
            (np.size(_life_cycle_cost, axis=0), 2), dtype=np.int32
        )

        for _idx, _ in enumerate(existing_investment_list):
            # sh
            _calculated_investments[
                existing_investment_list[_idx][0], 0
            ] = existing_investment_list[_idx][1]
            # sg
            _calculated_investments[
                existing_investment_list[_idx][0], 1
            ] = existing_investment_list[_idx][2]

        # prepare arrays
        sorted_sh = np.full(tuple(_life_cycle_cost.shape[0:2]), 999, dtype=int)
        sg_indices = np.full(tuple(_life_cycle_cost.shape[0:2]), 999, dtype=int)

        # then we loop over sections to get indices of those measures that are available
        for i in range(0, number_of_sections):
            sorted_sh[i, :], sg_indices[i, :] = self.get_sg_sh_indices(
                i,
                _life_cycle_cost,
                _calculated_investments,
                mechanism,
                _life_cycle_cost.shape[1],
            )

        # then we bundle the measures by getting the BC for the mechanism under consideration
        BC_list, counter_list = self.bundling_loop(
            init_mechanism_risk,
            _life_cycle_cost,
            sorted_sh,
            sg_indices,
            mechanism,
            n_runs=100,
        )

        # and we generate the required output
        if any(BC_list):
            BC_out, measure_index = self.bundling_output(
                BC_list, counter_list, sorted_sh, sg_indices, _calculated_investments
            )
            return measure_index, BC_out

        return [], 0

    def _add_aggregation(
        self, section_idx: int, sh_sequence_nr: int, sg_sequence_nr: int
    ):
        _aggregated_combinations = [
            _amc
            for _amc in self.sections[section_idx].aggregated_measure_combinations
            if _amc.sh_combination.sequence_nr == sh_sequence_nr - 1
            and _amc.sg_combination.sequence_nr == sg_sequence_nr - 1
        ]
        self.selected_aggregated_measures.append(
            (section_idx, _aggregated_combinations[0])
        )

    def evaluate(
        self,
        setting: str = "fast",
        BCstop: float = 0.1,
        max_count: int = 600,
        f_cautious: float = 1.5,
    ):
        """This is the main routine for a greedy evaluation of all solutions."""
        start = time.time()

        measure_list: list[tuple[int, int, int]] = []
        _probabilities = [
            self.traject_risk.get_initial_probabilities_dict(self.mechanisms)
        ]

        risk_per_step = []
        cost_per_step = [0]

        # TODO: add existing investments
        _spent_money = np.zeros(self.traject_risk.num_sections)
        _initial_cost_matrix = np.copy(self.LCCOption)

        BC_list = []

        # list to store the total risk for each step
        _total_risk_list = [self.traject_risk.get_total_risk()]

        _measures_per_section = np.zeros(
            (self.traject_risk.num_sections, 2), dtype=np.int32
        )
        for _count in range(0, max_count):
            init_risk = self.traject_risk.get_total_risk()

            risk_per_step.append(init_risk)
            cost_per_step.append(np.sum(_spent_money))
            # first we compute the BC-ratio for each combination of Sh, Sg, for each section
            _life_cycle_cost = np.full(
                [
                    self.traject_risk.num_sections,
                    self.traject_risk.num_sh_measures,
                    self.traject_risk.num_sg_measures,
                ],
                1e99,
            )
            _total_risk = np.full(_life_cycle_cost.shape, init_risk)
            for n in range(0, self.traject_risk.num_sections):
                # for each section, start from index 1 to prevent putting inf in top left cell
                for sg in range(1, self.traject_risk.num_sg_measures):
                    for sh in range(0, self.traject_risk.num_sh_measures):
                        if self.LCCOption[n, sh, sg] >= 1e20:
                            continue

                        _life_cycle_cost[n, sh, sg] = np.subtract(
                            self.LCCOption[n, sh, sg], _spent_money[n]
                        )

                        _total_risk[
                            n, sh, sg
                        ] = self.traject_risk.get_total_risk_for_measure((n, sh, sg))

            # do not go back:
            _life_cycle_cost = np.where(_life_cycle_cost <= 0, 1e99, _life_cycle_cost)
            dR = np.subtract(init_risk, _total_risk)
            BC = np.divide(dR, _life_cycle_cost)  # risk reduction/cost [n,sh,sg]
            TC = np.add(_life_cycle_cost, _total_risk)

            # compute additional measures where we combine overflow/revetment measures, here we optimize a package, purely based
            # on overflow/revetment, and compute a BC ratio for a combination of measures at different sections.

            # for overflow:
            BC_bundleOverflow = 0
            (overflow_bundle_index, BC_bundleOverflow) = self._bundling_of_measures(
                MechanismEnum.OVERFLOW,
                self.traject_risk.get_mechanism_risk(MechanismEnum.OVERFLOW),
                copy.deepcopy(measure_list),
                _life_cycle_cost,
            )
            # for revetment:
            BC_bundleRevetment = 0.0
            if MechanismEnum.REVETMENT in self.mechanisms:
                (
                    revetment_bundle_index,
                    BC_bundleRevetment,
                ) = self._bundling_of_measures(
                    MechanismEnum.REVETMENT,
                    self.traject_risk.get_mechanism_risk(MechanismEnum.REVETMENT),
                    copy.deepcopy(measure_list),
                    _life_cycle_cost,
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
                        _section_idx, _sh_sequence_nr, _sg_sequence_nr = Index_Best
                        self._add_aggregation(
                            _section_idx, _sh_sequence_nr, _sg_sequence_nr
                        )
                        # update init_probability
                        self.traject_risk.update_probabilities_for_measure(Index_Best)

                    elif (setting == "fast") or (setting == "cautious"):
                        BC_sections = np.empty(self.traject_risk.num_sections)
                        # find best measure for each section
                        for n in range(0, self.traject_risk.num_sections):
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
                            self._add_aggregation(
                                Index_Best[0], fast_measure[0], fast_measure[1]
                            )
                        else:
                            measure_list.append(Index_Best)
                            _section_idx, _sh_sequence_nr, _sg_sequence_nr = Index_Best
                            self._add_aggregation(
                                _section_idx, _sh_sequence_nr, _sg_sequence_nr
                            )
                    BC_list.append(BC[Index_Best])
                    self.traject_risk.update_probabilities_for_measure(Index_Best)

                    # TODO update risks
                    _spent_money[Index_Best[0]] += _life_cycle_cost[Index_Best]
                    self.LCCOption[Index_Best] = 1e99
                    _measures_per_section[Index_Best[0], 0] = Index_Best[1]
                    _measures_per_section[Index_Best[0], 1] = Index_Best[2]
                    _probabilities.append(
                        self.traject_risk.get_initial_probabilities_dict(
                            self.mechanisms
                        )
                    )
                    _total_risk_list.append(self.traject_risk.get_total_risk())

                    logging.info(
                        "Enkele maatregel in optimalisatiestap {} (BC-ratio = {:.2f})".format(
                            _count, BC[Index_Best]
                        )
                    )
                elif BC_bundleOverflow > BC_bundleRevetment:
                    for j in range(0, self.traject_risk.num_sections):
                        if overflow_bundle_index[j, 0] != _measures_per_section[j, 0]:
                            IndexMeasure = (
                                j,
                                overflow_bundle_index[j, 0],
                                overflow_bundle_index[j, 1],
                            )

                            measure_list.append(IndexMeasure)
                            (
                                _section_idx,
                                _sh_sequence_nr,
                                _sg_sequence_nr,
                            ) = IndexMeasure
                            self._add_aggregation(
                                _section_idx, _sh_sequence_nr, _sg_sequence_nr
                            )
                            BC_list.append(BC_bundleOverflow)

                            self.traject_risk.update_probabilities_for_measure(
                                IndexMeasure
                            )

                            _spent_money[IndexMeasure[0]] += _life_cycle_cost[
                                IndexMeasure
                            ]
                            self.LCCOption[IndexMeasure] = 1e99
                            _measures_per_section[IndexMeasure[0], 0] = IndexMeasure[1]
                            # no update of geotechnical risk needed
                            _probabilities.append(
                                self.traject_risk.get_initial_probabilities_dict(
                                    self.mechanisms
                                )
                            )
                            _total_risk_list.append(self.traject_risk.get_total_risk())

                    logging.info(
                        "Gebundelde maatregelen voor overslag in optimalisatiestap {} (BC-ratio = {:.2f})".format(
                            _count, BC_bundleOverflow
                        )
                    )
                elif BC_bundleRevetment > np.max(BC):
                    for j in range(0, self.traject_risk.num_sections):
                        if revetment_bundle_index[j, 0] != _measures_per_section[j, 0]:
                            IndexMeasure = (
                                j,
                                revetment_bundle_index[j, 0],
                                revetment_bundle_index[j, 1],
                            )

                            measure_list.append(IndexMeasure)
                            (
                                _section_idx,
                                _sh_sequence_nr,
                                _sg_sequence_nr,
                            ) = IndexMeasure
                            self._add_aggregation(
                                _section_idx, _sh_sequence_nr, _sg_sequence_nr
                            )
                            BC_list.append(BC_bundleRevetment)

                            self.traject_risk.update_probabilities_for_measure(
                                IndexMeasure
                            )

                            _spent_money[IndexMeasure[0]] += _life_cycle_cost[
                                IndexMeasure
                            ]
                            self.LCCOption[IndexMeasure] = 1e99
                            _measures_per_section[IndexMeasure[0], 0] = IndexMeasure[1]
                            # no update of geotechnical risk needed
                            _probabilities.append(
                                self.traject_risk.get_initial_probabilities_dict(
                                    self.mechanisms
                                )
                            )
                            _total_risk_list.append(self.traject_risk.get_total_risk())

                    logging.info(
                        "Gebundelde maatregelen voor bekleding in optimalisatiestap {} (BC-ratio = {:.2f})".format(
                            _count, BC_bundleRevetment
                        )
                    )

            else:
                break

        logging.info(
            "Totale rekentijd voor veiligheidsrendementoptimalisatie {:.2f} seconden".format(
                time.time() - start
            )
        )
        self.LCCOption = _initial_cost_matrix
        self.measures_taken = measure_list
        self.total_risk_per_step = _total_risk_list
        self.probabilities_per_step = _probabilities
