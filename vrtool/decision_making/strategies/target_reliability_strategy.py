from __future__ import annotations

import logging
import math
from collections import defaultdict

import numpy as np

from vrtool.common.enums import step_type_enum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.strategies.strategy_protocol import StrategyProtocol
from vrtool.decision_making.strategies.strategy_step import StrategyStep
from vrtool.decision_making.traject_risk import TrajectRisk
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.cross_sectional_requirements import (
    CrossSectionalRequirements,
)
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.optimization.measures.aggregated_measures_combination import (
    AggregatedMeasureCombination,
)
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.optimization.strategy_input.strategy_input import StrategyInput


class TargetReliabilityStrategy(StrategyProtocol):
    """Subclass for evaluation in accordance with basic OI2014 approach.
    This ensures that for a certain time horizon, each section satisfies the cross-sectional target reliability
    """

    @property
    def total_cost(self) -> float:
        if not self.optimization_steps:
            return 0.0
        return self.optimization_steps[-1].total_cost

    def __init__(self, strategy_input: StrategyInput, config: VrtoolConfig):
        self.OI_horizon = config.OI_horizon
        self.time_periods = config.T
        self.sections = strategy_input.sections
        self.optimization_steps = []

        self.traject_risk = TrajectRisk(strategy_input.Pf, strategy_input.D)

    def check_cross_sectional_requirements(
        self,
        section_idx: int,
        measure_idx: int,
        cross_sectional_requirements: CrossSectionalRequirements,
        year: int,
        mechanisms: list[MechanismEnum],
    ) -> bool:
        """This function checks if the cross-sectional requirements are met for a given measure and year.
        If the requirements are not met for any of the mechanisms, the function returns False, otherwise True.
        """
        _section_as_input: SectionAsInput = self.sections[section_idx]
        _measure = _section_as_input.aggregated_measure_combinations[measure_idx]
        for _mechanism in mechanisms:
            if _mechanism in [MechanismEnum.OVERFLOW, MechanismEnum.REVETMENT]:
                # look in sh, if any mechanism is not satisfied, return a False
                if (
                    _measure.sh_combination.mechanism_year_collection.get_probabilities(
                        _mechanism, [year]
                    )[0]
                    > cross_sectional_requirements.cross_sectional_requirement_per_mechanism[
                        _mechanism
                    ]
                ):
                    return False
            elif _mechanism in [MechanismEnum.PIPING, MechanismEnum.STABILITY_INNER]:
                _cross_sectional_requirement = cross_sectional_requirements.cross_sectional_requirement_per_mechanism[
                    _mechanism
                ]
                _a_factor = cross_sectional_requirements.dike_section_a_piping
                _b_factor = cross_sectional_requirements.dike_traject_b_piping
                if _mechanism == MechanismEnum.STABILITY_INNER:
                    _a_factor = (
                        cross_sectional_requirements.dike_section_a_stability_inner
                    )
                    _b_factor = (
                        cross_sectional_requirements.dike_traject_b_stability_inner
                    )

                _le_factor = max(
                    cross_sectional_requirements.dike_section_length
                    * _a_factor
                    / _b_factor,
                    1.0,
                )
                _section_requirement = _cross_sectional_requirement * _le_factor
                if (
                    _measure.sg_combination.mechanism_year_collection.get_probabilities(
                        _mechanism, [year]
                    )[0]
                    > _section_requirement
                ):
                    return False
        return True

    def _get_failure_probability_of_invalid_mechanisms(
        self,
        section_idx: int,
        measure_idx: int,
        year: int,
        mechanisms: list[MechanismEnum],
    ) -> float:
        """This function gets the failure probabilities of the mechanisms that are not satisfied by the measure and returns the total failure probability.

        Args:
            measure (AggregatedMeasureCombination): The measure for which the failure probability is to be calculated.
            year (int): The year for which the failure probability is to be calculated.
            mechanisms (list[MechanismEnum]): The mechanisms for which the failure probability is to be calculated.

            Returns:
                float: The total failure probability for the mechanisms that are not satisfied by the measure.
        """
        _p_nonf: float = 1
        _measure = self.sections[section_idx].aggregated_measure_combinations[
            measure_idx
        ]
        for _mechanism in mechanisms:
            if _mechanism in [MechanismEnum.OVERFLOW, MechanismEnum.REVETMENT]:
                # look in sh, if any mechanism is not satisfied, return a False
                _p_nonf *= (
                    1
                    - _measure.sh_combination.mechanism_year_collection.get_probabilities(
                        _mechanism, [year]
                    )[
                        0
                    ]
                )
            elif _mechanism in [MechanismEnum.PIPING, MechanismEnum.STABILITY_INNER]:
                _p_nonf *= (
                    1
                    - _measure.sg_combination.mechanism_year_collection.get_probabilities(
                        _mechanism, [year]
                    )[
                        0
                    ]
                )
        return 1 - _p_nonf

    def check_measure_validity(
        self,
        measure_idx: int,
        section_idx: int,
        mechanisms: list[MechanismEnum],
        cross_sectional_requirements: CrossSectionalRequirements,
        investment_year: int,
        design_year: int,
    ) -> bool:
        """Check if the measure combination is valid for the mechanisms given as input.

        Args:
            measure_combination (AggregatedMeasureCombination): The measure combination to check.
            mechanisms (list[MechanismEnum]): The mechanisms to check the measure combination for.
            cross_sectional_requirements (CrossSectionalRequirements): The cross-sectional requirements for the dike traject.
            investment_year (int): The investment year for the measure combination.
            design_year (int): The design year for the measure combination.

            Returns:
                bool: True if the measure combination is valid for the mechanisms, False otherwise.
        """
        _measure_combination = self.sections[
            section_idx
        ].aggregated_measure_combinations[measure_idx]
        if _measure_combination.year != investment_year:
            return False
        return self.check_cross_sectional_requirements(
            section_idx,
            measure_idx,
            cross_sectional_requirements,
            design_year,
            mechanisms,
        )

    def get_valid_measures_for_section(
        self,
        section_idx: int,
        cross_sectional_requirements: CrossSectionalRequirements,
    ) -> list[AggregatedMeasureCombination]:
        """Get the measures that satisfy the cross-sectional requirements for the mechanisms.

        Args:
            section_as_input (SectionAsInput): The SectionAsInput object for which the valid measures are to be found.
            cross_sectional_requirements (CrossSectionalRequirements): The cross-sectional requirements for the dike traject.

        Returns:
            list[AggregatedMeasureCombination]: The list of valid measures for the mechanisms.
        """
        # get the first possible investment year from the aggregated measures
        _section_as_input = self.sections[section_idx]
        if not _section_as_input.aggregated_measure_combinations:
            return []
        _invest_year = min(
            [
                measure.year
                for measure in _section_as_input.aggregated_measure_combinations
            ]
        )
        _design_horizon_year = _invest_year + self.OI_horizon

        _valid_measures = [
            _measure
            for measure_idx, _measure in enumerate(
                _section_as_input.aggregated_measure_combinations
            )
            if self.check_measure_validity(
                measure_idx,
                section_idx,
                _section_as_input.mechanisms,
                cross_sectional_requirements,
                _invest_year,
                _design_horizon_year,
            )
        ]

        return _valid_measures

    def get_best_measure_for_section(
        self,
        section_idx: int,
        cross_sectional_requirements: CrossSectionalRequirements,
    ) -> tuple[list[AggregatedMeasureCombination], list[MechanismEnum]]:
        """Get the measure with the lowest failure probability for the mechanisms that do not satisfy the cross-sectional requirements.

        Args:
            section_as_input (SectionAsInput): The SectionAsInput object for which the best measure is to be found.
            cross_sectional_requirements (CrossSectionalRequirements): The cross-sectional requirements for the dike traject.

        Returns:
            list[AggregatedMeasureCombination]: The list of best measures for the mechanisms that do not satisfy the cross-sectional requirements.
            list[MechanismEnum]: The list of mechanisms that do not satisfy the cross-sectional requirements.
        """
        _section_as_input = self.sections[section_idx]
        if not _section_as_input.aggregated_measure_combinations:
            return [], []

        # get the first possible investment year from the aggregated measures
        _invest_year = min(
            measure.year
            for measure in _section_as_input.aggregated_measure_combinations
        )
        _design_horizon_year = _invest_year + self.OI_horizon

        # for each mechanism we check if the cross-sectional requirements are met
        # initialize a dictionary
        _requirement_met_per_mechanism = defaultdict(lambda: False)

        # loop over all mechanisms and check if the requirements are met. Once they are met, set the value to True and break the loop
        for mechanism in _section_as_input.mechanisms:
            _requirement_met_per_mechanism[mechanism] = False
            for _measure_idx, _ in enumerate(
                _section_as_input.aggregated_measure_combinations
            ):
                if self.check_cross_sectional_requirements(
                    section_idx,
                    _measure_idx,
                    cross_sectional_requirements,
                    _design_horizon_year,
                    [mechanism],
                ):
                    _requirement_met_per_mechanism[mechanism] = True
                    break

        # next we get the mechanisms in _requirement_met_per_mechanism where values are True
        _valid_mechanisms = [
            mechanism
            for mechanism, value in _requirement_met_per_mechanism.items()
            if value
        ]

        # get the valid measures and corresponding idx
        _valid_measures_with_idx = [
            (_measure_idx, _measure)
            for _measure_idx, _measure in enumerate(
                _section_as_input.aggregated_measure_combinations
            )
            if self.check_measure_validity(
                _measure_idx,
                section_idx,
                _valid_mechanisms,
                cross_sectional_requirements,
                _invest_year,
                _design_horizon_year,
            )
        ]

        # Unpack the list of tuples into two separate lists
        _valid_measure_idx, _valid_measures = (
            zip(*_valid_measures_with_idx) if _valid_measures_with_idx else ([], [])
        )

        # get the mechanisms in _requirement_met_per_mechanism where values are False
        _invalid_mechanisms = [
            mechanism
            for mechanism, value in _requirement_met_per_mechanism.items()
            if not value
        ]
        # get the failure probabilities for the mechanisms in _invalid_mechanisms for all _valid_measures
        _failure_probabilities = [
            self._get_failure_probability_of_invalid_mechanisms(
                section_idx, _measure_idx, _design_horizon_year, _invalid_mechanisms
            )
            for _measure_idx in _valid_measure_idx
        ]

        # remove measures with a pf that is too high
        _valid_measures_low_prob = [
            _measure
            for _idx, _measure in enumerate(_valid_measures)
            if math.isclose(
                _failure_probabilities[_idx], min(_failure_probabilities), rel_tol=1e-9
            )
        ]

        # take one with lowest cost
        _valid_measure_lccs = [_measure.lcc for _measure in _valid_measures_low_prob]

        # filter further based on cost: only take measures that are cheapest
        _valid_measures_low_prob_cost = [
            _measure
            for _idx, _measure in enumerate(_valid_measures_low_prob)
            if math.isclose(_measure.lcc, min(_valid_measure_lccs))
        ]

        # return the first as they have the same cost and pf and are not distinctive
        return [_valid_measures_low_prob_cost[0]], _invalid_mechanisms

    @staticmethod
    def get_cross_sectional_requirements_for_target_reliability(
        dike_traject: DikeTraject, section_as_input: SectionAsInput
    ) -> CrossSectionalRequirements:
        """
        Class method to create a CrossSectionalRequirements object from a DikeTraject object.
        This method calculates the cross-sectional requirements for the dike traject based on the OI2014 approach.
        The cross-sectional requirements are calculated for each mechanism and stored in a dictionary with the mechanism as key and the cross-sectional requirements as value.

        Args:
            dike_traject (DikeTraject): The DikeTraject object for which the cross-sectional requirements are to be calculated.
            section_as_input (SectionAsInput): The section with the specific requirements to be applied for cross sectional computations.

        Returns:
            CrossSectionalRequirements: The CrossSectionalRequirements object with the cross-sectional requirements for the dike traject.
        """
        # compute cross sectional requirements
        n_piping = 1 + (
            dike_traject.general_info.aPiping
            * dike_traject.general_info.TrajectLength
            / dike_traject.general_info.bPiping
        )
        n_stab = 1 + (
            dike_traject.general_info.aStabilityInner
            * dike_traject.general_info.TrajectLength
            / dike_traject.general_info.bStabilityInner
        )
        n_overflow = 1
        n_revetment = 3
        omegaRevetment = 0.1

        _pf_cs_piping = (
            dike_traject.general_info.Pmax
            * dike_traject.general_info.omegaPiping
            / n_piping
        )
        _pf_cs_revetment = dike_traject.general_info.Pmax * omegaRevetment / n_revetment
        _pf_cs_stabinner = (
            dike_traject.general_info.Pmax
            * dike_traject.general_info.omegaStabilityInner
            / n_stab
        )
        _pf_cs_overflow = (
            dike_traject.general_info.Pmax
            * dike_traject.general_info.omegaOverflow
            / n_overflow
        )
        return CrossSectionalRequirements(
            cross_sectional_requirement_per_mechanism={
                MechanismEnum.PIPING: _pf_cs_piping,
                MechanismEnum.STABILITY_INNER: _pf_cs_stabinner,
                MechanismEnum.OVERFLOW: _pf_cs_overflow,
                MechanismEnum.REVETMENT: _pf_cs_revetment,
            },
            dike_traject_b_piping=dike_traject.general_info.bPiping,
            dike_traject_b_stability_inner=dike_traject.general_info.bStabilityInner,
            dike_section_a_piping=section_as_input.a_section_piping,
            dike_section_a_stability_inner=section_as_input.a_section_stability_inner,
            dike_section_length=section_as_input.section_length,
        )

    def evaluate(
        self,
        dike_traject: DikeTraject,
    ) -> None:
        """Evaluate the strategy for the given dike traject.
        This evaluates the target reliability of different measures.
        The general idea is that for a given design horizon the cross-sectional requirements are met for each section.
        If the requirements are not met, we select the measure that fits most cross-sectional requirements, and has the lowest failure probability for the mechanisms that are not satisfied by the measure.
        If the latter happens, a warning is logged.

        Args:
            dike_traject (DikeTraject): The DikeTraject object for which the strategy is to be evaluated.
        """
        # Get initial failure probabilities at design horizon. #TODO think about what year is to be used here.
        initial_section_pfs = [
            section.initial_assessment.get_section_probability(self.OI_horizon)
            for section in self.sections
        ]

        # Rank sections based on initial probability
        section_order = np.flip(np.argsort(initial_section_pfs))

        self.initial_step = StrategyStep(
            step_number=0,
            step_type=step_type_enum.StepTypeEnum.TARGET,
            total_risk=self.traject_risk.get_total_risk(),
            probabilities=self.traject_risk.get_initial_probabilities_dict(
                self.traject_risk.mechanisms
            ),
        )

        for _section_idx in section_order:
            # get the first possible investment year from the aggregated measures
            _cross_sectional_requirements = (
                self.get_cross_sectional_requirements_for_target_reliability(
                    dike_traject, self.sections[_section_idx]
                )
            )
            _valid_measures = self.get_valid_measures_for_section(
                _section_idx, _cross_sectional_requirements
            )

            if len(_valid_measures) == 0:
                # if no measures satisfy the requirements, get the measure that is best for the mechanisms that do not satisfy the requirements
                (
                    _valid_measures,
                    _invalid_mechanisms,
                ) = self.get_best_measure_for_section(
                    _section_idx,
                    _cross_sectional_requirements,
                )
                # make a concatenated string of _invalid_mechanisms
                _invalid_mechanisms_str = " en ".join(
                    [mechanism.name.capitalize() for mechanism in _invalid_mechanisms]
                )

                logging.warning(
                    "Geen maatregelen gevonden die voldoen aan doorsnede-eisen op dijkvak %s.",
                    self.sections[_section_idx].section_name,
                )
                if not _valid_measures:
                    continue

                logging.warning(
                    "De beste maatregel is gekozen, maar deze voldoet niet aan de eisen voor %s.",
                    _invalid_mechanisms_str,
                )

            # get measure with lowest lcc from _valid_measures
            _lcc = [measure.lcc for measure in _valid_measures]
            idx = np.argmin(_lcc)
            _measure_idx = (_section_idx, *_valid_measures[idx].get_combination_idx())
            _measure = (_measure_idx[0], _measure_idx[1] + 1, _measure_idx[2] + 1)

            # update the probabilities for the mechanisms
            self.traject_risk.update_probabilities_for_measure(_measure)

            # add the optimization step
            self.optimization_steps.append(
                StrategyStep(
                    step_number=len(self.optimization_steps) + 1,
                    step_type=step_type_enum.StepTypeEnum.TARGET,
                    measure=_measure,
                    section_idx=_section_idx,
                    aggregated_measure=_valid_measures[idx],
                    probabilities=self.traject_risk.get_initial_probabilities_dict(
                        self.traject_risk.mechanisms
                    ),
                    total_risk=self.traject_risk.get_total_risk(),
                    total_cost=self.total_cost + _lcc[idx],
                )
            )
