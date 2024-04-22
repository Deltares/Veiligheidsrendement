import copy
import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.strategies.strategy_protocol import StrategyProtocol
from vrtool.decision_making.strategy_evaluation import (
    compute_total_risk,
    implement_option,
)
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.optimization.measures.aggregated_measures_combination import (
    AggregatedMeasureCombination,
)
from vrtool.optimization.strategy_input.strategy_input import StrategyInput


@dataclass
class CrossSectionalRequirements:
    cross_sectional_requirement_per_mechanism: dict[MechanismEnum, np.ndarray]

    dike_traject_b_piping: float
    dike_traject_b_stability_inner: float

    @classmethod
    def from_dike_traject(cls, dike_traject: DikeTraject):
        """Class method to create a CrossSectionalRequirements object from a DikeTraject object.
        This method calculates the cross-sectional requirements for the dike traject based on the OI2014 approach.
        The cross-sectional requirements are calculated for each mechanism and stored in a dictionary with the mechanism as key and the cross-sectional requirements as value.

        Args:
            dike_traject (DikeTraject): The DikeTraject object for which the cross-sectional requirements are to be calculated.

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
        return cls(
            cross_sectional_requirement_per_mechanism={
                MechanismEnum.PIPING: _pf_cs_piping,
                MechanismEnum.STABILITY_INNER: _pf_cs_stabinner,
                MechanismEnum.OVERFLOW: _pf_cs_overflow,
                MechanismEnum.REVETMENT: _pf_cs_revetment,
            },
            dike_traject_b_piping=dike_traject.general_info.bPiping,
            dike_traject_b_stability_inner=dike_traject.general_info.bStabilityInner,
        )


class TargetReliabilityStrategy(StrategyProtocol):
    """Subclass for evaluation in accordance with basic OI2014 approach.
    This ensures that for a certain time horizon, each section satisfies the cross-sectional target reliability
    """

    def __init__(self, strategy_input: StrategyInput, config: VrtoolConfig):
        # Necessary config parameters:
        self.OI_horizon = config.OI_horizon
        self.time_periods = config.T
        # New mappings
        self.Pf = strategy_input.Pf
        self.D = strategy_input.D
        self.sections = strategy_input.sections

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
        _measure = self.sections[section_idx].aggregated_measure_combinations[
            measure_idx
        ]

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
                if (
                    _measure.sg_combination.mechanism_year_collection.get_probabilities(
                        _mechanism, [year]
                    )[0]
                    > cross_sectional_requirements.cross_sectional_requirement_per_mechanism[
                        _mechanism
                    ]
                ):
                    return False
        return True

    def _get_failure_probability_of_invalid_mechanisms(
        self,
        section_idx: int,
        measure_idx: int,
        year: int,
        mechanisms: list[MechanismEnum],
    ) -> bool:
        """This function gets the failure probabilities of the mechanisms that are not satisfied by the measure and returns the total failure probability.

        Args:
            measure (AggregatedMeasureCombination): The measure for which the failure probability is to be calculated.
            year (int): The year for which the failure probability is to be calculated.
            mechanisms (list[MechanismEnum]): The mechanisms for which the failure probability is to be calculated.

            Returns:
                float: The total failure probability for the mechanisms that are not satisfied by the measure.
        """
        _measure = self.sections[section_idx].aggregated_measure_combinations[
            measure_idx
        ]
        _p_nonf = 1
        for _mechanism in mechanisms:
            if _mechanism in [MechanismEnum.OVERFLOW, MechanismEnum.REVETMENT]:
                # look in sh, if any mechanism is not satisfied, return a False
                _p_nonf = (
                    _p_nonf
                    * _measure.sh_combination.mechanism_year_collection.get_probabilities(
                        _mechanism, [year]
                    )[
                        0
                    ]
                )
            elif _mechanism in [MechanismEnum.PIPING, MechanismEnum.STABILITY_INNER]:
                _p_nonf = (
                    _p_nonf
                    * _measure.sg_combination.mechanism_year_collection.get_probabilities(
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
        section_idx: float,
        cross_sectional_requirements: CrossSectionalRequirements,
    ) -> AggregatedMeasureCombination:
        """Get the measures that satisfy the cross-sectional requirements for the mechanisms.

        Args:
            section_as_input (SectionAsInput): The SectionAsInput object for which the valid measures are to be found.
            cross_sectional_requirements (CrossSectionalRequirements): The cross-sectional requirements for the dike traject.

        Returns:
            list[AggregatedMeasureCombination]: The list of valid measures for the mechanisms.
        """
        # get the first possible investment year from the aggregated measures
        _section_as_input = self.sections[section_idx]
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
    ) -> AggregatedMeasureCombination:
        """Get the measure with the lowest failure probability for the mechanisms that do not satisfy the cross-sectional requirements.

        Args:
            section_as_input (SectionAsInput): The SectionAsInput object for which the best measure is to be found.
            cross_sectional_requirements (CrossSectionalRequirements): The cross-sectional requirements for the dike traject.

        Returns:
            list[AggregatedMeasureCombination]: The list of best measures for the mechanisms that do not satisfy the cross-sectional requirements.
        """
        _section_as_input = self.sections[section_idx]
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
            _requirement_met_per_mechanism[mechanism]

        # next we get the mechanisms in _requirement_met_per_mechanism where values are True
        _valid_mechanisms = [
            mechanism
            for mechanism, value in _requirement_met_per_mechanism.items()
            if value
        ]

        # get the measures that are valid for the _valid_mechanisms
        _valid_measures = [
            _measure
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
            for _measure_idx, _ in enumerate(_valid_measures)
        ]

        # get the measure with the lowest failure probability for the mechanisms in _invalid_mechanisms
        return [_valid_measures[np.argmin(_failure_probabilities)]], _invalid_mechanisms

    def evaluate(
        self,
        dike_traject: DikeTraject,
    ):
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

        # get the cross-sectional requirements for the dike traject (probability)
        _cross_sectional_requirements = CrossSectionalRequirements.from_dike_traject(
            dike_traject
        )
        # and the risk for each step
        _taken_measures = {}
        _taken_measures_indices = []
        for _section_idx in section_order:
            # add probability for this step:

            # get the first possible investment year from the aggregated measures
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
                    f"Geen maatregelen gevonden die voldoen aan doorsnede-eisen op dijkvak {self.sections[_section_idx].section_name}. De beste maatregel is gekozen, maar deze voldoet niet aan de eisen voor {_invalid_mechanisms_str}."
                )

            # get measure with lowest lcc from _valid_measures
            _lcc = [measure.lcc for measure in _valid_measures]
            idx = np.argmin(_lcc)
            _taken_measures[self.sections[_section_idx].section_name] = _valid_measures[
                idx
            ]
            measure_idx = self.sections[_section_idx].get_combination_idx_for_aggregate(
                _taken_measures[self.sections[_section_idx].section_name]
            )
            _taken_measures_indices.append(
                (_section_idx, measure_idx[0] + 1, measure_idx[1] + 1)
            )

        # For output we need to give the list of measure indices, the total_risk per step, and the probabilities per step
        # First we get, and update the probabilities per step
        # we need to track probability for each step
        init_probability = {mech: self.Pf[mech][:, 0, :] for mech in self.Pf.keys()}
        self.probabilities_per_step = [copy.deepcopy(init_probability)]
        self.total_risk_per_step = [
            compute_total_risk(self.probabilities_per_step[-1], self.D)
        ]

        for step in range(0, len(_taken_measures)):
            section_id = _taken_measures_indices[step][0]
            self.probabilities_per_step.append(
                copy.deepcopy(self.probabilities_per_step[-1])
            )
            self.probabilities_per_step[-1] = implement_option(
                self.probabilities_per_step[-1],
                _taken_measures_indices[step],
                _taken_measures[self.sections[section_id].section_name],
            )
            self.total_risk_per_step.append(
                compute_total_risk(self.probabilities_per_step[-1], self.D)
            )

        self.measures_taken = _taken_measures_indices
